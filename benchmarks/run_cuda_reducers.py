from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import maxsim
from maxsim.experimental import int4_maxsim, int4_to_device, pack_int4_symmetric

try:
    import torch
except ImportError:  # pragma: no cover - optional CUDA benchmark dependency
    torch = None


REDUCERS = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")


def run_benchmark(
    *,
    output_path: Path | None = None,
    repeat: int = 20,
    warmup: int = 5,
    batch: int = 4,
    query_tokens: int = 16,
    docs: int = 256,
    min_doc_tokens: int = 4,
    max_doc_tokens: int = 32,
    candidates: int = 32,
    dim: int = 128,
    seed: int = 20260713,
    temperature: float = 0.7,
    shared_reducer_warps: int | None = None,
) -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA reducer benchmarks require torch with CUDA")
    try:
        cuda_extension = importlib.import_module("maxsim._maxsim_cuda")
    except ImportError as exc:
        raise RuntimeError("CUDA reducer benchmarks require a CUDA-enabled maxsim build") from exc
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be >= 1 and warmup must be >= 0")
    if batch < 1 or query_tokens < 1 or docs < 1 or dim < 1:
        raise ValueError("batch, query_tokens, docs, and dim must be >= 1")
    if dim % 8 != 0:
        raise ValueError("dim must be divisible by 8")
    if min_doc_tokens < 1 or max_doc_tokens < min_doc_tokens:
        raise ValueError("doc token bounds must satisfy 1 <= min <= max")
    if candidates < 1 or candidates > docs:
        raise ValueError("candidates must be between 1 and docs")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    warp_configuration = _configure_shared_reducer_warps(cuda_extension, shared_reducer_warps)

    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_doc_tokens, max_doc_tokens + 1, size=docs, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    doc_embeddings = rng.normal(size=(int(offsets[-1]), dim)).astype(np.float32)
    query = rng.normal(size=(batch, query_tokens, dim)).astype(np.float32)
    query_weights = rng.uniform(0.0, 2.0, size=(batch, query_tokens)).astype(np.float32)
    candidate_indices = np.ascontiguousarray(rng.choice(docs, size=candidates, replace=False), dtype=np.int64)

    binary_cpu = maxsim.pack_signs(doc_embeddings, offsets)
    binary = maxsim.to_device(binary_cpu, "cuda")
    binary_docs = np.where(doc_embeddings >= 0.0, 1.0, -1.0).astype(np.float32)

    int4_cpu = pack_int4_symmetric(doc_embeddings, offsets)
    int4 = int4_to_device(int4_cpu)
    int4_docs = int4_cpu.values.astype(np.float32) * np.float32(int4_cpu.scale)

    formats = {
        "binary": (binary, binary_docs),
        "int4": (int4, int4_docs),
    }
    references: dict[tuple[str, str, str], np.ndarray] = {}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for format_name, (_packed, reconstructed) in formats.items():
            for reducer in REDUCERS:
                kwargs = _reducer_kwargs(reducer, query_weights, temperature)
                references[(format_name, "full", reducer)] = _torch_reference(
                    query,
                    reconstructed,
                    offsets,
                    reducer=reducer,
                    **kwargs,
                )
                references[(format_name, "candidates", reducer)] = _torch_reference(
                    query,
                    reconstructed,
                    offsets,
                    reducer=reducer,
                    candidate_indices=candidate_indices,
                    **kwargs,
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    rows: list[dict[str, Any]] = []
    legacy_outputs: dict[str, np.ndarray] = {}
    for format_name, (packed, _reconstructed) in formats.items():
        legacy_call: Callable[[], np.ndarray]
        if format_name == "binary":
            legacy_call = lambda packed=packed: maxsim.maxsim(query, packed, device="cuda")
        else:
            legacy_call = lambda packed=packed: int4_maxsim(query, packed, device="cuda")
        legacy_row, legacy_output = _measure_case(
            legacy_call,
            references[(format_name, "full", "maxsim")],
            repeat=repeat,
            warmup=warmup,
        )
        legacy_outputs[format_name] = legacy_output
        rows.append(
            {
                "format": format_name,
                "scope": "full",
                "reducer": "maxsim",
                "implementation": "legacy_maxsim",
                **legacy_row,
            }
        )

        for scope, selected in (("full", None), ("candidates", candidate_indices)):
            for reducer in REDUCERS:
                kwargs = _reducer_kwargs(reducer, query_weights, temperature)

                def call(
                    packed=packed,
                    reducer=reducer,
                    selected=selected,
                    kwargs=kwargs,
                ):
                    return maxsim.score(
                        query,
                        packed,
                        reducer=reducer,
                        candidate_indices=selected,
                        device="cuda",
                        **kwargs,
                    )

                row, output = _measure_case(
                    call,
                    references[(format_name, scope, reducer)],
                    repeat=repeat,
                    warmup=warmup,
                )
                if scope == "full" and reducer == "maxsim":
                    row["max_abs_delta_vs_legacy"] = _max_abs_delta(output, legacy_outputs[format_name])
                rows.append(
                    {
                        "format": format_name,
                        "scope": scope,
                        "reducer": reducer,
                        "implementation": "shared_reducer_policy",
                        **row,
                    }
                )

    gate_passed = all(row["finite"] and row["max_abs_delta_vs_torch"] <= _tolerance(row) for row in rows)
    result = {
        "schema_version": 1,
        "benchmark": "compressed_cuda_reducers",
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "batch": batch,
            "query_tokens": query_tokens,
            "docs": docs,
            "total_doc_tokens": int(offsets[-1]),
            "min_doc_tokens": int(lengths.min()),
            "max_doc_tokens": int(lengths.max()),
            "candidate_docs": candidates,
            "dim": dim,
        },
        "temperature": float(temperature),
        "shared_reducer_warps": warp_configuration,
        "torch_reference_tf32": False,
        "candidate_indices": candidate_indices.tolist(),
        "topksim_short_doc_semantics": "mean over min(k, doc_tokens); empty documents score zero",
        "gate_passed": gate_passed,
        "summary": _summarize_rows(rows),
        "rows": rows,
    }

    output = output_path or Path("benchmark-results/cuda-reducers.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _reducer_kwargs(reducer: str, query_weights: np.ndarray, temperature: float) -> dict[str, Any]:
    if reducer == "weighted_maxsim":
        return {"query_weights": query_weights}
    if reducer == "smoothsim":
        return {"temperature": temperature}
    return {}


def _configure_shared_reducer_warps(cuda_extension: Any, requested: int | None) -> dict[str, Any]:
    """Select a shared-reducer launch variant when supported by the extension."""
    if requested is not None and requested not in (0, 4, 8):
        raise ValueError("shared_reducer_warps must be one of 0, 4, or 8")

    setter = getattr(cuda_extension, "set_shared_reducer_warps", None)
    getter = getattr(cuda_extension, "get_shared_reducer_warps", None)
    supported = callable(setter)
    if supported and requested is not None:
        setter(requested)
    active = getter() if callable(getter) else (requested if supported and requested is not None else None)
    return {
        "requested": requested,
        "active": None if active is None else int(active),
        "supported": supported,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build median-latency comparisons whose direction is explicit in the key."""
    latency = {
        (row["format"], row["scope"], row["reducer"], row["implementation"]): float(
            row["latency_median_ms"]
        )
        for row in rows
    }
    formats = sorted({row["format"] for row in rows})
    reducers = sorted(
        {
            row["reducer"]
            for row in rows
            if row["implementation"] == "shared_reducer_policy"
        }
    )

    shared_vs_legacy = {}
    candidate_vs_full = {}
    for format_name in formats:
        legacy = latency.get((format_name, "full", "maxsim", "legacy_maxsim"))
        shared = latency.get((format_name, "full", "maxsim", "shared_reducer_policy"))
        if legacy is not None and shared is not None:
            shared_vs_legacy[format_name] = shared / legacy

        reducer_speedups = {}
        for reducer in reducers:
            full = latency.get((format_name, "full", reducer, "shared_reducer_policy"))
            candidate = latency.get((format_name, "candidates", reducer, "shared_reducer_policy"))
            if full is not None and candidate is not None:
                reducer_speedups[reducer] = full / candidate
        if reducer_speedups:
            candidate_vs_full[format_name] = reducer_speedups

    return {
        "latency_basis": "median_ms",
        "shared_vs_legacy_maxsim_ratio": shared_vs_legacy,
        "shared_vs_legacy_ratio_interpretation": "below 1 means the shared reducer is faster",
        "candidate_vs_full_speedup": candidate_vs_full,
        "candidate_vs_full_speedup_interpretation": "above 1 means candidate-only scoring is faster",
    }


def _torch_reference(
    query: np.ndarray,
    docs: np.ndarray,
    offsets: np.ndarray,
    *,
    reducer: str,
    query_weights: np.ndarray | None = None,
    temperature: float = 1.0,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    query_tensor = torch.as_tensor(query, device="cuda", dtype=torch.float32)
    docs_tensor = torch.as_tensor(docs, device="cuda", dtype=torch.float32)
    weights = None if query_weights is None else torch.as_tensor(query_weights, device="cuda", dtype=torch.float32)
    doc_indices = np.arange(offsets.shape[0] - 1, dtype=np.int64) if candidate_indices is None else candidate_indices

    columns = []
    for doc_idx in doc_indices:
        start = int(offsets[int(doc_idx)])
        end = int(offsets[int(doc_idx) + 1])
        similarities = torch.einsum("bqd,td->bqt", query_tensor, docs_tensor[start:end])
        if reducer == "maxsim":
            per_query = similarities.amax(dim=-1)
        elif reducer == "weighted_maxsim":
            per_query = similarities.amax(dim=-1) * weights
        elif reducer in {"topk2", "topk4"}:
            requested_k = 2 if reducer == "topk2" else 4
            actual_k = min(requested_k, end - start)
            per_query = similarities.topk(actual_k, dim=-1).values.mean(dim=-1)
        elif reducer == "smoothsim":
            per_query = float(temperature) * torch.logsumexp(similarities / float(temperature), dim=-1)
        else:  # pragma: no cover - constant list controls callers
            raise AssertionError(f"unknown reducer: {reducer}")
        columns.append(per_query.sum(dim=-1))
    return torch.stack(columns, dim=-1).cpu().numpy().astype(np.float32, copy=False)


def _measure_case(call: Callable[[], np.ndarray], reference: np.ndarray, *, repeat: int, warmup: int):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    timings = []
    output = None
    for _ in range(repeat):
        start = time.perf_counter()
        output = call()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(output, dtype=np.float32)
    reference_values = np.asarray(reference, dtype=np.float32)
    delta = np.abs(values - reference_values)
    row = {
        "latency_best_ms": float(np.min(timings)),
        "latency_median_ms": float(np.median(timings)),
        "max_abs_delta_vs_torch": float(delta.max(initial=0.0)),
        "mean_abs_delta_vs_torch": float(delta.mean()) if delta.size else 0.0,
        "finite": bool(np.isfinite(values).all()),
        "output_shape": list(values.shape),
    }
    return row, values


def _max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)), initial=0.0))


def _tolerance(row: dict[str, Any]) -> float:
    if row["format"] == "int4":
        return 5e-3 if row["reducer"] == "smoothsim" else 2e-3
    return 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark shared CUDA reducer correctness and latency.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--query-tokens", type=int, default=16)
    parser.add_argument("--docs", type=int, default=256)
    parser.add_argument("--min-doc-tokens", type=int, default=4)
    parser.add_argument("--max-doc-tokens", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--shared-reducer-warps",
        type=int,
        choices=(0, 4, 8),
        help="shared reducer launch variant: 0=old block baseline, or a fixed 4/8 warps per block",
    )
    args = parser.parse_args()

    result = run_benchmark(
        output_path=args.output,
        repeat=args.repeat,
        warmup=args.warmup,
        batch=args.batch,
        query_tokens=args.query_tokens,
        docs=args.docs,
        min_doc_tokens=args.min_doc_tokens,
        max_doc_tokens=args.max_doc_tokens,
        candidates=args.candidates,
        dim=args.dim,
        seed=args.seed,
        temperature=args.temperature,
        shared_reducer_warps=args.shared_reducer_warps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
