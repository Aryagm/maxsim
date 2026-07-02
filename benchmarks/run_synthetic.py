from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import bitmax

try:
    import torch as _torch
except ImportError:  # pragma: no cover - depends on optional local install
    _torch = None


def run_stage(stage: str, *, output_path: Path | str | None = None, gate_path: Path | str | None = None) -> dict[str, Any]:
    output = Path(output_path) if output_path is not None else Path("benchmark-results") / f"{stage}.json"
    _check_stage_gate(stage, gate_path)
    specs, repeat = _stage_config(stage)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        query, docs, offsets = _make_fixture(spec)
        packed = bitmax.pack_signs(docs, offsets)
        reference_scores, reference_latency = _time_call(lambda: _python_reference_maxsim(query, packed), repeat=repeat)
        native_device, baseline_device = _benchmark_devices(stage)
        fp16_runner, fp16_metadata = _dense_baseline_runner(query, packed, storage_dtype=np.float16, device=baseline_device)
        int8_runner, int8_metadata = _dense_baseline_runner(query, packed, storage_dtype=np.int8, device=baseline_device)
        fp16_scores, fp16_latency = _time_call(fp16_runner, repeat=repeat)
        int8_scores, int8_latency = _time_call(int8_runner, repeat=repeat)
        native_name = "bitmax_cuda" if native_device == "cuda" else "bitmax_native"
        native_scores, native_latency = _time_call(lambda: bitmax.maxsim(query, packed, device=native_device), repeat=repeat)

        fp16_row = _row(
            stage,
            spec,
            "torch_fp16_baseline",
            fp16_latency,
            fp16_scores,
            reference_scores,
            doc_storage_bytes=_dense_doc_bytes(spec, 2),
            metadata=fp16_metadata,
        )
        int8_row = _row(
            stage,
            spec,
            "torch_int8_baseline",
            int8_latency,
            int8_scores,
            reference_scores,
            doc_storage_bytes=_dense_doc_bytes(spec, 1),
            metadata=int8_metadata,
        )
        rows.append(_row(stage, spec, "python_reference", reference_latency, reference_scores, reference_scores, doc_storage_bytes=_packed_doc_bytes(spec)))
        rows.append(fp16_row)
        rows.append(int8_row)
        rows.append(
            _row(
                stage,
                spec,
                native_name,
                native_latency,
                native_scores,
                reference_scores,
                doc_storage_bytes=_packed_doc_bytes(spec),
                baseline_rows=[fp16_row, int8_row],
            )
        )

    result = {
        "schema_version": 2,
        "stage": stage,
        "baselines": ["python_reference", "torch_fp16_baseline", "torch_int8_baseline"],
        "gate_passed": _gate_passed(rows),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _check_stage_gate(stage: str, gate_path: Path | str | None) -> None:
    required_stage = {
        "cuda-sweep": "cuda-smoke",
        "vast-large": "cuda-sweep",
    }.get(stage)
    if required_stage is None:
        return
    if gate_path is None:
        raise RuntimeError(f"{stage} requires a passing {required_stage} gate JSON")

    gate = Path(gate_path)
    if not gate.exists():
        raise RuntimeError(f"{stage} requires a passing {required_stage} gate JSON")

    data = json.loads(gate.read_text())
    if data.get("stage") != required_stage or data.get("gate_passed") is not True:
        raise RuntimeError(f"{stage} requires a passing {required_stage} gate JSON")


def _stage_config(stage: str) -> tuple[list[dict[str, Any]], int]:
    if stage == "stage0":
        return [
            {"dim": 8, "query_tokens": 2, "doc_tokens": 2, "docs": 3, "dtype": "int8"},
            {"dim": 128, "query_tokens": 4, "doc_tokens": 3, "docs": 3, "dtype": "float32"},
        ], 3
    if stage == "cpu-smoke":
        return [
            {"dim": 128, "query_tokens": 8, "doc_tokens": 16, "docs": 32, "dtype": "int8"},
            {"dim": 256, "query_tokens": 16, "doc_tokens": 32, "docs": 64, "dtype": "float32"},
        ], 5
    if stage == "cuda-smoke":
        return [
            {"dim": 128, "query_tokens": 8, "doc_tokens": 16, "docs": 64, "dtype": "float32"},
        ], 5
    if stage == "cuda-sweep":
        return [
            {"dim": 128, "query_tokens": 16, "doc_tokens": 32, "docs": 512, "dtype": "int8"},
            {"dim": 128, "query_tokens": 32, "doc_tokens": 64, "docs": 1_024, "dtype": "float32"},
            {"dim": 256, "query_tokens": 32, "doc_tokens": 64, "docs": 1_024, "dtype": "float32"},
        ], 7
    if stage == "vast-large":
        return [
            {"dim": 128, "query_tokens": 16, "doc_tokens": 64, "docs": 1_024, "dtype": "int8"},
            {"dim": 256, "query_tokens": 32, "doc_tokens": 128, "docs": 2_048, "dtype": "float32"},
        ], 7
    raise ValueError(f"unknown benchmark stage: {stage}")


def _benchmark_devices(stage: str) -> tuple[str, str]:
    if stage in {"cuda-smoke", "cuda-sweep", "vast-large"}:
        return "cuda", "cuda"
    return "auto", "cpu"


def _make_fixture(spec: dict[str, Any]):
    rng = np.random.default_rng(20260630 + spec["dim"] + spec["docs"])
    total_doc_tokens = spec["doc_tokens"] * spec["docs"]
    docs = rng.normal(size=(total_doc_tokens, spec["dim"])).astype(np.float32)
    offsets = np.arange(0, total_doc_tokens + 1, spec["doc_tokens"], dtype=np.int64)
    if spec["dtype"] == "int8":
        query = rng.integers(-8, 9, size=(spec["query_tokens"], spec["dim"]), dtype=np.int8)
    else:
        query = rng.normal(size=(spec["query_tokens"], spec["dim"])).astype(np.float32)
    return query, docs, offsets


def _time_call(fn, *, repeat: int):
    best_latency = float("inf")
    best_value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        latency_ms = (time.perf_counter() - start) * 1_000.0
        if latency_ms < best_latency:
            best_latency = latency_ms
            best_value = value
    return best_value, best_latency


def _python_reference_maxsim(query, packed: bitmax.PackedDocs):
    query_float = np.asarray(query, dtype=np.float32)
    signs = _unpack_signs(packed.data, packed.dim)
    scores = np.empty(packed.num_docs, dtype=np.float32)
    scale = 1.0 if packed.scale is None else float(packed.scale)
    for doc_idx in range(packed.num_docs):
        start = int(packed.doc_offsets[doc_idx])
        end = int(packed.doc_offsets[doc_idx + 1])
        doc = signs[start:end]
        scores[doc_idx] = np.max(query_float @ doc.T, axis=1).sum(dtype=np.float32) * scale
    return scores


def _dense_baseline_runner(query, packed: bitmax.PackedDocs, *, storage_dtype, device: str):
    formula = "dense_fp16_vectorized_maxsim" if storage_dtype == np.float16 else "dense_int8_vectorized_doc_maxsim"
    torch_device = _resolve_torch_device(device)
    backend = "torch" if torch_device is not None else "numpy_torch_equivalent"
    metadata = {
        "baseline_backend": backend,
        "baseline_device": str(torch_device) if torch_device is not None else "cpu",
        "requested_baseline_device": device,
        "formula": formula,
    }
    if torch_device is not None:
        return lambda: _torch_dense_baseline_maxsim(query, packed, storage_dtype=storage_dtype, device=torch_device), metadata
    return lambda: _numpy_dense_baseline_maxsim(query, packed, storage_dtype=storage_dtype), metadata


def _resolve_torch_device(requested: str):
    if _torch is None:
        return None
    if requested == "cuda":
        if _torch.cuda.is_available():
            if _torch_cuda_supports_current_device():
                return _torch.device("cuda")
        return None
    return _torch.device("cpu")


def _torch_cuda_supports_current_device() -> bool:
    try:
        major, minor = _torch.cuda.get_device_capability()
        supported_arches = set(_torch.cuda.get_arch_list())
    except Exception:
        return False
    if not supported_arches:
        return True
    # CUDA SASS is forward-compatible within a major architecture: kernels
    # compiled for sm_86 run on sm_89, so wheels often omit the exact minor.
    for arch in supported_arches:
        if not arch.startswith("sm_"):
            continue
        try:
            value = int(arch[3:])
        except ValueError:
            continue
        arch_major, arch_minor = divmod(value, 10)
        if arch_major == major and arch_minor <= minor:
            return True
    return False


def _numpy_dense_baseline_maxsim(query, packed: bitmax.PackedDocs, *, storage_dtype) -> np.ndarray:
    query_float = np.asarray(query, dtype=storage_dtype).astype(np.float32)
    signs = _unpack_signs(packed.data, packed.dim).astype(storage_dtype).astype(np.float32)
    scale = 1.0 if packed.scale is None else float(packed.scale)
    doc_tokens = _uniform_doc_tokens(packed)
    if doc_tokens is None:
        scores = np.empty(packed.num_docs, dtype=np.float32)
        for doc_idx in range(packed.num_docs):
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            doc = signs[start:end]
            scores[doc_idx] = np.max(query_float @ doc.T, axis=1).sum(dtype=np.float32) * scale
        return scores
    dots = query_float @ signs.T
    return dots.reshape(query_float.shape[0], packed.num_docs, doc_tokens).max(axis=2).sum(axis=0, dtype=np.float32) * scale


def _torch_dense_baseline_maxsim(query, packed: bitmax.PackedDocs, *, storage_dtype, device) -> np.ndarray:
    torch_dtype = _torch.float16 if storage_dtype == np.float16 else _torch.int8
    query_tensor = _torch.as_tensor(np.asarray(query), dtype=torch_dtype, device=device)
    signs_tensor = _torch.as_tensor(_unpack_signs(packed.data, packed.dim), dtype=torch_dtype, device=device)
    scale = 1.0 if packed.scale is None else float(packed.scale)
    doc_tokens = _uniform_doc_tokens(packed)
    if doc_tokens is None:
        scores = []
        for doc_idx in range(packed.num_docs):
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            dots = query_tensor.to(_torch.float32) @ signs_tensor[start:end].to(_torch.float32).T
            scores.append(dots.max(dim=1).values.sum() * scale)
        result = _torch.stack(scores)
    else:
        dots = query_tensor.to(_torch.float32) @ signs_tensor.to(_torch.float32).T
        result = dots.reshape(query_tensor.shape[0], packed.num_docs, doc_tokens).max(dim=2).values.sum(dim=0) * scale
    if str(device) == "cuda":
        _torch.cuda.synchronize()
    return result.detach().cpu().numpy().astype(np.float32, copy=False)


def _uniform_doc_tokens(packed: bitmax.PackedDocs) -> int | None:
    lengths = np.diff(packed.doc_offsets)
    if lengths.size == 0 or np.any(lengths != lengths[0]):
        return None
    return int(lengths[0])


def _unpack_signs(data: np.ndarray, dim: int) -> np.ndarray:
    signs = np.empty((data.shape[0], dim), dtype=np.float32)
    for bit_idx in range(dim):
        bits = (data[:, bit_idx // 8] >> (bit_idx % 8)) & 1
        signs[:, bit_idx] = np.where(bits == 1, 1.0, -1.0)
    return signs


def _row(
    stage: str,
    spec: dict[str, Any],
    implementation: str,
    latency_ms: float,
    scores,
    reference_scores,
    *,
    doc_storage_bytes: int,
    baseline_rows: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
):
    delta = float(np.max(np.abs(np.asarray(scores, dtype=np.float32) - np.asarray(reference_scores, dtype=np.float32))))
    tolerance = _correctness_tolerance(spec, implementation)
    docs = int(spec["docs"])
    doc_tokens = int(spec["doc_tokens"])
    dim = int(spec["dim"])
    row = {
        "stage": stage,
        "implementation": implementation,
        "gate_blocking": implementation == "python_reference" or implementation.startswith("bitmax_"),
        "dtype": spec["dtype"],
        "dim": dim,
        "query_tokens": int(spec["query_tokens"]),
        "doc_tokens": doc_tokens,
        "docs": docs,
        "latency_ms": float(latency_ms),
        "docs_per_second": float(docs / max(latency_ms / 1_000.0, 1e-12)),
        "bytes_read": int(doc_storage_bytes),
        "doc_storage_bytes": int(doc_storage_bytes),
        "correctness_delta": delta,
        "correctness_tolerance": tolerance,
        "score_checksum": float(np.sum(scores, dtype=np.float64)),
    }
    if implementation.startswith("bitmax_") and baseline_rows is not None:
        baseline_latency = {row["implementation"]: row["latency_ms"] for row in baseline_rows}
        row["baseline_latency_ms"] = baseline_latency
        row["speedup_vs_torch_fp16"] = baseline_latency["torch_fp16_baseline"] / max(float(latency_ms), 1e-12)
        row["speedup_vs_torch_int8"] = baseline_latency["torch_int8_baseline"] / max(float(latency_ms), 1e-12)
        row["doc_memory_compression_vs_fp16"] = _dense_doc_bytes(spec, 2) / max(int(doc_storage_bytes), 1)
        row["doc_memory_compression_vs_fp32"] = _dense_doc_bytes(spec, 4) / max(int(doc_storage_bytes), 1)
    if metadata is not None:
        row.update(metadata)
    return row


def _packed_doc_bytes(spec: dict[str, Any]) -> int:
    return int(spec["docs"]) * int(spec["doc_tokens"]) * (int(spec["dim"]) // 8)


def _dense_doc_bytes(spec: dict[str, Any], bytes_per_value: int) -> int:
    return int(spec["docs"]) * int(spec["doc_tokens"]) * int(spec["dim"]) * bytes_per_value


def _correctness_tolerance(spec: dict[str, Any], implementation: str) -> float:
    if implementation == "torch_fp16_baseline":
        return 5e-1
    if spec["dtype"] == "int8":
        return 1e-5
    return 1e-3


def _gate_passed(rows: list[dict[str, Any]]) -> bool:
    return all(row["correctness_delta"] <= row["correctness_tolerance"] for row in rows if row["gate_blocking"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged synthetic bitmax benchmarks.")
    parser.add_argument("--stage", required=True, choices=["stage0", "cpu-smoke", "cuda-smoke", "cuda-sweep", "vast-large"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gate", type=Path, default=None)
    args = parser.parse_args()

    result = run_stage(args.stage, output_path=args.output, gate_path=args.gate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
