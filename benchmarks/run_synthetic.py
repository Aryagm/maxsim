from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import bitmax


def run_stage(stage: str, *, output_path: Path | str | None = None, gate_path: Path | str | None = None) -> dict[str, Any]:
    output = Path(output_path) if output_path is not None else Path("benchmark-results") / f"{stage}.json"
    _check_stage_gate(stage, gate_path)

    if stage == "stage0":
        specs = [
            {"dim": 8, "query_tokens": 2, "doc_tokens": 2, "docs": 3, "dtype": "int8"},
            {"dim": 128, "query_tokens": 4, "doc_tokens": 3, "docs": 3, "dtype": "float32"},
        ]
        repeat = 3
    elif stage == "cpu-smoke":
        specs = [
            {"dim": 128, "query_tokens": 8, "doc_tokens": 16, "docs": 32, "dtype": "int8"},
            {"dim": 256, "query_tokens": 16, "doc_tokens": 32, "docs": 64, "dtype": "float32"},
        ]
        repeat = 5
    elif stage == "cuda-smoke":
        specs = [
            {"dim": 128, "query_tokens": 8, "doc_tokens": 16, "docs": 64, "dtype": "float32"},
        ]
        repeat = 5
    elif stage == "vast-large":
        specs = [
            {"dim": 128, "query_tokens": 16, "doc_tokens": 64, "docs": 1_024, "dtype": "int8"},
            {"dim": 256, "query_tokens": 32, "doc_tokens": 128, "docs": 2_048, "dtype": "float32"},
        ]
        repeat = 7
    else:
        raise ValueError(f"unknown benchmark stage: {stage}")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        query, docs, offsets = _make_fixture(spec)
        packed = bitmax.pack_signs(docs, offsets)
        reference_scores, reference_latency = _time_call(lambda: _python_reference_maxsim(query, packed), repeat=repeat)
        native_device = "cuda" if stage == "cuda-smoke" else "auto"
        native_name = "bitmax_cuda" if native_device == "cuda" else "bitmax_native"
        native_scores, native_latency = _time_call(lambda: bitmax.maxsim(query, packed, device=native_device), repeat=repeat)

        rows.append(_row(stage, spec, "python_reference", reference_latency, reference_scores, reference_scores))
        rows.append(_row(stage, spec, native_name, native_latency, native_scores, reference_scores))

    result = {
        "stage": stage,
        "gate_passed": _gate_passed(rows),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _check_stage_gate(stage: str, gate_path: Path | str | None) -> None:
    if stage != "vast-large":
        return
    if gate_path is None:
        raise RuntimeError("vast-large requires a passing cuda-smoke gate JSON")

    gate = Path(gate_path)
    if not gate.exists():
        raise RuntimeError("vast-large requires a passing cuda-smoke gate JSON")

    data = json.loads(gate.read_text())
    if data.get("stage") != "cuda-smoke" or data.get("gate_passed") is not True:
        raise RuntimeError("vast-large requires a passing cuda-smoke gate JSON")


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


def _unpack_signs(data: np.ndarray, dim: int) -> np.ndarray:
    signs = np.empty((data.shape[0], dim), dtype=np.float32)
    for bit_idx in range(dim):
        bits = (data[:, bit_idx // 8] >> (bit_idx % 8)) & 1
        signs[:, bit_idx] = np.where(bits == 1, 1.0, -1.0)
    return signs


def _row(stage: str, spec: dict[str, Any], implementation: str, latency_ms: float, scores, reference_scores):
    delta = float(np.max(np.abs(np.asarray(scores, dtype=np.float32) - np.asarray(reference_scores, dtype=np.float32))))
    tolerance = 1e-5 if spec["dtype"] == "int8" else 1e-3
    docs = int(spec["docs"])
    doc_tokens = int(spec["doc_tokens"])
    dim = int(spec["dim"])
    bytes_read = docs * doc_tokens * (dim // 8)
    return {
        "stage": stage,
        "implementation": implementation,
        "dtype": spec["dtype"],
        "dim": dim,
        "query_tokens": int(spec["query_tokens"]),
        "doc_tokens": doc_tokens,
        "docs": docs,
        "latency_ms": float(latency_ms),
        "docs_per_second": float(docs / max(latency_ms / 1_000.0, 1e-12)),
        "bytes_read": int(bytes_read),
        "correctness_delta": delta,
        "correctness_tolerance": tolerance,
        "score_checksum": float(np.sum(scores, dtype=np.float64)),
    }


def _gate_passed(rows: list[dict[str, Any]]) -> bool:
    return all(row["correctness_delta"] <= row["correctness_tolerance"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged synthetic bitmax benchmarks.")
    parser.add_argument("--stage", required=True, choices=["stage0", "cpu-smoke", "cuda-smoke", "vast-large"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gate", type=Path, default=None)
    args = parser.parse_args()

    result = run_stage(args.stage, output_path=args.output, gate_path=args.gate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
