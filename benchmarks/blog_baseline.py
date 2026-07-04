from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import maxsim
from maxsim.experimental import pack_ternary, ternary_maxsim


def run_benchmark(stage: str = "smoke", *, output_path: Path | str | None = None, repeat: int | None = None) -> dict[str, Any]:
    specs = _stage_specs(stage)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        query, docs, offsets = _make_fixture(spec)
        actual_repeat = int(repeat if repeat is not None else spec["repeat"])

        reference_scores, reference_ms = _time_call(lambda: _dense_maxsim(query, docs, offsets), repeat=actual_repeat)
        fp32_row = _row(
            spec,
            implementation="fp32_query_fp32_docs",
            formula="sum(max(q_fp32 @ d_fp32.T))",
            latency_ms=reference_ms,
            scores=reference_scores,
            reference_scores=reference_scores,
            doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=32),
        )
        rows.append(fp32_row)

        query_int8 = _quantize_int8(query)
        docs_int8 = _quantize_int8(docs)
        int8_scores, int8_ms = _time_call(lambda: _dense_maxsim(query_int8.astype(np.float32), docs_int8.astype(np.float32), offsets), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="int8_query_int8_docs",
                formula="sum(max(q_int8 @ d_int8.T))",
                latency_ms=int8_ms,
                scores=int8_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=8),
                baseline_latency_ms=reference_ms,
            )
        )

        packed_binary = maxsim.pack_signs(docs, offsets)
        binary_scores, binary_ms = _time_call(lambda: maxsim.maxsim(query_int8, packed_binary), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="int8_query_binary_docs",
                formula="sum(max(q_int8 @ sign(doc).T))",
                latency_ms=binary_ms,
                scores=binary_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=1),
                baseline_latency_ms=reference_ms,
            )
        )

        query_binary = np.where(query >= 0, 1.0, -1.0).astype(np.float32)
        binary_binary_scores, binary_binary_ms = _time_call(lambda: maxsim.maxsim(query_binary, packed_binary), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="binary_query_binary_docs",
                formula="sum(max(sign(query) @ sign(doc).T))",
                latency_ms=binary_binary_ms,
                scores=binary_binary_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=1),
                baseline_latency_ms=reference_ms,
            )
        )

        packed_ternary = pack_ternary(docs, offsets, threshold=_ternary_threshold(docs))
        ternary_scores, ternary_ms = _time_call(lambda: ternary_maxsim(query_int8, packed_ternary), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="int8_query_ternary_docs",
                formula="sum(max(q_int8 @ ternary(doc).T))",
                latency_ms=ternary_ms,
                scores=ternary_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=2),
                baseline_latency_ms=reference_ms,
            )
        )

        token_scales = _token_scales(docs)
        token_scale_scores, token_scale_ms = _time_call(lambda: _scaled_sign_maxsim(query_int8, docs, offsets, token_scales), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="int8_query_binary_docs_token_scale",
                formula="sum(max(q_int8 @ (sign(doc) * token_scale).T))",
                latency_ms=token_scale_ms,
                scores=token_scale_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=1) + int(spec["doc_tokens"]) * 4,
                baseline_latency_ms=reference_ms,
            )
        )

        group_size = 16
        group_scales = _group_scales(docs, group_size)
        group_scale_scores, group_scale_ms = _time_call(lambda: _group_scaled_sign_maxsim(query_int8, docs, offsets, group_scales, group_size), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="int8_query_binary_docs_group_scale_16",
                formula="sum(max(q_int8 @ (sign(doc) * group_scale_16).T))",
                latency_ms=group_scale_ms,
                scores=group_scale_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=1) + int(spec["doc_tokens"]) * (int(spec["dim"]) // group_size) * 4,
                baseline_latency_ms=reference_ms,
            )
        )

        thresholds = _calibrated_thresholds(docs)
        threshold_scores, threshold_ms = _time_call(lambda: _threshold_sign_maxsim(query_int8, docs, offsets, thresholds), repeat=actual_repeat)
        rows.append(
            _row(
                spec,
                implementation="int8_query_binary_docs_calibrated_threshold",
                formula="sum(max(q_int8 @ threshold_sign(doc).T))",
                latency_ms=threshold_ms,
                scores=threshold_scores,
                reference_scores=reference_scores,
                doc_storage_bytes_per_doc=_doc_storage_bytes(spec, bits_per_dim=1),
                baseline_latency_ms=reference_ms,
            )
        )

    result = {
        "schema_version": 1,
        "benchmark": "blog_baseline",
        "stage": stage,
        "specs": [_public_spec(spec) for spec in specs],
        "results": rows,
    }
    output = Path(output_path) if output_path is not None else Path("benchmark-results") / f"blog-baseline-{stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _stage_specs(stage: str) -> list[dict[str, Any]]:
    if stage == "smoke":
        return [
            {
                "name": "smoke_2q_4d_4t_16dim",
                "query_tokens": 2,
                "docs": 4,
                "doc_tokens": 4,
                "dim": 16,
                "repeat": 3,
            }
        ]
    if stage == "blog-shape":
        return [
            {
                "name": "blog_33q_1000d_786t_128dim",
                "query_tokens": 33,
                "docs": 1000,
                "doc_tokens": 786,
                "dim": 128,
                "repeat": 3,
            }
        ]
    raise ValueError(f"unknown blog baseline stage: {stage}")


def _make_fixture(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260630 + int(spec["query_tokens"]) + int(spec["docs"]) + int(spec["doc_tokens"]) + int(spec["dim"]))
    query = rng.normal(loc=0.0, scale=2.0, size=(spec["query_tokens"], spec["dim"])).astype(np.float32)
    docs = rng.normal(loc=0.0, scale=2.0, size=(spec["docs"] * spec["doc_tokens"], spec["dim"])).astype(np.float32)
    offsets = np.arange(0, docs.shape[0] + 1, spec["doc_tokens"], dtype=np.int64)
    return query, docs, offsets


def _dense_maxsim(query: np.ndarray, docs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    query_float = np.asarray(query, dtype=np.float32)
    docs_float = np.asarray(docs, dtype=np.float32)
    doc_tokens = _uniform_doc_tokens(offsets)
    if doc_tokens is not None:
        dots = query_float @ docs_float.T
        return dots.reshape(query_float.shape[0], offsets.shape[0] - 1, doc_tokens).max(axis=2).sum(axis=0, dtype=np.float32)

    scores = np.empty(offsets.shape[0] - 1, dtype=np.float32)
    for doc_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        doc = docs_float[int(start) : int(end)]
        scores[doc_idx] = np.max(query_float @ doc.T, axis=1).sum(dtype=np.float32) if doc.shape[0] else 0.0
    return scores


def _quantize_int8(values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values)
    return np.clip(rounded, -127, 127).astype(np.int8)


def _ternary_threshold(docs: np.ndarray) -> float:
    return float(np.percentile(np.abs(docs), 25.0))


def _token_scales(docs: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)


def _group_scales(docs: np.ndarray, group_size: int) -> np.ndarray:
    if docs.shape[1] % group_size != 0:
        raise ValueError("dim must be divisible by group_size")
    groups = docs.reshape(docs.shape[0], docs.shape[1] // group_size, group_size)
    return np.mean(np.abs(groups), axis=2, dtype=np.float64).astype(np.float32)


def _calibrated_thresholds(docs: np.ndarray) -> np.ndarray:
    return np.median(docs, axis=0).astype(np.float32)


def _scaled_sign_maxsim(query: np.ndarray, docs: np.ndarray, offsets: np.ndarray, token_scales: np.ndarray) -> np.ndarray:
    signs = np.where(docs >= 0, 1.0, -1.0).astype(np.float32)
    scaled_docs = signs * token_scales[:, np.newaxis].astype(np.float32, copy=False)
    return _dense_maxsim(query, scaled_docs, offsets)


def _group_scaled_sign_maxsim(query: np.ndarray, docs: np.ndarray, offsets: np.ndarray, group_scales: np.ndarray, group_size: int) -> np.ndarray:
    signs = np.where(docs >= 0, 1.0, -1.0).astype(np.float32)
    expanded_scales = np.repeat(group_scales, group_size, axis=1).astype(np.float32, copy=False)
    return _dense_maxsim(query, signs * expanded_scales, offsets)


def _threshold_sign_maxsim(query: np.ndarray, docs: np.ndarray, offsets: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    signs = np.where(docs >= thresholds[np.newaxis, :], 1.0, -1.0).astype(np.float32)
    return _dense_maxsim(query, signs, offsets)


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


def _row(
    spec: dict[str, Any],
    *,
    implementation: str,
    formula: str,
    latency_ms: float,
    scores: np.ndarray,
    reference_scores: np.ndarray,
    doc_storage_bytes_per_doc: int,
    baseline_latency_ms: float | None = None,
) -> dict[str, Any]:
    score_values = np.asarray(scores, dtype=np.float32)
    reference_values = np.asarray(reference_scores, dtype=np.float32)
    return {
        **_public_spec(spec),
        "implementation": implementation,
        "formula": formula,
        "latency_ms": float(latency_ms),
        "doc_storage_bytes_per_doc": int(doc_storage_bytes_per_doc),
        "speedup_vs_fp32": float((baseline_latency_ms or latency_ms) / latency_ms),
        "max_abs_delta_vs_reference": float(np.max(np.abs(score_values - reference_values))),
        "score_checksum": float(np.sum(score_values, dtype=np.float64)),
    }


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(spec["name"]),
        "query_tokens": int(spec["query_tokens"]),
        "docs": int(spec["docs"]),
        "doc_tokens": int(spec["doc_tokens"]),
        "dim": int(spec["dim"]),
    }


def _doc_storage_bytes(spec: dict[str, Any], *, bits_per_dim: int) -> int:
    return int(spec["doc_tokens"] * spec["dim"] * bits_per_dim // 8)


def _uniform_doc_tokens(offsets: np.ndarray) -> int | None:
    lengths = np.diff(offsets)
    if lengths.size == 0 or np.any(lengths != lengths[0]):
        return None
    return int(lengths[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blog-style late-interaction quantization baselines.")
    parser.add_argument("--stage", default="smoke", choices=["smoke", "blog-shape"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    args = parser.parse_args()

    print(json.dumps(run_benchmark(stage=args.stage, output_path=args.output, repeat=args.repeat), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
