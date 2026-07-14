"""Validate the production per-token int4 and residual CUDA paths.

The correctness phase is intentionally small and always dim-128. It exercises
every reducer, duplicate candidate IDs, int8-query MaxSim, residual scoring,
the cascade, and workspace growth. The timing phase uses a separate seeded
synthetic corpus so CPU references do not dominate the production benchmark.

Default RTX 4090 run::

    python -m benchmarks.run_cuda_step1 \
        --output benchmark-results/cuda-step1-rtx4090.json \
        --fail-on-gate

Compute Sanitizer target::

    python -m benchmarks.run_cuda_step1 \
        --correctness-only --output /tmp/cuda-step1-correctness.json
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import maxsim
from maxsim.cascade import (
    cascade_topk,
    pack_residual_int4,
    prefix_topk,
    residual_int4_to_device,
    residual_score,
)
from maxsim.experimental import (
    int4_maxsim,
    int4_maxsim_int8q,
    int4_to_device,
    pack_int4_symmetric,
    topk_int4_maxsim,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional CUDA benchmark dependency
    torch = None


DIM = 128
REDUCERS = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")
DEFAULT_DOC_COUNTS = (4096,)
DEFAULT_CANDIDATE_COUNTS = (32, 128, 512, 2048, 4096)
SMOKE_DEFAULTS = {
    "doc_counts": (64,),
    "candidate_counts": (4, 16, 64),
    "min_doc_tokens": 4,
    "max_doc_tokens": 16,
    "batch": 1,
    "query_tokens": 4,
    "k": 4,
    "warmup": 1,
    "repeat": 2,
    "runs": 1,
}
PERFORMANCE_THRESHOLDS = {
    "max_per_token_full_overhead": 1.25,
    "max_per_token_topk_overhead": 1.25,
    "min_per_token_int8_topk_speedup": 1.25,
    "max_residual_candidate_over_per_token": 2.25,
    "min_residual_candidate_speedup_at_512": 2.0,
    "min_cascade_speedup_at_512": 1.0,
    "max_p95_over_p50": 1.25,
    "candidate_monotonic_slack": 1.10,
}


def run_benchmark(
    *,
    output_path: Path,
    doc_counts: Sequence[int] = DEFAULT_DOC_COUNTS,
    candidate_counts: Sequence[int] = DEFAULT_CANDIDATE_COUNTS,
    min_doc_tokens: int = 64,
    max_doc_tokens: int = 192,
    batch: int = 4,
    query_tokens: int = 32,
    k: int = 10,
    warmup: int = 5,
    repeat: int = 20,
    runs: int = 3,
    seed: int = 20260713,
    correctness_only: bool = False,
    performance_gates_enabled: bool = True,
) -> dict[str, Any]:
    """Run correctness gates and, unless disabled, production-shaped timings."""
    config = _validate_config(
        doc_counts=doc_counts,
        candidate_counts=candidate_counts,
        min_doc_tokens=min_doc_tokens,
        max_doc_tokens=max_doc_tokens,
        batch=batch,
        query_tokens=query_tokens,
        k=k,
        warmup=warmup,
        repeat=repeat,
        runs=runs,
        seed=seed,
    )
    torch_module, cuda_extension = _require_cuda()
    residual_mode = int(cuda_extension.get_residual_reducer_warps())
    residual_force_mode = int(
        cuda_extension.get_residual_reducer_force_warps()
    )
    if residual_mode != 8 or residual_force_mode != -1:
        raise RuntimeError(
            "the canonical benchmark requires adaptive residual routing "
            "(base mode 8, force mode -1); found "
            f"base mode {residual_mode}, force mode {residual_force_mode}"
        )
    correctness = _run_correctness_phase(seed, torch_module)

    cases: list[dict[str, Any]] = []
    if not correctness_only:
        for doc_count in config["doc_counts"]:
            cases.append(
                _run_timing_case(
                    doc_count=doc_count,
                    candidate_counts=_candidate_counts_for_docs(
                        config["candidate_counts"], doc_count, config["k"]
                    ),
                    min_doc_tokens=config["min_doc_tokens"],
                    max_doc_tokens=config["max_doc_tokens"],
                    batch=config["batch"],
                    query_tokens=config["query_tokens"],
                    k=config["k"],
                    warmup=config["warmup"],
                    repeat=config["repeat"],
                    runs=config["runs"],
                    seed=config["seed"],
                    torch_module=torch_module,
                )
            )

    timing_rows = [row for case in cases for row in case["timings"]]
    performance_gates = (
        _evaluate_performance_gates(timing_rows)
        if timing_rows and performance_gates_enabled
        else []
    )
    performance_passed = all(
        gate["passed"] for gate in performance_gates if gate["applied"]
    )
    gate_passed = bool(correctness["gate_passed"] and performance_passed)
    result = {
        "schema_version": 2,
        "benchmark": "cuda_step1_production_paths",
        "metadata": _runtime_metadata(torch_module, cuda_extension),
        "config": {
            **config,
            "dim": DIM,
            "correctness_only": bool(correctness_only),
            "performance_gates_enabled": bool(performance_gates_enabled),
        },
        "correctness": correctness,
        "cases": cases,
        "gates": {
            "thresholds": PERFORMANCE_THRESHOLDS,
            "performance": performance_gates,
            "correctness_passed": bool(correctness["gate_passed"]),
            "performance_passed": bool(performance_passed),
        },
        "gate_passed": gate_passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _run_correctness_phase(seed: int, torch_module: Any) -> dict[str, Any]:
    rng = np.random.default_rng(seed ^ 0x51A7)
    lengths = np.array([0, 1, 2, 3, 7, 17], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    docs = rng.standard_normal((int(offsets[-1]), DIM), dtype=np.float32)
    candidates = np.array([5, 1, 5, 0, 2], dtype=np.int64)

    tensor_cpu = pack_int4_symmetric(docs, offsets, scale_granularity="tensor")
    token_cpu = pack_int4_symmetric(docs, offsets, scale_granularity="token")
    residual_cpu = pack_residual_int4(docs, offsets)
    tensor_cuda = int4_to_device(tensor_cpu)
    token_cuda = int4_to_device(token_cpu)
    residual_cuda = residual_int4_to_device(residual_cpu)

    rows: list[dict[str, Any]] = []
    query_shapes = (("initial", 2, 5), ("resized", 3, 17))
    for shape_name, batch, query_tokens in query_shapes:
        query = rng.standard_normal((batch, query_tokens, DIM), dtype=np.float32)
        weights = rng.uniform(-0.5, 2.0, size=(batch, query_tokens)).astype(
            np.float32
        )

        for format_name, cpu_packed, cuda_packed in (
            ("int4_tensor", tensor_cpu, tensor_cuda),
            ("int4_per_token", token_cpu, token_cuda),
        ):
            for reducer in REDUCERS:
                kwargs = _reducer_kwargs(reducer, weights)
                expected_full = int4_maxsim(
                    query, cpu_packed, device="cpu", reducer=reducer, **kwargs
                )
                actual_full = int4_maxsim(
                    query, cuda_packed, device="cuda", reducer=reducer, **kwargs
                )
                rows.append(
                    _parity_row(
                        format_name,
                        "full",
                        reducer,
                        shape_name,
                        actual_full,
                        expected_full,
                    )
                )

                expected_candidates = int4_maxsim(
                    query,
                    cpu_packed,
                    device="cpu",
                    reducer=reducer,
                    candidate_indices=candidates,
                    **kwargs,
                )
                actual_candidates = int4_maxsim(
                    query,
                    cuda_packed,
                    device="cuda",
                    reducer=reducer,
                    candidate_indices=candidates,
                    **kwargs,
                )
                rows.append(
                    _parity_row(
                        format_name,
                        "candidates",
                        reducer,
                        shape_name,
                        actual_candidates,
                        expected_candidates,
                        projection=np.take(actual_full, candidates, axis=-1),
                    )
                )

        for reducer in REDUCERS:
            kwargs = _reducer_kwargs(reducer, weights)
            expected_full = residual_score(
                query, residual_cpu, device="cpu", reducer=reducer, **kwargs
            )
            actual_full = residual_score(
                query, residual_cuda, device="cuda", reducer=reducer, **kwargs
            )
            rows.append(
                _parity_row(
                    "int4_residual",
                    "full",
                    reducer,
                    shape_name,
                    actual_full,
                    expected_full,
                )
            )

            expected_candidates = residual_score(
                query,
                residual_cpu,
                device="cpu",
                reducer=reducer,
                candidate_indices=candidates,
                **kwargs,
            )
            actual_candidates = residual_score(
                query,
                residual_cuda,
                device="cuda",
                reducer=reducer,
                candidate_indices=candidates,
                **kwargs,
            )
            rows.append(
                _parity_row(
                    "int4_residual",
                    "candidates",
                    reducer,
                    shape_name,
                    actual_candidates,
                    expected_candidates,
                    projection=np.take(actual_full, candidates, axis=-1),
                )
            )

        for format_name, cpu_packed, cuda_packed in (
            ("int4_tensor", tensor_cpu, tensor_cuda),
            ("int4_per_token", token_cpu, token_cuda),
        ):
            expected_int8 = _int8q_reference(cpu_packed, query)
            actual_int8 = int4_maxsim_int8q(query, cuda_packed, device="cuda")
            rows.append(
                _parity_row(
                    format_name,
                    "full",
                    "maxsim_int8q",
                    shape_name,
                    actual_int8,
                    expected_int8,
                    tolerance=1e-3,
                )
            )
            expected_indices = _stable_topk(expected_int8, 4)
            expected_scores = np.take_along_axis(
                expected_int8, expected_indices, axis=1
            )
            actual_scores, actual_indices = topk_int4_maxsim(
                query,
                cuda_packed,
                4,
                device="cuda",
                prefer_int8_query=True,
            )
            rows.append(
                _parity_row(
                    format_name,
                    "topk",
                    "maxsim_int8q",
                    shape_name,
                    actual_scores,
                    expected_scores,
                    tolerance=1e-3,
                    indices_exact=bool(np.array_equal(actual_indices, expected_indices)),
                )
            )

        expected_prefix_scores, expected_prefix_indices = prefix_topk(
            query, residual_cpu, 4, device="cpu"
        )
        actual_prefix_scores, actual_prefix_indices = prefix_topk(
            query, residual_cuda, 4, device="cuda"
        )
        rows.append(
            _parity_row(
                "int4_residual",
                "prefix_topk",
                "maxsim",
                shape_name,
                actual_prefix_scores,
                expected_prefix_scores,
                indices_exact=bool(
                    np.array_equal(actual_prefix_indices, expected_prefix_indices)
                ),
            )
        )

        expected_cascade_scores, expected_cascade_indices = cascade_topk(
            query, residual_cpu, 2, candidates=4, device="cpu"
        )
        actual_cascade_scores, actual_cascade_indices = cascade_topk(
            query, residual_cuda, 2, candidates=4, device="cuda"
        )
        rows.append(
            _parity_row(
                "int4_residual",
                "cascade",
                "maxsim",
                shape_name,
                actual_cascade_scores,
                expected_cascade_scores,
                indices_exact=bool(
                    np.array_equal(actual_cascade_indices, expected_cascade_indices)
                ),
            )
        )

    torch_module.cuda.synchronize()
    workspace = {
        "int4_tensor": int(tensor_cuda.data.workspace_bytes),
        "int4_per_token": int(token_cuda.data.workspace_bytes),
        "int4_residual": int(residual_cuda.prefix_data.workspace_bytes),
    }
    return {
        "dim": DIM,
        "doc_lengths": lengths.tolist(),
        "duplicate_candidates": candidates.tolist(),
        "query_shapes": [
            {"name": name, "batch": batch, "query_tokens": query_tokens}
            for name, batch, query_tokens in query_shapes
        ],
        "workspace_bytes_after_resize": workspace,
        "rows": rows,
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "gate_passed": all(row["passed"] for row in rows),
    }


def _run_timing_case(
    *,
    doc_count: int,
    candidate_counts: Sequence[int],
    min_doc_tokens: int,
    max_doc_tokens: int,
    batch: int,
    query_tokens: int,
    k: int,
    warmup: int,
    repeat: int,
    runs: int,
    seed: int,
    torch_module: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + doc_count * 1009)
    lengths = rng.integers(
        min_doc_tokens, max_doc_tokens + 1, size=doc_count, dtype=np.int64
    )
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    docs = rng.standard_normal((int(offsets[-1]), DIM), dtype=np.float32)
    query = rng.standard_normal((batch, query_tokens, DIM), dtype=np.float32)
    candidate_order = np.ascontiguousarray(rng.permutation(doc_count), dtype=np.int64)
    dense_fp32_bytes = int(docs.nbytes)

    tensor_cpu = pack_int4_symmetric(docs, offsets, scale_granularity="tensor")
    token_cpu = pack_int4_symmetric(docs, offsets, scale_granularity="token")
    residual_cpu = pack_residual_int4(docs, offsets)
    tensor_cuda = int4_to_device(tensor_cpu)
    token_cuda = int4_to_device(token_cpu)
    residual_cuda = residual_int4_to_device(residual_cpu)
    del docs, tensor_cpu, token_cpu, residual_cpu
    gc.collect()

    tensor_full = int4_maxsim(query, tensor_cuda, device="cuda")
    tensor_int8_full = int4_maxsim_int8q(query, tensor_cuda, device="cuda")
    token_full = int4_maxsim(query, token_cuda, device="cuda")
    token_int8_full = int4_maxsim_int8q(query, token_cuda, device="cuda")
    residual_full = residual_score(query, residual_cuda, device="cuda")

    timings: list[dict[str, Any]] = []

    def measure(
        operation: str,
        format_name: str,
        scope: str,
        call: Callable[[], Any],
        *,
        query_mode: str = "fp32",
        candidate_count: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> Any:
        summary, output = _measure_case(
            call,
            torch_module=torch_module,
            warmup=warmup,
            repeat=repeat,
            runs=runs,
        )
        row = {
            "doc_count": doc_count,
            "candidate_count": candidate_count,
            "operation": operation,
            "format": format_name,
            "scope": scope,
            "reducer": "maxsim",
            "query_mode": query_mode,
            **summary,
            **_result_diagnostics(output),
        }
        if diagnostics:
            row.update(diagnostics)
        timings.append(row)
        return output

    tensor_full_output = measure(
        "int4_tensor_full_fp32",
        "int4_tensor",
        "full",
        lambda: int4_maxsim(query, tensor_cuda, device="cuda"),
    )
    timings[-1]["max_abs_error_vs_reference_scores"] = _max_abs_error(
        tensor_full_output, tensor_full
    )
    token_full_output = measure(
        "int4_per_token_full_fp32",
        "int4_per_token",
        "full",
        lambda: int4_maxsim(query, token_cuda, device="cuda"),
    )
    timings[-1]["max_abs_error_vs_reference_scores"] = _max_abs_error(
        token_full_output, token_full
    )
    tensor_int8_full_output = measure(
        "int4_tensor_full_int8",
        "int4_tensor",
        "full",
        lambda: int4_maxsim_int8q(query, tensor_cuda, device="cuda"),
        query_mode="int8",
    )
    timings[-1]["max_abs_error_vs_reference_scores"] = _max_abs_error(
        tensor_int8_full_output, tensor_int8_full
    )
    token_int8_full_output = measure(
        "int4_per_token_full_int8",
        "int4_per_token",
        "full",
        lambda: int4_maxsim_int8q(query, token_cuda, device="cuda"),
        query_mode="int8",
    )
    timings[-1]["max_abs_error_vs_reference_scores"] = _max_abs_error(
        token_int8_full_output, token_int8_full
    )
    residual_full_output = measure(
        "int4_residual_full_fp32",
        "int4_residual",
        "full",
        lambda: residual_score(query, residual_cuda, device="cuda"),
    )
    timings[-1]["max_abs_error_vs_reference_scores"] = _max_abs_error(
        residual_full_output, residual_full
    )

    tensor_topk_indices = _stable_topk(tensor_full, k)
    tensor_int8_topk_indices = _stable_topk(tensor_int8_full, k)
    token_topk_indices = _stable_topk(token_full, k)
    token_int8_topk_indices = _stable_topk(token_int8_full, k)
    for operation, format_name, packed, prefer_int8, reference, reference_indices in (
        (
            "int4_tensor_topk_fp32",
            "int4_tensor",
            tensor_cuda,
            False,
            tensor_full,
            tensor_topk_indices,
        ),
        (
            "int4_tensor_topk_int8",
            "int4_tensor",
            tensor_cuda,
            True,
            tensor_int8_full,
            tensor_int8_topk_indices,
        ),
        (
            "int4_per_token_topk_fp32",
            "int4_per_token",
            token_cuda,
            False,
            token_full,
            token_topk_indices,
        ),
        (
            "int4_per_token_topk_int8",
            "int4_per_token",
            token_cuda,
            True,
            token_int8_full,
            token_int8_topk_indices,
        ),
    ):
        output = measure(
            operation,
            format_name,
            "topk",
            lambda packed=packed, prefer_int8=prefer_int8: topk_int4_maxsim(
                query,
                packed,
                k,
                device="cuda",
                prefer_int8_query=prefer_int8,
            ),
            query_mode="int8" if prefer_int8 else "fp32",
        )
        scores, indices = output
        expected_scores = np.take_along_axis(reference, reference_indices, axis=1)
        timings[-1].update(
            {
                "max_abs_error_vs_full_scores": _max_abs_error(
                    scores, expected_scores
                ),
                "indices_exact_vs_stable_full": bool(
                    np.array_equal(indices, reference_indices)
                ),
            }
        )

    residual_full_topk = _stable_topk(residual_full, k)
    for candidate_count in candidate_counts:
        candidate_ids = np.ascontiguousarray(
            candidate_order[:candidate_count], dtype=np.int64
        )
        token_output = measure(
            "int4_per_token_candidates_fp32",
            "int4_per_token",
            "candidates",
            lambda candidate_ids=candidate_ids: int4_maxsim(
                query,
                token_cuda,
                device="cuda",
                candidate_indices=candidate_ids,
            ),
            candidate_count=candidate_count,
        )
        timings[-1]["max_abs_error_vs_full_projection"] = _max_abs_error(
            token_output, np.take(token_full, candidate_ids, axis=-1)
        )

        residual_output = measure(
            "int4_residual_candidates_fp32",
            "int4_residual",
            "candidates",
            lambda candidate_ids=candidate_ids: residual_score(
                query,
                residual_cuda,
                device="cuda",
                candidate_indices=candidate_ids,
            ),
            candidate_count=candidate_count,
        )
        timings[-1]["max_abs_error_vs_full_projection"] = _max_abs_error(
            residual_output, np.take(residual_full, candidate_ids, axis=-1)
        )

        cascade_output = measure(
            "int4_residual_cascade",
            "int4_residual",
            "cascade",
            lambda candidate_count=candidate_count: cascade_topk(
                query,
                residual_cuda,
                k,
                candidates=candidate_count,
                device="cuda",
            ),
            candidate_count=candidate_count,
        )
        cascade_scores, cascade_indices = cascade_output
        timings[-1]["max_abs_error_vs_full_scores"] = _max_abs_error(
            cascade_scores,
            np.take_along_axis(residual_full, cascade_indices, axis=1),
        )
        if candidate_count == doc_count:
            timings[-1]["indices_exact_vs_stable_full"] = bool(
                np.array_equal(cascade_indices, residual_full_topk)
            )
        timings[-1]["topk_overlap_vs_full_residual"] = _topk_overlap(
            cascade_indices, residual_full_topk
        )

    storage = {
        "dense_fp32_bytes": dense_fp32_bytes,
        "int4_tensor": _storage_report(tensor_cuda, dense_fp32_bytes),
        "int4_per_token": _storage_report(token_cuda, dense_fp32_bytes),
        "int4_residual": _storage_report(residual_cuda, dense_fp32_bytes),
    }
    result = {
        "shape": {
            "docs": doc_count,
            "total_doc_tokens": int(offsets[-1]),
            "min_doc_tokens": int(lengths.min()),
            "max_doc_tokens": int(lengths.max()),
            "mean_doc_tokens": float(lengths.mean()),
            "batch": batch,
            "query_tokens": query_tokens,
            "dim": DIM,
            "k": k,
            "candidate_counts": list(candidate_counts),
        },
        "storage": storage,
        "timings": timings,
    }

    del tensor_cuda, token_cuda, residual_cuda
    gc.collect()
    torch_module.cuda.empty_cache()
    return result


def _measure_case(
    call: Callable[[], Any],
    *,
    torch_module: Any,
    warmup: int,
    repeat: int,
    runs: int,
) -> tuple[dict[str, Any], Any]:
    for _ in range(warmup):
        call()
    torch_module.cuda.synchronize()

    samples: list[list[float]] = []
    output = None
    for _ in range(runs):
        run_samples = []
        for _ in range(repeat):
            torch_module.cuda.synchronize()
            started = time.perf_counter()
            output = call()
            torch_module.cuda.synchronize()
            run_samples.append((time.perf_counter() - started) * 1000.0)
        samples.append(run_samples)
    return _summarize_samples(samples), output


def _summarize_samples(samples: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not samples or any(not run for run in samples):
        raise ValueError("latency samples must contain at least one non-empty run")
    try:
        values = np.asarray(samples, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            "latency samples must be a rectangular finite non-negative matrix"
        ) from exc
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("latency samples must be a rectangular finite non-negative matrix")
    flattened = values.reshape(-1)
    median = float(np.median(flattened))
    return {
        "latency_best_ms": float(np.min(flattened)),
        "latency_median_ms": median,
        "latency_p50_ms": median,
        "latency_p95_ms": float(np.percentile(flattened, 95)),
        "latency_run_medians_ms": [float(value) for value in np.median(values, axis=1)],
        "latency_samples_ms": [[float(value) for value in run] for run in values],
    }


def _evaluate_performance_gates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (int(row["doc_count"]), row["operation"], row.get("candidate_count")): row
        for row in rows
    }
    doc_counts = sorted({int(row["doc_count"]) for row in rows})
    gates: list[dict[str, Any]] = []

    for doc_count in doc_counts:
        def value(operation: str, candidate_count: int | None = None) -> float | None:
            row = by_key.get((doc_count, operation, candidate_count))
            return None if row is None else float(row["latency_p50_ms"])

        tensor_full = value("int4_tensor_full_fp32")
        token_full = value("int4_per_token_full_fp32")
        gates.append(
            _ratio_gate(
                "per_token_full_over_tensor",
                doc_count,
                token_full,
                tensor_full,
                "<=",
                PERFORMANCE_THRESHOLDS["max_per_token_full_overhead"],
            )
        )
        tensor_topk = value("int4_tensor_topk_fp32")
        token_topk = value("int4_per_token_topk_fp32")
        gates.append(
            _ratio_gate(
                "per_token_topk_over_tensor",
                doc_count,
                token_topk,
                tensor_topk,
                "<=",
                PERFORMANCE_THRESHOLDS["max_per_token_topk_overhead"],
            )
        )
        token_int8_topk = value("int4_per_token_topk_int8")
        gates.append(
            _ratio_gate(
                "per_token_int8_topk_speedup",
                doc_count,
                token_topk,
                token_int8_topk,
                ">=",
                PERFORMANCE_THRESHOLDS["min_per_token_int8_topk_speedup"],
            )
        )

        candidate_counts = sorted(
            {
                int(row["candidate_count"])
                for row in rows
                if int(row["doc_count"]) == doc_count
                and row.get("candidate_count") is not None
                and row["operation"] == "int4_residual_candidates_fp32"
            }
        )
        residual_candidate_values = []
        cascade_values = []
        for candidate_count in candidate_counts:
            token_candidate = value(
                "int4_per_token_candidates_fp32", candidate_count
            )
            residual_candidate = value(
                "int4_residual_candidates_fp32", candidate_count
            )
            gates.append(
                _ratio_gate(
                    "residual_candidate_over_per_token",
                    doc_count,
                    residual_candidate,
                    token_candidate,
                    "<=",
                    PERFORMANCE_THRESHOLDS[
                        "max_residual_candidate_over_per_token"
                    ],
                    candidate_count=candidate_count,
                )
            )
            residual_candidate_values.append((candidate_count, residual_candidate))
            cascade_values.append(
                (candidate_count, value("int4_residual_cascade", candidate_count))
            )

        gates.extend(
            _monotonic_gates(
                "residual_candidate_monotonic",
                doc_count,
                residual_candidate_values,
            )
        )
        gates.extend(
            _monotonic_gates(
                "cascade_monotonic",
                doc_count,
                [
                    item
                    for item in cascade_values
                    if item[0] < doc_count
                ],
            )
        )

        residual_full = value("int4_residual_full_fp32")
        residual_512 = (
            value("int4_residual_candidates_fp32", 512)
            if doc_count >= 4096
            else None
        )
        cascade_512 = (
            value("int4_residual_cascade", 512) if doc_count >= 4096 else None
        )
        gates.append(
            _ratio_gate(
                "residual_candidate_speedup_at_512",
                doc_count,
                residual_full,
                residual_512,
                ">=",
                PERFORMANCE_THRESHOLDS[
                    "min_residual_candidate_speedup_at_512"
                ],
                candidate_count=512,
            )
        )
        gates.append(
            _ratio_gate(
                "cascade_speedup_at_512",
                doc_count,
                residual_full,
                cascade_512,
                ">=",
                PERFORMANCE_THRESHOLDS["min_cascade_speedup_at_512"],
                candidate_count=512,
            )
        )

    for row in rows:
        output_tolerance = 1e-3 if row.get("query_mode") == "int8" else 2e-3
        scope = row.get("scope")
        required_error_fields = {
            "full": ("max_abs_error_vs_reference_scores",),
            "topk": ("max_abs_error_vs_full_scores",),
            "candidates": ("max_abs_error_vs_full_projection",),
            "cascade": ("max_abs_error_vs_full_scores",),
        }.get(scope, ())
        missing_fields = [
            name for name in required_error_fields if row.get(name) is None
        ]
        output_errors = [
            float(row[name])
            for name in (
                "max_abs_error_vs_reference_scores",
                "max_abs_error_vs_full_scores",
                "max_abs_error_vs_full_projection",
            )
            if row.get(name) is not None
        ]
        output_error = max(output_errors) if output_errors else None
        indices_exact = row.get("indices_exact_vs_stable_full")
        requires_exact_indices = scope == "topk" or (
            scope == "cascade"
            and row.get("candidate_count") == row.get("doc_count")
        )
        if requires_exact_indices and indices_exact is None:
            missing_fields.append("indices_exact_vs_stable_full")
        output_passed = (
            bool(row.get("finite", False))
            and not missing_fields
            and output_error is not None
            and output_error <= output_tolerance
            and (not requires_exact_indices or indices_exact is True)
        )
        gates.append(
            {
                "name": "output_validity",
                "doc_count": int(row["doc_count"]),
                "candidate_count": row.get("candidate_count"),
                "operation": row["operation"],
                "metric": "max_abs_parity_error",
                "value": output_error,
                "comparison": "<=",
                "threshold": output_tolerance,
                "indices_exact": indices_exact,
                "missing_fields": missing_fields,
                "applied": True,
                "passed": bool(output_passed),
            }
        )

        p50 = float(row["latency_p50_ms"])
        p95 = float(row["latency_p95_ms"])
        ratio = p95 / p50 if p50 > 0.0 else float("inf")
        threshold = PERFORMANCE_THRESHOLDS["max_p95_over_p50"]
        gates.append(
            {
                "name": "latency_stability",
                "doc_count": int(row["doc_count"]),
                "candidate_count": row.get("candidate_count"),
                "operation": row["operation"],
                "metric": "p95_ms / p50_ms",
                "value": ratio,
                "comparison": "<=",
                "threshold": threshold,
                "applied": True,
                "passed": bool(ratio <= threshold),
            }
        )
    return gates


def _ratio_gate(
    name: str,
    doc_count: int,
    numerator: float | None,
    denominator: float | None,
    comparison: str,
    threshold: float,
    *,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    applied = numerator is not None and denominator is not None and denominator > 0.0
    value = None if not applied else float(numerator / denominator)
    passed = None
    if applied:
        passed = bool(value <= threshold if comparison == "<=" else value >= threshold)
    return {
        "name": name,
        "doc_count": int(doc_count),
        "candidate_count": candidate_count,
        "metric": "latency_p50_ratio",
        "value": value,
        "comparison": comparison,
        "threshold": float(threshold),
        "applied": bool(applied),
        "passed": passed,
    }


def _monotonic_gates(
    name: str,
    doc_count: int,
    values: Sequence[tuple[int, float | None]],
) -> list[dict[str, Any]]:
    gates = []
    slack = PERFORMANCE_THRESHOLDS["candidate_monotonic_slack"]
    for (left_count, left), (right_count, right) in zip(values, values[1:]):
        applied = left is not None and right is not None
        ratio = None if not applied or right == 0.0 else float(left / right)
        passed = None if not applied else bool(left <= right * slack)
        gates.append(
            {
                "name": name,
                "doc_count": int(doc_count),
                "candidate_count": [int(left_count), int(right_count)],
                "metric": "smaller_candidate_p50 / larger_candidate_p50",
                "value": ratio,
                "comparison": "<=",
                "threshold": float(slack),
                "applied": bool(applied),
                "passed": passed,
            }
        )
    return gates


def _parity_row(
    format_name: str,
    scope: str,
    reducer: str,
    query_shape: str,
    actual: Any,
    expected: Any,
    *,
    tolerance: float | None = None,
    projection: Any | None = None,
    indices_exact: bool | None = None,
) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype=np.float32)
    expected_values = np.asarray(expected, dtype=np.float32)
    error = np.abs(actual_values - expected_values)
    maximum = float(error.max(initial=0.0))
    tolerance_value = float(
        tolerance
        if tolerance is not None
        else (5e-3 if reducer == "smoothsim" else 2e-3)
    )
    projection_error = (
        None
        if projection is None
        else _max_abs_error(actual_values, np.asarray(projection, dtype=np.float32))
    )
    finite = bool(np.isfinite(actual_values).all())
    passed = finite and maximum <= tolerance_value
    if projection_error is not None:
        passed = passed and projection_error <= tolerance_value
    if indices_exact is not None:
        passed = passed and indices_exact
    return {
        "format": format_name,
        "scope": scope,
        "reducer": reducer,
        "query_shape": query_shape,
        "output_shape": list(actual_values.shape),
        "finite": finite,
        "max_abs_error": maximum,
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
        "max_abs_error_vs_full_projection": projection_error,
        "indices_exact": indices_exact,
        "tolerance": tolerance_value,
        "passed": bool(passed),
    }


def _int8q_reference(packed: Any, query: np.ndarray) -> np.ndarray:
    values = packed.values.astype(np.float32)
    result = np.empty((query.shape[0], packed.num_docs), dtype=np.float32)
    for batch_idx, query_matrix in enumerate(query):
        max_abs = np.max(np.abs(query_matrix), axis=1)
        query_scale = np.where(max_abs == 0.0, 1.0, max_abs / 127.0).astype(
            np.float32
        )
        query_int = np.clip(
            np.rint(query_matrix / query_scale[:, None]), -127, 127
        ).astype(np.float32)
        for doc_idx in range(packed.num_docs):
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            if start == end:
                result[batch_idx, doc_idx] = 0.0
                continue
            dots = query_int @ values[start:end].T
            if packed.token_scale is not None:
                dots *= packed.token_scale[start:end][None, :]
            best = dots.max(axis=1).astype(np.float64)
            result[batch_idx, doc_idx] = np.float32(
                float((best * query_scale).sum()) * packed.scale
            )
    return result


def _storage_report(packed: Any, dense_fp32_bytes: int) -> dict[str, Any]:
    handle = packed.prefix_data if hasattr(packed, "prefix_data") else packed.data
    encoded = int(packed.storage_bytes)
    offsets = int(packed.doc_offsets.nbytes)
    total = encoded + offsets
    return {
        "encoded_bytes": encoded,
        "offset_bytes": offsets,
        "encoded_plus_offsets_bytes": total,
        "compression_vs_fp32_encoded": float(dense_fp32_bytes / encoded),
        "compression_vs_fp32_with_offsets": float(dense_fp32_bytes / total),
        "device_resident_bytes": int(handle.resident_bytes),
        "device_workspace_bytes": int(handle.workspace_bytes),
    }


def _result_diagnostics(output: Any) -> dict[str, Any]:
    if isinstance(output, tuple):
        scores = np.asarray(output[0], dtype=np.float32)
        indices = np.asarray(output[1], dtype=np.int64)
        return {
            "output_shape": [list(scores.shape), list(indices.shape)],
            "finite": bool(np.isfinite(scores).all()),
            "score_checksum": float(scores.sum(dtype=np.float64)),
            "index_checksum": int(indices.sum(dtype=np.int64)),
        }
    values = np.asarray(output, dtype=np.float32)
    return {
        "output_shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "score_checksum": float(values.sum(dtype=np.float64)),
    }


def _stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim == 1:
        return np.lexsort((np.arange(values.shape[0], dtype=np.int64), -values))[:k]
    return np.stack(
        [
            np.lexsort((np.arange(row.shape[0], dtype=np.int64), -row))[:k]
            for row in values
        ]
    )


def _topk_overlap(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_values = np.asarray(actual, dtype=np.int64)
    expected_values = np.asarray(expected, dtype=np.int64)
    if actual_values.ndim == 1:
        actual_values = actual_values[None, :]
        expected_values = expected_values[None, :]
    return float(
        np.mean(
            [
                np.intersect1d(left, right, assume_unique=False).size / left.size
                for left, right in zip(actual_values, expected_values)
            ]
        )
    )


def _max_abs_error(actual: Any, expected: Any) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(actual, dtype=np.float32)
                - np.asarray(expected, dtype=np.float32)
            ),
            initial=0.0,
        )
    )


def _reducer_kwargs(reducer: str, weights: np.ndarray) -> dict[str, Any]:
    if reducer == "weighted_maxsim":
        return {"query_weights": weights}
    if reducer == "smoothsim":
        return {"temperature": 0.7}
    return {}


def _candidate_counts_for_docs(
    candidate_counts: Sequence[int], doc_count: int, k: int
) -> tuple[int, ...]:
    selected = tuple(sorted({int(value) for value in candidate_counts if k <= value <= doc_count}))
    if not selected:
        raise ValueError(
            f"no candidate count satisfies k={k} <= candidates <= docs={doc_count}"
        )
    return selected


def _validate_config(**values: Any) -> dict[str, Any]:
    doc_counts = tuple(sorted({int(value) for value in values["doc_counts"]}))
    candidate_counts = tuple(
        sorted({int(value) for value in values["candidate_counts"]})
    )
    if not doc_counts or any(value < 1 for value in doc_counts):
        raise ValueError("doc_counts must contain positive integers")
    if not candidate_counts or any(value < 1 for value in candidate_counts):
        raise ValueError("candidate_counts must contain positive integers")
    for name in ("min_doc_tokens", "max_doc_tokens", "batch", "query_tokens", "k", "repeat", "runs"):
        if int(values[name]) < 1:
            raise ValueError(f"{name} must be >= 1")
    if int(values["warmup"]) < 0:
        raise ValueError("warmup must be >= 0")
    if int(values["max_doc_tokens"]) < int(values["min_doc_tokens"]):
        raise ValueError("max_doc_tokens must be >= min_doc_tokens")
    if int(values["k"]) > min(doc_counts):
        raise ValueError("k cannot exceed the smallest doc count")
    for doc_count in doc_counts:
        _candidate_counts_for_docs(candidate_counts, doc_count, int(values["k"]))
    return {
        "doc_counts": list(doc_counts),
        "candidate_counts": list(candidate_counts),
        "min_doc_tokens": int(values["min_doc_tokens"]),
        "max_doc_tokens": int(values["max_doc_tokens"]),
        "batch": int(values["batch"]),
        "query_tokens": int(values["query_tokens"]),
        "k": int(values["k"]),
        "warmup": int(values["warmup"]),
        "repeat": int(values["repeat"]),
        "runs": int(values["runs"]),
        "seed": int(values["seed"]),
    }


def _require_cuda() -> tuple[Any, Any]:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("run_cuda_step1 requires PyTorch with an available CUDA GPU")
    try:
        extension = importlib.import_module("maxsim._maxsim_cuda")
    except ImportError as exc:
        raise RuntimeError(
            "run_cuda_step1 requires a CUDA-enabled maxsim build"
        ) from exc
    device_name = torch.cuda.get_device_name(0)
    if (
        torch.cuda.get_device_capability(0) != (8, 9)
        or device_name != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError(
            "run_cuda_step1 is the RTX 4090/SM89 validation harness; "
            f"found {device_name!r} with compute capability "
            f"{torch.cuda.get_device_capability(0)}"
        )
    return torch, extension


def _runtime_metadata(torch_module: Any, cuda_extension: Any) -> dict[str, Any]:
    properties = torch_module.cuda.get_device_properties(0)
    dirty_override = os.environ.get("MAXSIM_GIT_DIRTY")
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.environ.get("MAXSIM_GIT_COMMIT")
        or _command_output(("git", "rev-parse", "HEAD")),
        "git_dirty": (
            _git_dirty()
            if dirty_override is None
            else dirty_override.lower() not in {"", "0", "false", "no"}
        ),
        "source_archive_sha256": os.environ.get(
            "MAXSIM_SOURCE_ARCHIVE_SHA256"
        ),
        "source_archive_path": os.environ.get("MAXSIM_SOURCE_ARCHIVE_PATH"),
        "vast_instance_id": os.environ.get("MAXSIM_VAST_INSTANCE_ID"),
        "vast_offer_id": os.environ.get("MAXSIM_VAST_OFFER_ID"),
        "container_image": os.environ.get("MAXSIM_CONTAINER_IMAGE"),
        "build_command": os.environ.get("MAXSIM_BUILD_COMMAND"),
        "benchmark_command": os.environ.get("MAXSIM_BENCHMARK_COMMAND"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_module.__version__,
        "maxsim": _package_version("maxsim"),
        "cuda_runtime": torch_module.version.cuda,
        "cuda_arch_env": os.environ.get("MAXSIM_CUDA_ARCH"),
        "gpu": properties.name,
        "compute_capability": list(torch_module.cuda.get_device_capability(0)),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "driver_version": _command_output(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            )
        ),
        "nvcc_version": _nvcc_version(),
        "cuda_extension": str(Path(cuda_extension.__file__).resolve()),
        "shared_reducer_warps": int(cuda_extension.get_shared_reducer_warps()),
        "residual_reducer_warps": int(
            cuda_extension.get_residual_reducer_warps()
        ),
        "residual_reducer_force_warps": int(
            cuda_extension.get_residual_reducer_force_warps()
        ),
    }


def _nvcc_version() -> str | None:
    try:
        completed = subprocess.run(
            ("nvcc", "--version"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return next((line for line in reversed(lines) if "release" in line), None)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/cuda-step1.json"))
    parser.add_argument("--docs", type=int, action="append", help="document count; repeat for a sweep")
    parser.add_argument("--candidates", type=int, action="append", help="candidate count; repeat for a sweep")
    parser.add_argument("--min-doc-tokens", type=int)
    parser.add_argument("--max-doc-tokens", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--query-tokens", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a 64-document, two-sample timing matrix",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="run only the dim-128 correctness/workspace phase (sanitizer target)",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="exit nonzero after writing JSON when any applied gate fails",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    defaults = SMOKE_DEFAULTS if args.smoke else {
        "doc_counts": DEFAULT_DOC_COUNTS,
        "candidate_counts": DEFAULT_CANDIDATE_COUNTS,
        "min_doc_tokens": 64,
        "max_doc_tokens": 192,
        "batch": 4,
        "query_tokens": 32,
        "k": 10,
        "warmup": 5,
        "repeat": 20,
        "runs": 3,
    }
    return _validate_config(
        doc_counts=args.docs or defaults["doc_counts"],
        candidate_counts=args.candidates or defaults["candidate_counts"],
        min_doc_tokens=(
            defaults["min_doc_tokens"]
            if args.min_doc_tokens is None
            else args.min_doc_tokens
        ),
        max_doc_tokens=(
            defaults["max_doc_tokens"]
            if args.max_doc_tokens is None
            else args.max_doc_tokens
        ),
        batch=defaults["batch"] if args.batch is None else args.batch,
        query_tokens=(
            defaults["query_tokens"]
            if args.query_tokens is None
            else args.query_tokens
        ),
        k=defaults["k"] if args.k is None else args.k,
        warmup=defaults["warmup"] if args.warmup is None else args.warmup,
        repeat=defaults["repeat"] if args.repeat is None else args.repeat,
        runs=defaults["runs"] if args.runs is None else args.runs,
        seed=args.seed,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = _config_from_args(args)
        result = run_benchmark(
            output_path=args.output,
            correctness_only=args.correctness_only,
            performance_gates_enabled=not (args.smoke or args.correctness_only),
            **config,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gpu": result["metadata"]["gpu"],
                "correctness_only": args.correctness_only,
                "gate_passed": result["gate_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_gate and not result["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
