#!/usr/bin/env python3
"""Benchmark residual-int4 CUDA reducer routing on an SM89 GPU.

The large-corpus sweep force-selects the block, four-warp, and eight-warp
implementations so short-query measurements cannot collapse onto the adaptive
route. It compares every forced route against the block implementation and
checks candidate-only output against the corresponding full-score projection.
A small-corpus pass also checks every route directly against CPU scoring.

Default RTX 4090 run::

    python -m benchmarks.run_cuda_residual_routing \
        --output benchmark-results/cuda-residual-routing-rtx4090.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from maxsim.cascade import (
    pack_residual_int4,
    residual_int4_to_device,
    residual_score,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional CUDA benchmark dependency
    torch = None


FORCED_ROUTES = (0, 4, 8)
QUERY_TOKENS = (1, 3, 4, 7, 8, 16, 32)
ADAPTIVE_BASE_MODE = 8
REDUCERS_AT_Q32 = ("maxsim", "topk4", "smoothsim")
TEMPERATURE = 0.7
DEFAULT_MAX_ADAPTIVE_REGRET = 0.05
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
EXPECTED_COMPUTE_CAPABILITY = (8, 9)
REQUIRED_PROVENANCE_ENV = {
    "git_commit": "MAXSIM_GIT_COMMIT",
    "git_dirty": "MAXSIM_GIT_DIRTY",
    "source_archive_sha256": "MAXSIM_SOURCE_ARCHIVE_SHA256",
    "source_archive_path": "MAXSIM_SOURCE_ARCHIVE_PATH",
    "build_command": "MAXSIM_BUILD_COMMAND",
    "benchmark_command": "MAXSIM_BENCHMARK_COMMAND",
    "vast_instance_id": "MAXSIM_VAST_INSTANCE_ID",
    "vast_offer_id": "MAXSIM_VAST_OFFER_ID",
    "container_image": "MAXSIM_CONTAINER_IMAGE",
}


def _reducers(query_tokens: int) -> tuple[str, ...]:
    return REDUCERS_AT_Q32 if query_tokens == 32 else ("maxsim",)


def _kwargs(reducer: str) -> dict[str, Any]:
    return {"temperature": TEMPERATURE} if reducer == "smoothsim" else {}


def _tolerance(reducer: str) -> float:
    return 5e-3 if reducer == "smoothsim" else 2e-3


def _route_name(route_warps: int) -> str:
    return "block" if route_warps == 0 else f"warp{route_warps}"


def _activate_forced_route(
    extension: Any, route_warps: int, query_tokens: int
) -> int:
    extension.set_residual_reducer_force_warps(route_warps)
    return int(extension.get_effective_residual_reducer_warps(query_tokens))


def _max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float32)
                - np.asarray(right, dtype=np.float32)
            ),
            initial=0.0,
        )
    )


def _describe_samples(samples_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "samples": int(values.size),
        "latency_samples_ms": [float(value) for value in values],
        "latency_best_ms": float(values.min()),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_mean_ms": float(values.mean()),
        "latency_std_ms": float(values.std()),
    }


def _measure_interleaved(
    extension: Any,
    calls: dict[int, Callable[[], np.ndarray]],
    *,
    query_tokens: int,
    warmup: int,
    repeat: int,
    runs: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    actual_routes: dict[int, int] = {}
    for route_warps in FORCED_ROUTES:
        actual_routes[route_warps] = _activate_forced_route(
            extension, route_warps, query_tokens
        )
        for _ in range(warmup):
            calls[route_warps]()
    torch.cuda.synchronize()

    samples: dict[int, list[float]] = {mode: [] for mode in FORCED_ROUTES}
    for run in range(runs):
        for iteration in range(repeat):
            start = (run + iteration) % len(FORCED_ROUTES)
            order = FORCED_ROUTES[start:] + FORCED_ROUTES[:start]
            for route_warps in order:
                _activate_forced_route(extension, route_warps, query_tokens)
                torch.cuda.synchronize()
                begin = time.perf_counter_ns()
                calls[route_warps]()
                torch.cuda.synchronize()
                samples[route_warps].append(
                    (time.perf_counter_ns() - begin) / 1e6
                )
    timing = {
        mode: _describe_samples(samples[mode]) for mode in FORCED_ROUTES
    }
    return timing, actual_routes


def _make_corpus(
    rng: np.random.Generator,
    *,
    docs: int,
    min_doc_tokens: int,
    max_doc_tokens: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = rng.integers(
        min_doc_tokens,
        max_doc_tokens + 1,
        size=docs,
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    embeddings = rng.standard_normal((int(offsets[-1]), dim), dtype=np.float32)
    return embeddings, offsets, lengths


def _adaptive_policy(extension: Any) -> list[dict[str, Any]]:
    extension.set_residual_reducer_warps(ADAPTIVE_BASE_MODE)
    extension.set_residual_reducer_force_warps(-1)
    policy: list[dict[str, Any]] = []
    for query_tokens in QUERY_TOKENS:
        actual_route = int(
            extension.get_effective_residual_reducer_warps(query_tokens)
        )
        policy.append(
            {
                "query_tokens": query_tokens,
                "base_mode": ADAPTIVE_BASE_MODE,
                "force_mode": -1,
                "actual_route_warps": actual_route,
                "actual_route": _route_name(actual_route),
            }
        )
    return policy


def _cpu_spot_checks(
    extension: Any, *, seed: int, dim: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 1)
    docs, offsets, _ = _make_corpus(
        rng,
        docs=24,
        min_doc_tokens=4,
        max_doc_tokens=12,
        dim=dim,
    )
    cpu_packed = pack_residual_int4(docs, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)
    query_max = rng.standard_normal(
        (2, max(QUERY_TOKENS), dim), dtype=np.float32
    )
    candidates = np.ascontiguousarray(
        rng.choice(24, size=8, replace=False), dtype=np.int64
    )
    checks: list[dict[str, Any]] = []

    for query_tokens in QUERY_TOKENS:
        query = np.ascontiguousarray(query_max[:, :query_tokens, :])
        for reducer in _reducers(query_tokens):
            kwargs = _kwargs(reducer)
            for scope, selected in (("full", None), ("candidates", candidates)):
                expected = residual_score(
                    query,
                    cpu_packed,
                    device="cpu",
                    reducer=reducer,
                    candidate_indices=selected,
                    **kwargs,
                )
                for requested_route in FORCED_ROUTES:
                    actual_route = _activate_forced_route(
                        extension, requested_route, query_tokens
                    )
                    actual = residual_score(
                        query,
                        cuda_packed,
                        device="cuda",
                        reducer=reducer,
                        candidate_indices=selected,
                        **kwargs,
                    )
                    error = _max_abs_delta(actual, expected)
                    tolerance = _tolerance(reducer)
                    finite = bool(np.isfinite(actual).all())
                    route_matched = actual_route == requested_route
                    checks.append(
                        {
                            "query_tokens": query_tokens,
                            "reducer": reducer,
                            "scope": scope,
                            "requested_mode": requested_route,
                            "requested_force_warps": requested_route,
                            "requested_route": _route_name(requested_route),
                            "actual_route_warps": actual_route,
                            "actual_route": _route_name(actual_route),
                            "route_matched": route_matched,
                            "max_abs_error_vs_cpu": error,
                            "tolerance": tolerance,
                            "finite": finite,
                            "passed": bool(
                                finite and route_matched and error <= tolerance
                            ),
                        }
                    )
    del cuda_packed
    torch.cuda.empty_cache()
    return checks


def _performance_sweep(
    extension: Any,
    *,
    docs: int,
    min_doc_tokens: int,
    max_doc_tokens: int,
    candidates: int,
    batch: int,
    dim: int,
    seed: int,
    warmup: int,
    repeat: int,
    runs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    embeddings, offsets, lengths = _make_corpus(
        rng,
        docs=docs,
        min_doc_tokens=min_doc_tokens,
        max_doc_tokens=max_doc_tokens,
        dim=dim,
    )
    cpu_packed = pack_residual_int4(embeddings, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)
    query_max = rng.standard_normal(
        (batch, max(QUERY_TOKENS), dim), dtype=np.float32
    )
    candidate_ids = np.ascontiguousarray(
        rng.choice(docs, size=candidates, replace=False), dtype=np.int64
    )
    del embeddings, cpu_packed

    rows: list[dict[str, Any]] = []
    for query_tokens in QUERY_TOKENS:
        query = np.ascontiguousarray(query_max[:, :query_tokens, :])
        for reducer in _reducers(query_tokens):
            kwargs = _kwargs(reducer)
            outputs: dict[tuple[str, int], np.ndarray] = {}
            calls_by_scope: dict[
                str, dict[int, Callable[[], np.ndarray]]
            ] = {}

            for scope, selected in (("full", None), ("candidates", candidate_ids)):
                calls: dict[int, Callable[[], np.ndarray]] = {}
                for requested_route in FORCED_ROUTES:

                    def call(
                        *,
                        selected=selected,
                        query=query,
                        reducer=reducer,
                        kwargs=kwargs,
                    ) -> np.ndarray:
                        return residual_score(
                            query,
                            cuda_packed,
                            device="cuda",
                            reducer=reducer,
                            candidate_indices=selected,
                            **kwargs,
                        )

                    calls[requested_route] = call
                    _activate_forced_route(
                        extension, requested_route, query_tokens
                    )
                    outputs[(scope, requested_route)] = call()
                calls_by_scope[scope] = calls

            for scope in ("full", "candidates"):
                timing, actual_routes = _measure_interleaved(
                    extension,
                    calls_by_scope[scope],
                    query_tokens=query_tokens,
                    warmup=warmup,
                    repeat=repeat,
                    runs=runs,
                )
                block = outputs[(scope, 0)]
                for requested_route in FORCED_ROUTES:
                    output = outputs[(scope, requested_route)]
                    delta_vs_block = _max_abs_delta(output, block)
                    projection_delta = None
                    if scope == "candidates":
                        projection_delta = _max_abs_delta(
                            output,
                            np.take(
                                outputs[("full", requested_route)],
                                candidate_ids,
                                axis=-1,
                            ),
                        )
                    tolerance = _tolerance(reducer)
                    finite = bool(np.isfinite(output).all())
                    actual_route = actual_routes[requested_route]
                    route_matched = actual_route == requested_route
                    correct = bool(
                        finite
                        and route_matched
                        and delta_vs_block <= tolerance
                        and (
                            projection_delta is None
                            or projection_delta <= tolerance
                        )
                    )
                    rows.append(
                        {
                            "query_tokens": query_tokens,
                            "reducer": reducer,
                            "scope": scope,
                            "targets": docs if scope == "full" else candidates,
                            "requested_mode": requested_route,
                            "requested_force_warps": requested_route,
                            "requested_route": _route_name(requested_route),
                            "actual_route_warps": actual_route,
                            "actual_route": _route_name(actual_route),
                            "route_matched": route_matched,
                            **timing[requested_route],
                            "max_abs_delta_vs_block": delta_vs_block,
                            "max_abs_delta_vs_full_projection": projection_delta,
                            "tolerance": tolerance,
                            "finite": finite,
                            "correctness_passed": correct,
                        }
                    )

    shape = {
        "batch": batch,
        "docs": docs,
        "total_doc_tokens": int(offsets[-1]),
        "min_doc_tokens": int(lengths.min()),
        "max_doc_tokens": int(lengths.max()),
        "mean_doc_tokens": float(lengths.mean()),
        "candidate_docs": candidates,
        "dim": dim,
        "query_tokens": list(QUERY_TOKENS),
    }
    del cuda_packed
    torch.cuda.empty_cache()
    return shape, rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted(
        {(r["query_tokens"], r["reducer"], r["scope"]) for r in rows}
    )
    for query_tokens, reducer, scope in keys:
        group = [
            row
            for row in rows
            if (row["query_tokens"], row["reducer"], row["scope"])
            == (query_tokens, reducer, scope)
        ]
        by_mode = {row["requested_mode"]: row for row in group}
        block_ms = float(by_mode[0]["latency_median_ms"])
        best = min(group, key=lambda row: float(row["latency_median_ms"]))
        summaries.append(
            {
                "query_tokens": query_tokens,
                "reducer": reducer,
                "scope": scope,
                "best_requested_mode": int(best["requested_mode"]),
                "best_actual_route_warps": int(best["actual_route_warps"]),
                "best_median_ms": float(best["latency_median_ms"]),
                "best_speedup_vs_block": block_ms
                / float(best["latency_median_ms"]),
                "mode4_speedup_vs_block": block_ms
                / float(by_mode[4]["latency_median_ms"]),
                "mode8_speedup_vs_block": block_ms
                / float(by_mode[8]["latency_median_ms"]),
            }
        )
    return summaries


def _adaptive_policy_decisions(
    rows: list[dict[str, Any]],
    adaptive_policy: list[dict[str, Any]],
    *,
    max_regret: float,
) -> list[dict[str, Any]]:
    route_by_query = {
        int(row["query_tokens"]): int(row["actual_route_warps"])
        for row in adaptive_policy
    }
    decisions: list[dict[str, Any]] = []
    keys = sorted(
        {(r["query_tokens"], r["reducer"], r["scope"]) for r in rows}
    )
    for query_tokens, reducer, scope in keys:
        group = [
            row
            for row in rows
            if (row["query_tokens"], row["reducer"], row["scope"])
            == (query_tokens, reducer, scope)
        ]
        by_mode = {int(row["requested_mode"]): row for row in group}
        missing_modes = [mode for mode in FORCED_ROUTES if mode not in by_mode]
        adaptive_mode = route_by_query.get(int(query_tokens))
        adaptive_row = by_mode.get(adaptive_mode) if adaptive_mode is not None else None

        invalid_latency_modes = [
            mode
            for mode, row in by_mode.items()
            if not np.isfinite(float(row["latency_median_ms"]))
            or float(row["latency_median_ms"]) <= 0.0
        ]
        valid_rows = [
            row
            for mode, row in by_mode.items()
            if mode not in invalid_latency_modes
        ]
        best_row = (
            min(valid_rows, key=lambda row: float(row["latency_median_ms"]))
            if valid_rows
            else None
        )
        best_ms = (
            float(best_row["latency_median_ms"])
            if best_row is not None
            else None
        )
        adaptive_ms = (
            float(adaptive_row["latency_median_ms"])
            if adaptive_row is not None
            and np.isfinite(float(adaptive_row["latency_median_ms"]))
            and float(adaptive_row["latency_median_ms"]) > 0.0
            else None
        )
        regret = (
            max(0.0, adaptive_ms / best_ms - 1.0)
            if adaptive_ms is not None and best_ms is not None
            else None
        )
        passed = bool(
            not missing_modes
            and not invalid_latency_modes
            and adaptive_row is not None
            and best_row is not None
            and regret is not None
            and regret <= max_regret + 1e-12
        )
        decisions.append(
            {
                "query_tokens": int(query_tokens),
                "reducer": reducer,
                "scope": scope,
                "adaptive_route_warps": adaptive_mode,
                "adaptive_route": (
                    _route_name(adaptive_mode)
                    if adaptive_mode is not None
                    else None
                ),
                "adaptive_median_ms": adaptive_ms,
                "best_route_warps": (
                    int(best_row["requested_mode"])
                    if best_row is not None
                    else None
                ),
                "best_route": (
                    _route_name(int(best_row["requested_mode"]))
                    if best_row is not None
                    else None
                ),
                "best_median_ms": best_ms,
                "regret_fraction": regret,
                "max_regret_fraction": max_regret,
                "missing_forced_routes": missing_modes,
                "invalid_latency_routes": invalid_latency_modes,
                "passed": passed,
            }
        )
    return decisions


def _parse_bool_env(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _release_provenance() -> dict[str, dict[str, Any]]:
    values = {
        field: os.environ.get(env_name, "").strip()
        for field, env_name in REQUIRED_PROVENANCE_ENV.items()
    }
    missing = [
        REQUIRED_PROVENANCE_ENV[field]
        for field, value in values.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "release routing benchmark requires nonempty provenance variables: "
            + ", ".join(missing)
        )
    git_dirty = _parse_bool_env(
        REQUIRED_PROVENANCE_ENV["git_dirty"], values["git_dirty"]
    )
    return {
        "source": {
            "git_commit": values["git_commit"],
            "git_dirty": git_dirty,
            "source_archive_sha256": values["source_archive_sha256"],
            "source_archive_path": values["source_archive_path"],
        },
        "execution": {
            "build_command": values["build_command"],
            "benchmark_command": values["benchmark_command"],
            "vast_instance_id": values["vast_instance_id"],
            "vast_offer_id": values["vast_offer_id"],
            "container_image": values["container_image"],
        },
    }


def _command_output(command: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output or None


def _nvcc_version() -> str | None:
    output = _command_output(("nvcc", "--version"))
    if output is None:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return next((line for line in reversed(lines) if "release" in line), None)


def _runtime_toolchain() -> dict[str, str]:
    driver_version = _command_output(
        (
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        )
    )
    nvcc_version = _nvcc_version()
    missing = [
        name
        for name, value in (
            ("NVIDIA driver version", driver_version),
            ("nvcc version", nvcc_version),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(
            "release routing benchmark could not determine " + ", ".join(missing)
        )
    return {
        "driver_version": driver_version,
        "nvcc_version": nvcc_version,
    }


def _validate_release_gpu(torch_module: Any) -> tuple[str, tuple[int, int]]:
    gpu_name = torch_module.cuda.get_device_name(0)
    compute_capability = tuple(torch_module.cuda.get_device_capability(0))
    if (
        gpu_name != EXPECTED_GPU_NAME
        or compute_capability != EXPECTED_COMPUTE_CAPABILITY
    ):
        raise RuntimeError(
            "this release routing benchmark requires "
            f"{EXPECTED_GPU_NAME} with SM{EXPECTED_COMPUTE_CAPABILITY[0]}"
            f"{EXPECTED_COMPUTE_CAPABILITY[1]}; found {gpu_name} with "
            f"compute capability {compute_capability}"
        )
    return gpu_name, compute_capability


def _validate_args(args: argparse.Namespace) -> None:
    if args.docs < 1:
        raise ValueError("--docs must be >= 1")
    if not (1 <= args.candidates <= args.docs):
        raise ValueError("--candidates must satisfy 1 <= candidates <= docs")
    if args.min_doc_tokens < 1:
        raise ValueError("--min-doc-tokens must be >= 1")
    if args.max_doc_tokens < args.min_doc_tokens:
        raise ValueError("--max-doc-tokens must be >= --min-doc-tokens")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    if args.dim < 1 or args.dim % 8 != 0:
        raise ValueError("--dim must be positive and divisible by 8")
    if args.warmup < 0 or args.repeat < 1 or args.runs < 1:
        raise ValueError("warmup must be >= 0; repeat and runs must be >= 1")
    if not np.isfinite(args.max_adaptive_regret) or args.max_adaptive_regret < 0.0:
        raise ValueError("--max-adaptive-regret must be finite and >= 0")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(
            "this benchmark requires a CUDA-enabled PyTorch runtime"
        )
    gpu_name, compute_capability = _validate_release_gpu(torch)
    provenance = _release_provenance()
    toolchain = _runtime_toolchain()
    extension = importlib.import_module("maxsim._maxsim_cuda")
    prior_mode = int(extension.get_residual_reducer_warps())
    prior_force_mode = int(extension.get_residual_reducer_force_warps())
    restored_mode: int | None = None
    restored_force_mode: int | None = None
    try:
        adaptive_policy = _adaptive_policy(extension)
        cpu_checks = _cpu_spot_checks(extension, seed=args.seed, dim=args.dim)
        shape, rows = _performance_sweep(
            extension,
            docs=args.docs,
            min_doc_tokens=args.min_doc_tokens,
            max_doc_tokens=args.max_doc_tokens,
            candidates=args.candidates,
            batch=args.batch,
            dim=args.dim,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
            runs=args.runs,
        )
    finally:
        extension.set_residual_reducer_warps(prior_mode)
        extension.set_residual_reducer_force_warps(prior_force_mode)
        restored_mode = int(extension.get_residual_reducer_warps())
        restored_force_mode = int(
            extension.get_residual_reducer_force_warps()
        )

    cpu_gate = all(row["passed"] for row in cpu_checks)
    sweep_gate = all(row["correctness_passed"] for row in rows)
    route_gate = all(row["route_matched"] for row in cpu_checks + rows)
    policy_decisions = _adaptive_policy_decisions(
        rows,
        adaptive_policy,
        max_regret=args.max_adaptive_regret,
    )
    policy_gate = bool(policy_decisions) and all(
        decision["passed"] for decision in policy_decisions
    )
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 2,
        "benchmark": "residual_int4_cuda_routing_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **provenance,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": gpu_name,
            "compute_capability": list(compute_capability),
            "gpu_memory_bytes": int(properties.total_memory),
            **toolchain,
        },
        "configuration": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "runs": args.runs,
            "temperature": TEMPERATURE,
            "forced_routes": list(FORCED_ROUTES),
            "adaptive_base_mode": ADAPTIVE_BASE_MODE,
            "max_adaptive_regret_fraction": args.max_adaptive_regret,
            "extension_prior_mode": prior_mode,
            "extension_prior_force_mode": prior_force_mode,
            "restored_mode": restored_mode,
            "restored_force_mode": restored_force_mode,
            "timing_method": (
                "interleaved wall-clock API latency with CUDA synchronization; "
                "forced-route order rotates per sample"
            ),
        },
        "shape": shape,
        "adaptive_policy": adaptive_policy,
        "gate_passed": bool(
            cpu_gate and sweep_gate and route_gate and policy_gate
        ),
        "gates": {
            "cpu_spot_checks_passed": cpu_gate,
            "large_sweep_mode_and_projection_checks_passed": sweep_gate,
            "forced_routes_executed_as_requested": route_gate,
            "adaptive_policy_regret_passed": policy_gate,
        },
        "summary": _summarize(rows),
        "adaptive_policy_decisions": policy_decisions,
        "cpu_spot_checks": cpu_checks,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs", type=int, default=4096)
    parser.add_argument("--min-doc-tokens", type=int, default=64)
    parser.add_argument("--max-doc-tokens", type=int, default=192)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--max-adaptive-regret",
        type=float,
        default=DEFAULT_MAX_ADAPTIVE_REGRET,
        help=(
            "maximum allowed adaptive-route median latency regret relative "
            "to the best forced route (default: 0.05)"
        ),
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gate_passed": result["gate_passed"],
                "gpu": result["environment"]["gpu"],
                "rows": len(result["rows"]),
                "cpu_spot_checks": len(result["cpu_spot_checks"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
