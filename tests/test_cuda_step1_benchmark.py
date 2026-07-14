import numpy as np
import pytest

from benchmarks.run_cuda_step1 import (
    DEFAULT_CANDIDATE_COUNTS,
    DEFAULT_DOC_COUNTS,
    _build_parser,
    _candidate_counts_for_docs,
    _config_from_args,
    _evaluate_performance_gates,
    _summarize_samples,
    _validate_config,
)


def test_cuda_step1_cli_defaults_match_release_matrix():
    args = _build_parser().parse_args([])

    config = _config_from_args(args)

    assert config == {
        "doc_counts": list(DEFAULT_DOC_COUNTS),
        "candidate_counts": list(DEFAULT_CANDIDATE_COUNTS),
        "min_doc_tokens": 64,
        "max_doc_tokens": 192,
        "batch": 4,
        "query_tokens": 32,
        "k": 10,
        "warmup": 5,
        "repeat": 20,
        "runs": 3,
        "seed": 20260713,
    }
    assert args.correctness_only is False
    assert args.fail_on_gate is False


def test_cuda_step1_smoke_defaults_and_explicit_overrides():
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--docs",
            "32",
            "--candidates",
            "8",
            "--candidates",
            "32",
            "--k",
            "8",
            "--repeat",
            "1",
            "--correctness-only",
        ]
    )

    config = _config_from_args(args)

    assert config["doc_counts"] == [32]
    assert config["candidate_counts"] == [8, 32]
    assert config["min_doc_tokens"] == 4
    assert config["max_doc_tokens"] == 16
    assert config["batch"] == 1
    assert config["query_tokens"] == 4
    assert config["k"] == 8
    assert config["repeat"] == 1
    assert args.correctness_only is True


def test_cuda_step1_candidate_filter_and_config_validation():
    assert _candidate_counts_for_docs((32, 128, 512), 64, 10) == (32,)
    with pytest.raises(ValueError, match="no candidate count"):
        _candidate_counts_for_docs((4, 128), 64, 10)

    with pytest.raises(ValueError, match="max_doc_tokens"):
        _validate_config(
            doc_counts=(64,),
            candidate_counts=(16,),
            min_doc_tokens=8,
            max_doc_tokens=4,
            batch=1,
            query_tokens=4,
            k=4,
            warmup=0,
            repeat=1,
            runs=1,
            seed=1,
        )


def test_cuda_step1_latency_summary_keeps_raw_samples_and_percentiles():
    summary = _summarize_samples([[4.0, 2.0], [3.0, 1.0]])

    assert summary["latency_best_ms"] == 1.0
    assert summary["latency_median_ms"] == 2.5
    assert summary["latency_p50_ms"] == 2.5
    assert summary["latency_p95_ms"] == pytest.approx(
        np.percentile([4.0, 2.0, 3.0, 1.0], 95)
    )
    assert summary["latency_run_medians_ms"] == [3.0, 2.0]
    assert summary["latency_samples_ms"] == [[4.0, 2.0], [3.0, 1.0]]

    with pytest.raises(ValueError, match="rectangular"):
        _summarize_samples([[1.0], [1.0, 2.0]])


def test_cuda_step1_performance_gates_have_explicit_direction_and_failure():
    rows = [
        _timing("int4_tensor_full_fp32", 10.0),
        _timing("int4_per_token_full_fp32", 12.0),
        _timing("int4_tensor_topk_fp32", 11.0),
        _timing("int4_per_token_topk_fp32", 12.0),
        _timing("int4_per_token_topk_int8", 6.0),
        _timing("int4_residual_full_fp32", 20.0),
    ]
    for candidates, token_ms, residual_ms, cascade_ms in (
        (32, 1.0, 2.0, 10.0),
        (128, 2.0, 4.0, 11.0),
        (512, 4.0, 8.0, 15.0),
        (2048, 8.0, 16.0, 25.0),
        # The full-corpus cascade bypasses the coarse scan, so it may be
        # faster than a near-full candidate cascade and is not monotonic.
        (4096, 16.0, 32.0, 20.0),
    ):
        rows.extend(
            [
                _timing("int4_per_token_candidates_fp32", token_ms, candidates),
                _timing("int4_residual_candidates_fp32", residual_ms, candidates),
                _timing("int4_residual_cascade", cascade_ms, candidates),
            ]
        )

    gates = _evaluate_performance_gates(rows)

    assert all(gate["passed"] for gate in gates if gate["applied"])
    int8_gate = next(
        gate for gate in gates if gate["name"] == "per_token_int8_topk_speedup"
    )
    assert int8_gate["comparison"] == ">="
    assert int8_gate["value"] == 2.0
    candidate_gate = next(
        gate
        for gate in gates
        if gate["name"] == "residual_candidate_speedup_at_512"
    )
    assert candidate_gate["value"] == 2.5

    full_cascade = next(
        row
        for row in rows
        if row["operation"] == "int4_residual_cascade"
        and row["candidate_count"] == 4096
    )
    full_cascade["max_abs_error_vs_full_scores"] = 1e-4
    full_cascade["indices_exact_vs_stable_full"] = True
    assert all(
        gate["passed"]
        for gate in _evaluate_performance_gates(rows)
        if gate["applied"]
    )

    failing = [dict(row) for row in rows]
    next(
        row for row in failing if row["operation"] == "int4_per_token_full_fp32"
    )["latency_p50_ms"] = 13.0
    failed_gates = _evaluate_performance_gates(failing)
    overhead = next(
        gate for gate in failed_gates if gate["name"] == "per_token_full_over_tensor"
    )
    assert overhead["passed"] is False

    bad_full_cascade = [dict(row) for row in rows]
    next(
        row
        for row in bad_full_cascade
        if row["operation"] == "int4_residual_cascade"
        and row["candidate_count"] == 4096
    )["indices_exact_vs_stable_full"] = False
    cascade_gates = _evaluate_performance_gates(bad_full_cascade)
    output_gate = next(
        gate
        for gate in cascade_gates
        if gate["name"] == "output_validity"
        and gate["operation"] == "int4_residual_cascade"
        and gate["candidate_count"] == 4096
    )
    assert output_gate["passed"] is False

    missing_full_cascade_parity = [dict(row) for row in rows]
    incomplete = next(
        row
        for row in missing_full_cascade_parity
        if row["operation"] == "int4_residual_cascade"
        and row["candidate_count"] == 4096
    )
    del incomplete["max_abs_error_vs_full_scores"]
    del incomplete["indices_exact_vs_stable_full"]
    missing_gate = next(
        gate
        for gate in _evaluate_performance_gates(missing_full_cascade_parity)
        if gate["name"] == "output_validity"
        and gate["operation"] == "int4_residual_cascade"
        and gate["candidate_count"] == 4096
    )
    assert missing_gate["passed"] is False
    assert missing_gate["missing_fields"] == [
        "max_abs_error_vs_full_scores",
        "indices_exact_vs_stable_full",
    ]

    missing_full_parity = [dict(row) for row in rows]
    incomplete = next(
        row
        for row in missing_full_parity
        if row["operation"] == "int4_tensor_full_fp32"
    )
    del incomplete["max_abs_error_vs_reference_scores"]
    missing_gate = next(
        gate
        for gate in _evaluate_performance_gates(missing_full_parity)
        if gate["name"] == "output_validity"
        and gate["operation"] == "int4_tensor_full_fp32"
    )
    assert missing_gate["passed"] is False
    assert missing_gate["value"] is None
    assert missing_gate["missing_fields"] == [
        "max_abs_error_vs_reference_scores"
    ]


def _timing(operation, p50, candidates=None):
    if "topk" in operation:
        scope = "topk"
    elif "candidates" in operation:
        scope = "candidates"
    elif "cascade" in operation:
        scope = "cascade"
    else:
        scope = "full"
    row = {
        "doc_count": 4096,
        "candidate_count": candidates,
        "operation": operation,
        "scope": scope,
        "latency_p50_ms": p50,
        "latency_p95_ms": p50 * 1.1,
        "finite": True,
    }
    if scope == "topk":
        row["max_abs_error_vs_full_scores"] = 0.0
        row["indices_exact_vs_stable_full"] = True
    elif scope == "full":
        row["max_abs_error_vs_reference_scores"] = 0.0
    elif scope == "candidates":
        row["max_abs_error_vs_full_projection"] = 0.0
    elif scope == "cascade":
        row["max_abs_error_vs_full_scores"] = 0.0
        if candidates == 4096:
            row["indices_exact_vs_stable_full"] = True
    return row
