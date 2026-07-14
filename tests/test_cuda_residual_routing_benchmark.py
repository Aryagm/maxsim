from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.run_cuda_residual_routing import (
    REQUIRED_PROVENANCE_ENV,
    _adaptive_policy_decisions,
    _release_provenance,
    _validate_args,
)


def _row(mode: int, latency_ms: float) -> dict[str, object]:
    return {
        "query_tokens": 4,
        "reducer": "maxsim",
        "scope": "full",
        "requested_mode": mode,
        "latency_median_ms": latency_ms,
    }


def test_adaptive_policy_regret_gate_enforces_five_percent_limit():
    policy = [{"query_tokens": 4, "actual_route_warps": 4}]

    at_limit = _adaptive_policy_decisions(
        [_row(0, 1.0), _row(4, 1.05), _row(8, 1.2)],
        policy,
        max_regret=0.05,
    )
    over_limit = _adaptive_policy_decisions(
        [_row(0, 1.0), _row(4, 1.051), _row(8, 1.2)],
        policy,
        max_regret=0.05,
    )

    assert at_limit[0]["passed"] is True
    assert over_limit[0]["passed"] is False


def test_release_provenance_requires_every_field(monkeypatch):
    for field, env_name in REQUIRED_PROVENANCE_ENV.items():
        monkeypatch.setenv(env_name, "1" if field == "git_dirty" else field)

    provenance = _release_provenance()

    assert provenance["source"]["git_dirty"] is True
    assert provenance["source"]["source_archive_sha256"] == (
        "source_archive_sha256"
    )
    assert provenance["execution"]["vast_instance_id"] == "vast_instance_id"


def test_routing_benchmark_rejects_empty_batches():
    args = SimpleNamespace(
        docs=4096,
        candidates=512,
        min_doc_tokens=64,
        max_doc_tokens=192,
        batch=0,
        dim=128,
        warmup=5,
        repeat=20,
        runs=3,
        max_adaptive_regret=0.05,
    )

    with pytest.raises(ValueError, match="--batch must be >= 1"):
        _validate_args(args)
