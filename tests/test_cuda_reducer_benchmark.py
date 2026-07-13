from types import SimpleNamespace

import pytest

from benchmarks.run_cuda_reducers import _configure_shared_reducer_warps, _summarize_rows


def test_configure_shared_reducer_warps_uses_extension_setter_and_getter():
    selected = []
    extension = SimpleNamespace(
        set_shared_reducer_warps=selected.append,
        get_shared_reducer_warps=lambda: selected[-1],
    )

    configuration = _configure_shared_reducer_warps(extension, 4)

    assert selected == [4]
    assert configuration == {"requested": 4, "active": 4, "supported": True}


def test_configure_shared_reducer_warps_preserves_extension_default_when_unspecified():
    selected = []
    extension = SimpleNamespace(
        set_shared_reducer_warps=selected.append,
        get_shared_reducer_warps=lambda: 8,
    )

    configuration = _configure_shared_reducer_warps(extension, None)

    assert selected == []
    assert configuration == {"requested": None, "active": 8, "supported": True}


def test_configure_shared_reducer_warps_is_backward_compatible_and_validates():
    assert _configure_shared_reducer_warps(SimpleNamespace(), 0) == {
        "requested": 0,
        "active": None,
        "supported": False,
    }
    with pytest.raises(ValueError, match="0, 4, or 8"):
        _configure_shared_reducer_warps(SimpleNamespace(), 2)


def test_summarize_rows_reports_ratios_with_explicit_directions():
    rows = [
        _row("binary", "full", "maxsim", "legacy_maxsim", 2.0),
        _row("binary", "full", "maxsim", "shared_reducer_policy", 1.0),
        _row("binary", "candidates", "maxsim", "shared_reducer_policy", 0.25),
        _row("binary", "full", "smoothsim", "shared_reducer_policy", 3.0),
        _row("binary", "candidates", "smoothsim", "shared_reducer_policy", 1.0),
        _row("int4", "full", "maxsim", "legacy_maxsim", 4.0),
        _row("int4", "full", "maxsim", "shared_reducer_policy", 5.0),
        _row("int4", "candidates", "maxsim", "shared_reducer_policy", 2.0),
    ]

    summary = _summarize_rows(rows)

    assert summary["latency_basis"] == "median_ms"
    assert summary["shared_vs_legacy_maxsim_ratio"] == {"binary": 0.5, "int4": 1.25}
    assert summary["candidate_vs_full_speedup"] == {
        "binary": {"maxsim": 4.0, "smoothsim": 3.0},
        "int4": {"maxsim": 2.5},
    }


def _row(format_name, scope, reducer, implementation, latency):
    return {
        "format": format_name,
        "scope": scope,
        "reducer": reducer,
        "implementation": implementation,
        "latency_median_ms": latency,
    }
