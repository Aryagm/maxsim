import json

import pytest

from benchmarks.run_synthetic import run_stage


@pytest.mark.benchmark_smoke
def test_stage0_benchmark_emits_correctness_and_timing_signal(tmp_path):
    output_path = tmp_path / "stage0.json"

    result = run_stage("stage0", output_path=output_path)

    assert result["stage"] == "stage0"
    assert result["gate_passed"] is True
    assert output_path.exists()
    from_disk = json.loads(output_path.read_text())
    assert from_disk["stage"] == "stage0"
    assert {row["implementation"] for row in result["results"]} >= {"python_reference", "bitmax_native"}
    assert all(row["latency_ms"] >= 0 for row in result["results"])
    assert all(row["correctness_delta"] <= 1e-5 for row in result["results"])


def test_larger_benchmark_stage_requires_prior_gate_json(tmp_path):
    with pytest.raises(RuntimeError, match="requires a passing cuda-smoke gate"):
        run_stage("vast-large", output_path=tmp_path / "large.json", gate_path=tmp_path / "missing.json")


@pytest.mark.benchmark_smoke
def test_cpu_smoke_gate_allows_float32_accumulation_noise(tmp_path):
    result = run_stage("cpu-smoke", output_path=tmp_path / "cpu-smoke.json")

    assert result["gate_passed"] is True
    assert all(row["correctness_delta"] <= row["correctness_tolerance"] for row in result["results"])
