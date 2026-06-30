import json

import pytest

from benchmarks.run_synthetic import _benchmark_devices, _stage_config, run_stage


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
    assert all(row["correctness_delta"] <= row["correctness_tolerance"] for row in result["results"] if row["gate_blocking"])


def test_larger_benchmark_stage_requires_prior_gate_json(tmp_path):
    with pytest.raises(RuntimeError, match="requires a passing cuda-smoke gate"):
        run_stage("cuda-sweep", output_path=tmp_path / "sweep.json", gate_path=tmp_path / "missing.json")


def test_vast_large_requires_prior_cuda_sweep_gate_json(tmp_path):
    cuda_smoke_gate = tmp_path / "cuda-smoke.json"
    cuda_smoke_gate.write_text('{"stage": "cuda-smoke", "gate_passed": true}')

    with pytest.raises(RuntimeError, match="requires a passing cuda-sweep gate"):
        run_stage("vast-large", output_path=tmp_path / "large.json", gate_path=cuda_smoke_gate)


@pytest.mark.benchmark_smoke
def test_cpu_smoke_gate_allows_float32_accumulation_noise(tmp_path):
    result = run_stage("cpu-smoke", output_path=tmp_path / "cpu-smoke.json")

    assert result["gate_passed"] is True
    assert all(row["correctness_delta"] <= row["correctness_tolerance"] for row in result["results"] if row["gate_blocking"])


@pytest.mark.benchmark_smoke
def test_cpu_smoke_includes_pytorch_style_baselines_and_ratios(tmp_path):
    result = run_stage("cpu-smoke", output_path=tmp_path / "cpu-smoke.json")

    rows_by_shape = {}
    for row in result["results"]:
        shape_key = (row["dim"], row["query_tokens"], row["doc_tokens"], row["docs"], row["dtype"])
        rows_by_shape.setdefault(shape_key, {})[row["implementation"]] = row

    for implementations in rows_by_shape.values():
        assert "torch_fp16_baseline" in implementations
        assert "torch_int8_baseline" in implementations
        native = implementations["bitmax_native"]
        fp16 = implementations["torch_fp16_baseline"]
        int8 = implementations["torch_int8_baseline"]
        assert native["speedup_vs_torch_fp16"] == pytest.approx(fp16["latency_ms"] / native["latency_ms"])
        assert native["speedup_vs_torch_int8"] == pytest.approx(int8["latency_ms"] / native["latency_ms"])
        assert native["doc_memory_compression_vs_fp16"] == pytest.approx(16.0)
        assert native["doc_memory_compression_vs_fp32"] == pytest.approx(32.0)
        assert native["baseline_latency_ms"]["torch_fp16_baseline"] == fp16["latency_ms"]
        assert native["baseline_latency_ms"]["torch_int8_baseline"] == int8["latency_ms"]


def test_stage0_json_records_benchmark_schema_version(tmp_path):
    result = run_stage("stage0", output_path=tmp_path / "stage0.json")

    assert result["schema_version"] == 2
    assert result["baselines"] == ["python_reference", "torch_fp16_baseline", "torch_int8_baseline"]


def test_torch_style_baseline_rows_describe_backend_and_formula(tmp_path):
    result = run_stage("stage0", output_path=tmp_path / "stage0.json")

    baseline_rows = [row for row in result["results"] if row["implementation"].startswith("torch_")]
    assert baseline_rows
    for row in baseline_rows:
        assert row["baseline_backend"] in {"torch", "numpy_torch_equivalent"}
        assert row["baseline_device"] in {"cpu", "cuda"}
        assert row["requested_baseline_device"] in {"cpu", "cuda"}
        assert row["formula"] in {"dense_fp16_vectorized_maxsim", "dense_int8_vectorized_doc_maxsim"}
        assert row["gate_blocking"] is False


def test_torch_cuda_device_resolver_rejects_unsupported_arch(monkeypatch):
    import benchmarks.run_synthetic as synthetic

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (12, 0)

        @staticmethod
        def get_arch_list():
            return ["sm_80", "sm_90"]

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(name):
            return name

    monkeypatch.setattr(synthetic, "_torch", FakeTorch())

    assert synthetic._resolve_torch_device("cuda") is None


def test_cuda_sweep_stage_uses_larger_shapes_than_smoke():
    specs, repeat = _stage_config("cuda-sweep")

    assert repeat >= 7
    assert len(specs) >= 3
    assert max(spec["docs"] for spec in specs) >= 1_024
    assert max(spec["doc_tokens"] for spec in specs) >= 64
    assert {spec["dim"] for spec in specs} >= {128, 256}


def test_cuda_stages_request_cuda_for_baselines_and_native_kernel():
    assert _benchmark_devices("cpu-smoke") == ("auto", "cpu")
    assert _benchmark_devices("cuda-smoke") == ("cuda", "cuda")
    assert _benchmark_devices("cuda-sweep") == ("cuda", "cuda")
    assert _benchmark_devices("vast-large") == ("cuda", "cuda")
