import json

import numpy as np

import benchmarks.compare_open_source as compare_open_source
from benchmarks.compare_open_source import main


def _write_fixture(path):
    doc_embeddings = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    query_embeddings = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
            [[1, -1, 1, -1, 1, -1, 1, -1]],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 3, 6], dtype=np.int64),
        query_embeddings=query_embeddings,
        qrels=np.eye(3, dtype=np.float32),
        doc_ids=np.array(["positive", "negative", "mixed"]),
        dataset_name=np.array("oss-fixture"),
    )


def test_compare_open_source_emits_dense_and_bitmax_rows(tmp_path, capsys):
    input_path = tmp_path / "fixture.npz"
    output_path = tmp_path / "oss.json"
    _write_fixture(input_path)

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--implementations",
            "dense_fp16,bitmax_binary,bitmax_int4",
            "--metric-ks",
            "1,5,10",
            "--repeat",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert "dense_fp16_baseline" in captured.out
    assert "bitmax_binary" in captured.out
    data = json.loads(output_path.read_text())
    rows = {row["implementation"]: row for row in data["results"]}
    assert data["benchmark"] == "open_source_comparison"
    assert rows["dense_fp16_baseline"]["ndcg_at_10"] == 1.0
    assert rows["dense_fp16_baseline"]["recall_at_5"] == 1.0
    assert rows["dense_fp16_baseline"]["ndcg_at_5"] == 1.0
    assert len(rows["dense_fp16_baseline"]["latency_samples_ms"]) == 3
    assert rows["dense_fp16_baseline"]["latency_p50_ms"] >= 0.0
    assert rows["dense_fp16_baseline"]["latency_p95_ms"] >= rows["dense_fp16_baseline"]["latency_p50_ms"]
    assert rows["dense_fp16_baseline"]["latency_p99_ms"] >= rows["dense_fp16_baseline"]["latency_p95_ms"]
    buckets = rows["dense_fp16_baseline"]["token_bucket_quality_at_10"]
    assert set(buckets) == {"short", "medium", "long"}
    assert buckets["short"]["queries"] == 1
    assert buckets["long"]["mean_relevant_doc_tokens"] == 3.0
    assert rows["bitmax_binary"]["doc_memory_compression_vs_fp32"] == 32.0
    assert rows["bitmax_int4"]["implementation_kind"] == "bitmax_sdk"


def test_compare_open_source_emits_unavailable_optional_competitors(tmp_path, capsys):
    input_path = tmp_path / "fixture.npz"
    output_path = tmp_path / "oss.json"
    _write_fixture(input_path)

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--implementations",
            "dense_fp16,qdrant_multivector,cuvs_pooled,colbert_plaid,fast_plaid,vespa_multivector",
            "--allow-unavailable",
        ]
    )

    captured = capsys.readouterr()
    assert "qdrant_multivector" in captured.out
    data = json.loads(output_path.read_text())
    rows = {row["implementation"]: row for row in data["results"]}
    assert rows["qdrant_multivector"]["status"] in {"ok", "unavailable"}
    assert rows["cuvs_cpu_mean_pool_flat_ip"]["status"] in {"ok", "unavailable"}
    assert rows["colbert_plaid"]["status"] == "not_applicable_to_embedding_slice"
    assert rows["fast_plaid"]["status"] in {"ok", "unavailable"}
    assert rows["vespa_multivector"]["status"] == "requires_service_benchmark"


def test_release_cuda_cache_calls_torch_empty_cache_when_cuda_is_available(monkeypatch):
    class FakeCuda:
        def __init__(self):
            self.empty_cache_called = False

        def is_available(self):
            return True

        def empty_cache(self):
            self.empty_cache_called = True

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(compare_open_source, "torch", FakeTorch)

    compare_open_source._release_cuda_cache()

    assert FakeTorch.cuda.empty_cache_called
