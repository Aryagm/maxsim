import json

import pytest

from benchmarks.run_multidataset import (
    DatasetSpec,
    _aggregate_comparison_results,
    _embedding_filename,
    _parse_dataset_specs,
)


def test_embedding_filename_matches_existing_default_cache_name():
    spec = DatasetSpec("vidore/docvqa_test_subsampled")

    assert _embedding_filename(spec, "vidore/colqwen2-v1.0-hf", 64) == "vidore-docvqa-colqwen2-limit64.npz"


def test_parse_dataset_specs_supports_aliases_and_full_dataset_names():
    specs = _parse_dataset_specs("docvqa,vidore/infovqa_test_subsampled")

    assert specs == (
        DatasetSpec("vidore/docvqa_test_subsampled"),
        DatasetSpec("vidore/infovqa_test_subsampled"),
    )


def test_parse_dataset_specs_rejects_unknown_alias():
    with pytest.raises(ValueError, match="unknown dataset alias"):
        _parse_dataset_specs("not-a-dataset")


def test_aggregate_comparison_results_computes_per_implementation_means(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "dataset": {"name": "dataset-a:64", "queries": 64, "docs": 64},
                "results": [
                    {
                        "implementation": "dense_fp16_baseline",
                        "status": "ok",
                        "latency_ms": 100.0,
                        "latency_p50_ms": 110.0,
                        "latency_p95_ms": 130.0,
                        "latency_p99_ms": 140.0,
                        "doc_memory_compression_vs_fp32": 2.0,
                        "recall_at_1": 0.5,
                        "recall_at_5": 0.7,
                        "recall_at_10": 0.8,
                        "ndcg_at_5": 0.6,
                        "ndcg_at_10": 0.7,
                        "speedup_vs_dense_fp16": 1.0,
                        "token_bucket_quality_at_10": {
                            "short": {"queries": 1, "recall_at_10": 1.0, "ndcg_at_10": 0.9},
                            "long": {"queries": 1, "recall_at_10": 0.6, "ndcg_at_10": 0.5},
                        },
                    },
                    {
                        "implementation": "bitmax_binary",
                        "status": "ok",
                        "latency_ms": 10.0,
                        "latency_p50_ms": 11.0,
                        "latency_p95_ms": 13.0,
                        "latency_p99_ms": 14.0,
                        "query_count": 64,
                        "docs": 64,
                        "doc_memory_compression_vs_fp32": 32.0,
                        "recall_at_1": 0.4,
                        "recall_at_5": 0.6,
                        "recall_at_10": 0.7,
                        "ndcg_at_5": 0.5,
                        "ndcg_at_10": 0.6,
                        "quality_delta_vs_dense_ndcg_at_5": -0.1,
                        "quality_delta_vs_dense_ndcg_at_10": -0.1,
                        "speedup_vs_dense_fp16": 10.0,
                        "token_bucket_quality_at_10": {
                            "short": {"queries": 1, "recall_at_10": 1.0, "ndcg_at_10": 0.8},
                            "long": {"queries": 1, "recall_at_10": 0.4, "ndcg_at_10": 0.4},
                        },
                    },
                ],
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "dataset": {"name": "dataset-b:256", "queries": 256, "docs": 256},
                "results": [
                    {
                        "implementation": "bitmax_binary",
                        "status": "ok",
                        "latency_ms": 20.0,
                        "latency_p50_ms": 22.0,
                        "latency_p95_ms": 26.0,
                        "latency_p99_ms": 28.0,
                        "query_count": 256,
                        "docs": 256,
                        "doc_memory_compression_vs_fp32": 32.0,
                        "recall_at_1": 0.6,
                        "recall_at_5": 0.8,
                        "recall_at_10": 0.9,
                        "ndcg_at_5": 0.7,
                        "ndcg_at_10": 0.8,
                        "quality_delta_vs_dense_ndcg_at_5": -0.02,
                        "quality_delta_vs_dense_ndcg_at_10": -0.03,
                        "speedup_vs_dense_fp16": 5.0,
                        "token_bucket_quality_at_10": {
                            "short": {"queries": 2, "recall_at_10": 0.9, "ndcg_at_10": 0.7},
                            "long": {"queries": 2, "recall_at_10": 0.8, "ndcg_at_10": 0.6},
                        },
                    },
                    {
                        "implementation": "qdrant_multivector",
                        "status": "unavailable",
                        "reason": "missing optional service",
                    },
                ],
            }
        )
    )

    summary = _aggregate_comparison_results((first, second))
    rows = {row["implementation"]: row for row in summary["aggregate_results"]}

    assert summary["datasets"] == ["dataset-a:64", "dataset-b:256"]
    assert rows["bitmax_binary"]["datasets_ok"] == 2
    assert rows["bitmax_binary"]["mean_ndcg_at_10"] == pytest.approx(0.7)
    assert rows["bitmax_binary"]["mean_ndcg_at_5"] == pytest.approx(0.6)
    assert rows["bitmax_binary"]["mean_latency_p95_ms"] == pytest.approx(19.5)
    assert rows["bitmax_binary"]["mean_docs"] == pytest.approx(160.0)
    assert rows["bitmax_binary"]["mean_query_count"] == pytest.approx(160.0)
    assert rows["bitmax_binary"]["mean_latency_ms"] == pytest.approx(15.0)
    assert rows["bitmax_binary"]["mean_doc_memory_compression_vs_fp32"] == pytest.approx(32.0)
    assert rows["bitmax_binary"]["token_bucket_quality_at_10"]["short"]["mean_ndcg_at_10"] == pytest.approx(0.75)
    assert summary["per_dataset_deltas"]["dataset-a:64"]["bitmax_binary"]["ndcg_delta_at_10"] == pytest.approx(-0.1)
    assert summary["scaling_results"]["64"]["bitmax_binary"]["mean_latency_ms"] == pytest.approx(10.0)
    assert summary["scaling_results"]["256"]["bitmax_binary"]["mean_latency_ms"] == pytest.approx(20.0)
    assert rows["qdrant_multivector"]["datasets_unavailable"] == 1
