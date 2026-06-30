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
                "dataset": {"name": "dataset-a"},
                "results": [
                    {
                        "implementation": "dense_fp16_baseline",
                        "status": "ok",
                        "latency_ms": 100.0,
                        "doc_memory_compression_vs_fp32": 2.0,
                        "recall_at_10": 0.8,
                        "ndcg_at_10": 0.7,
                        "speedup_vs_dense_fp16": 1.0,
                    },
                    {
                        "implementation": "bitmax_binary",
                        "status": "ok",
                        "latency_ms": 10.0,
                        "doc_memory_compression_vs_fp32": 32.0,
                        "recall_at_10": 0.7,
                        "ndcg_at_10": 0.6,
                        "speedup_vs_dense_fp16": 10.0,
                    },
                ],
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "dataset": {"name": "dataset-b"},
                "results": [
                    {
                        "implementation": "bitmax_binary",
                        "status": "ok",
                        "latency_ms": 20.0,
                        "doc_memory_compression_vs_fp32": 32.0,
                        "recall_at_10": 0.9,
                        "ndcg_at_10": 0.8,
                        "speedup_vs_dense_fp16": 5.0,
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

    assert summary["datasets"] == ["dataset-a", "dataset-b"]
    assert rows["bitmax_binary"]["datasets_ok"] == 2
    assert rows["bitmax_binary"]["mean_ndcg_at_10"] == pytest.approx(0.7)
    assert rows["bitmax_binary"]["mean_latency_ms"] == pytest.approx(15.0)
    assert rows["bitmax_binary"]["mean_doc_memory_compression_vs_fp32"] == pytest.approx(32.0)
    assert rows["qdrant_multivector"]["datasets_unavailable"] == 1
