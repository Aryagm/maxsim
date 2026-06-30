import json

import numpy as np
import pytest

from benchmarks.run_retrieval import run_stage


def _write_tiny_embedding_file(path):
    doc_embeddings = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    query_embeddings = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )
    qrels = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int8,
    )
    np.savez(
        path,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        query_embeddings=query_embeddings,
        qrels=qrels,
        doc_ids=np.array(["doc-0", "doc-1", "doc-2"]),
        query_ids=np.array(["query-0", "query-1"]),
        dataset_name=np.array("tiny"),
    )


@pytest.mark.benchmark_smoke
def test_fixture_retrieval_stage_emits_quality_and_timing_signal(tmp_path):
    output_path = tmp_path / "retrieval-fixture.json"

    result = run_stage("fixture-smoke", output_path=output_path)

    assert result["schema_version"] == 1
    assert result["stage"] == "fixture-smoke"
    assert result["gate_passed"] is True
    assert output_path.exists()
    from_disk = json.loads(output_path.read_text())
    assert from_disk["stage"] == "fixture-smoke"
    implementations = {row["implementation"] for row in result["results"]}
    assert implementations >= {"dense_fp16_baseline", "bitmax_native"}
    bitmax_row = next(row for row in result["results"] if row["implementation"] == "bitmax_native")
    assert bitmax_row["recall_at_1"] == pytest.approx(1.0)
    assert bitmax_row["mrr_at_10"] == pytest.approx(1.0)
    assert bitmax_row["ndcg_at_10"] == pytest.approx(1.0)
    assert bitmax_row["doc_memory_compression_vs_fp16"] == pytest.approx(16.0)
    assert bitmax_row["latency_ms"] >= 0.0


def test_embedding_stage_requires_input_path(tmp_path):
    with pytest.raises(RuntimeError, match="embeddings-smoke requires --input"):
        run_stage("embeddings-smoke", output_path=tmp_path / "retrieval.json")


@pytest.mark.benchmark_smoke
def test_embedding_stage_scores_npz_retrieval_fixture(tmp_path):
    input_path = tmp_path / "tiny-retrieval.npz"
    _write_tiny_embedding_file(input_path)

    result = run_stage("embeddings-smoke", input_path=input_path, output_path=tmp_path / "retrieval.json")

    assert result["dataset"]["name"] == "tiny"
    assert result["dataset"]["queries"] == 2
    assert result["dataset"]["docs"] == 3
    assert result["gate_passed"] is True
    rows = {row["implementation"]: row for row in result["results"]}
    assert rows["dense_fp16_baseline"]["recall_at_1"] == pytest.approx(1.0)
    assert rows["bitmax_native"]["recall_at_1"] == pytest.approx(1.0)
    assert rows["bitmax_native"]["topk_agreement_vs_dense_at_10"] == pytest.approx(1.0)
    assert rows["bitmax_native"]["quality_delta_vs_dense_ndcg_at_10"] == pytest.approx(0.0)


def test_retrieval_stage_rejects_missing_qrels(tmp_path):
    input_path = tmp_path / "missing-qrels.npz"
    np.savez(
        input_path,
        doc_embeddings=np.ones((2, 8), dtype=np.float32),
        doc_offsets=np.array([0, 1, 2], dtype=np.int64),
        query_embeddings=np.ones((1, 1, 8), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="qrels"):
        run_stage("embeddings-smoke", input_path=input_path, output_path=tmp_path / "retrieval.json")

