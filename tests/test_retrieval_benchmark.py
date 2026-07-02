import json

import numpy as np
import pytest

from benchmarks.run_retrieval import run_stage
from benchmarks.run_retrieval import RetrievalEmbeddings, _bitmax_scores, _prepare_bitmax_packed


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


@pytest.mark.benchmark_smoke
def test_embedding_stage_doc_scale_can_restore_dense_ranking(tmp_path):
    input_path = tmp_path / "doc-scale-retrieval.npz"
    signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
    np.savez(
        input_path,
        doc_embeddings=np.stack([signs, signs * 10.0, -signs], axis=0).astype(np.float32),
        doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        query_embeddings=signs.reshape(1, 1, 8).astype(np.float32),
        qrels=np.array([[0, 1, 0]], dtype=np.float32),
        dataset_name=np.array("doc-scale"),
    )

    unscaled = run_stage("embeddings-smoke", input_path=input_path, output_path=tmp_path / "unscaled.json")
    scaled = run_stage("embeddings-smoke", input_path=input_path, output_path=tmp_path / "scaled.json", scale="doc")
    unscaled_row = next(row for row in unscaled["results"] if row["implementation"] == "bitmax_native")
    scaled_row = next(row for row in scaled["results"] if row["implementation"] == "bitmax_native")

    assert unscaled_row["recall_at_1"] == pytest.approx(0.0)
    assert scaled_row["recall_at_1"] == pytest.approx(1.0)
    assert scaled_row["scale"] == "doc"
    assert scaled_row["topk_agreement_vs_dense_at_10"] == pytest.approx(1.0)


@pytest.mark.benchmark_smoke
def test_embedding_stage_can_emit_experimental_pareto_variants(tmp_path):
    input_path = tmp_path / "pareto-retrieval.npz"
    _write_tiny_embedding_file(input_path)

    result = run_stage(
        "embeddings-smoke",
        input_path=input_path,
        output_path=tmp_path / "pareto.json",
        variants="all",
    )

    rows = {row["implementation"]: row for row in result["results"]}
    expected = {
        "dense_fp16_baseline",
        "bitmax_binary",
        "bitmax_binary_doc_scale",
        "ternary_threshold",
        "binary_token_scale",
        "binary_token_scale_cuda",
        "binary_token_scale_fp16_cuda",
        "binary_group_scale_16",
        "int4_symmetric_per_tensor",
        "binary_calibrated_threshold",
        "binary_dim_centroid_zero",
        "binary_dim_centroid_q40",
        "binary_dim_centroid_lloyd",
    }
    assert set(rows) == expected
    for name in expected - {"dense_fp16_baseline"}:
        assert rows[name]["recall_at_1"] >= 0.0
        assert rows[name]["recall_at_10"] >= 0.0
        assert rows[name]["mrr_at_10"] >= 0.0
        assert rows[name]["ndcg_at_10"] >= 0.0
        assert rows[name]["doc_storage_bytes"] > 0
        assert rows[name]["speedup_vs_dense_fp16"] > 0.0


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


def test_cuda_retrieval_scores_pad_ragged_queries_for_single_batch_call(monkeypatch):
    docs = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=np.float32,
    )
    dataset = RetrievalEmbeddings(
        name="ragged",
        query_embeddings=(
            np.ones((1, 8), dtype=np.float32),
            np.full((3, 8), 2.0, dtype=np.float32),
        ),
        doc_embeddings=docs,
        doc_offsets=np.array([0, 1, 2], dtype=np.int64),
        qrels=np.eye(2, dtype=np.float32),
        query_ids=("q0", "q1"),
        doc_ids=("d0", "d1"),
    )
    packed = __import__("bitmax").pack_signs(dataset.doc_embeddings, dataset.doc_offsets)
    calls = []

    def fake_maxsim(query, packed_arg, *, device):
        calls.append((query.copy(), packed_arg, device))
        assert query.shape == (2, 3, 8)
        np.testing.assert_array_equal(query[0, 1:], np.zeros((2, 8), dtype=np.float32))
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr("benchmarks.run_retrieval.bitmax.maxsim", fake_maxsim)

    scores = _bitmax_scores(dataset, packed, device="cuda")

    assert len(calls) == 1
    assert calls[0][2] == "cuda"
    np.testing.assert_array_equal(scores, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_cuda_retrieval_prepares_resident_packed_docs_once(monkeypatch):
    packed = __import__("bitmax").pack_signs(np.ones((2, 8), dtype=np.float32))
    calls = []

    def fake_to_device(packed_arg, device):
        calls.append((packed_arg, device))
        return "cuda-packed"

    monkeypatch.setattr("benchmarks.run_retrieval.bitmax.to_device", fake_to_device)

    assert _prepare_bitmax_packed(packed, "cuda") == "cuda-packed"
    assert calls == [(packed, "cuda")]
    assert _prepare_bitmax_packed(packed, "auto") is packed


@pytest.mark.benchmark_smoke
def test_embedding_stage_centroid_binary_can_restore_dimension_magnitude_ranking(tmp_path):
    input_path = tmp_path / "centroid-binary-retrieval.npz"
    doc_embeddings = np.array(
        [
            [-1, 1, -1, -1, -1, -1, -1, -1],
            [10, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )
    query = np.array([[[1, 1, 0, 0, 0, 0, 0, 0]]], dtype=np.float32)
    np.savez(
        input_path,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 2], dtype=np.int64),
        query_embeddings=query,
        qrels=np.array([[0, 1]], dtype=np.float32),
        dataset_name=np.array("centroid-binary"),
    )

    result = run_stage(
        "embeddings-smoke",
        input_path=input_path,
        output_path=tmp_path / "centroid.json",
        variants="binary,binary_dim_centroid_zero,binary_dim_centroid_q40,binary_dim_centroid_lloyd",
    )

    rows = {row["implementation"]: row for row in result["results"]}
    assert rows["dense_fp16_baseline"]["recall_at_1"] == pytest.approx(1.0)
    assert rows["bitmax_binary"]["recall_at_1"] == pytest.approx(0.0)
    assert rows["binary_dim_centroid_zero"]["recall_at_1"] == pytest.approx(1.0)
    assert rows["binary_dim_centroid_q40"]["recall_at_1"] == pytest.approx(1.0)
    assert rows["binary_dim_centroid_lloyd"]["recall_at_1"] == pytest.approx(1.0)
    assert rows["binary_dim_centroid_zero"]["doc_storage_bytes"] == rows["bitmax_binary"]["doc_storage_bytes"] + 3 * 8 * 4
    assert rows["binary_dim_centroid_q40"]["doc_storage_bytes"] == rows["binary_dim_centroid_zero"]["doc_storage_bytes"]
