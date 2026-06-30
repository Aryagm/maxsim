import numpy as np

from benchmarks.build_docscale_stress import build_docscale_stress


def test_build_docscale_stress_repeats_only_nonpositive_docs_and_keeps_qrels_zero_for_repeats(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "stress.npz"
    doc_embeddings = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    query_embeddings = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    np.savez(
        source,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        query_embeddings=query_embeddings,
        query_offsets=np.array([0, 1, 2], dtype=np.int64),
        qrels=np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
        doc_ids=np.array(["d0", "d1", "d2", "d3", "d4"]),
        query_ids=np.array(["q0", "q1"]),
        dataset_name=np.asarray("source:test:5"),
        model_name=np.asarray("test-model"),
    )

    build_docscale_stress(input_path=source, output_path=output, target_docs=8, query_limit=2)

    with np.load(output, allow_pickle=False) as data:
        assert data["doc_offsets"].tolist() == list(range(9))
        assert data["query_offsets"].tolist() == [0, 1, 2]
        assert data["qrels"].shape == (2, 8)
        assert data["qrels"][:, :5].tolist() == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        assert data["qrels"][:, 5:].tolist() == [[0, 0, 0], [0, 0, 0]]
        assert data["doc_ids"].tolist() == ["d0", "d1", "d2", "d3", "d4", "d2:stress-5", "d3:stress-6", "d4:stress-7"]
        assert str(data["dataset_name"].item()) == "source:test:5:stress:8"


def test_build_docscale_stress_can_trim_while_preserving_positive_docs(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "stress.npz"
    np.savez(
        source,
        doc_embeddings=np.ones((4, 8), dtype=np.float32),
        doc_offsets=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        query_embeddings=np.ones((1, 8), dtype=np.float32),
        query_offsets=np.array([0, 1], dtype=np.int64),
        qrels=np.array([[0, 0, 0, 1]], dtype=np.float32),
        doc_ids=np.array(["d0", "d1", "d2", "d3"]),
        query_ids=np.array(["q0"]),
        dataset_name=np.asarray("source:test:4"),
    )

    build_docscale_stress(input_path=source, output_path=output, target_docs=2, query_limit=1)

    with np.load(output, allow_pickle=False) as data:
        assert data["doc_ids"].tolist() == ["d3", "d0"]
        assert data["qrels"].tolist() == [[1.0, 0.0]]
