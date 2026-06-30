import numpy as np

from benchmarks.build_mixed_embeddings import build_mixed_embeddings


def _write_source(path, *, name, docs, doc_offsets, queries, query_offsets, qrels, doc_ids, query_ids):
    np.savez(
        path,
        doc_embeddings=np.asarray(docs, dtype=np.float32),
        doc_offsets=np.asarray(doc_offsets, dtype=np.int64),
        query_embeddings=np.asarray(queries, dtype=np.float32),
        query_offsets=np.asarray(query_offsets, dtype=np.int64),
        qrels=np.asarray(qrels, dtype=np.float32),
        doc_ids=np.asarray(doc_ids),
        query_ids=np.asarray(query_ids),
        dataset_name=np.asarray(name),
        model_name=np.asarray("test-model"),
    )


def test_build_mixed_embeddings_concatenates_offsets_and_block_diagonal_qrels(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "mixed.npz"
    _write_source(
        first,
        name="first:test:2",
        docs=np.ones((3, 8), dtype=np.float32),
        doc_offsets=[0, 1, 3],
        queries=np.ones((2, 8), dtype=np.float32),
        query_offsets=[0, 1, 2],
        qrels=[[1, 0], [0, 1]],
        doc_ids=["a", "b"],
        query_ids=["qa", "qb"],
    )
    _write_source(
        second,
        name="second:test:1",
        docs=np.full((2, 8), 2, dtype=np.float32),
        doc_offsets=[0, 2],
        queries=np.full((1, 8), 2, dtype=np.float32),
        query_offsets=[0, 1],
        qrels=[[1]],
        doc_ids=["c"],
        query_ids=["qc"],
    )

    build_mixed_embeddings(output_path=output, input_paths=(first, second), dataset_name="mixed:test:3")

    with np.load(output, allow_pickle=False) as data:
        assert data["doc_embeddings"].shape == (5, 8)
        assert data["doc_offsets"].tolist() == [0, 1, 3, 5]
        assert data["query_embeddings"].shape == (3, 8)
        assert data["query_offsets"].tolist() == [0, 1, 2, 3]
        assert data["qrels"].tolist() == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert data["doc_ids"].tolist() == ["first:test:2:a", "first:test:2:b", "second:test:1:c"]
        assert data["query_ids"].tolist() == ["first:test:2:qa", "first:test:2:qb", "second:test:1:qc"]
        assert str(data["dataset_name"].item()) == "mixed:test:3"
        assert data["source_datasets"].tolist() == ["first:test:2", "second:test:1"]
