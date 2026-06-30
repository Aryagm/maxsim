import numpy as np
import pytest

import bitmax


def _tiny_multivector_docs():
    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    return docs, offsets


def test_corpus_from_embeddings_exposes_metadata_and_storage():
    docs, offsets = _tiny_multivector_docs()

    corpus = bitmax.Corpus.from_embeddings(
        doc_ids=["positive", "negative", "mixed"],
        embeddings=docs,
        offsets=offsets,
        mode="binary",
        metadata={"model": "tiny"},
    )

    assert corpus.doc_ids == ("positive", "negative", "mixed")
    assert corpus.num_docs == 3
    assert corpus.dim == 8
    assert corpus.mode == "binary"
    assert corpus.metadata == {"model": "tiny"}
    assert corpus.storage_bytes == 3


def test_reranker_search_returns_ranked_results_for_single_query():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.search(query, k=2)

    assert [result.doc_id for result in results] == ["positive", "mixed"]
    assert [result.rank for result in results] == [1, 2]
    assert all(isinstance(result.score, float) for result in results)


def test_reranker_search_returns_batch_results_for_batched_queries():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    query = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )

    results = reranker.search(query, k=1)

    assert [[result.doc_id for result in row] for row in results] == [["positive"], ["negative"]]
    assert [[result.rank for result in row] for row in results] == [[1], [1]]


def test_reranker_rerank_returns_only_candidates_and_collapses_duplicates():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.rerank(query, candidate_ids=["negative", "mixed", "mixed"], k=3)

    assert [result.doc_id for result in results] == ["mixed", "negative"]
    assert [result.rank for result in results] == [1, 2]


def test_reranker_rejects_unknown_candidate_id():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    with pytest.raises(KeyError, match="missing"):
        reranker.rerank(np.ones((1, 8), dtype=np.float32), candidate_ids=["missing"], k=1)
