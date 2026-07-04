import numpy as np
import pytest

import maxsim


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

    corpus = maxsim.Corpus.from_embeddings(
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
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.search(query, k=2)

    assert [result.doc_id for result in results] == ["positive", "mixed"]
    assert [result.rank for result in results] == [1, 2]
    assert all(isinstance(result.score, float) for result in results)


def test_reranker_search_returns_batch_results_for_batched_queries():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

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
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.rerank(query, candidate_ids=["negative", "mixed", "mixed"], k=3)

    assert [result.doc_id for result in results] == ["mixed", "negative"]
    assert [result.rank for result in results] == [1, 2]


def test_reranker_rejects_unknown_candidate_id():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)

    with pytest.raises(KeyError, match="missing"):
        reranker.rerank(np.ones((1, 8), dtype=np.float32), candidate_ids=["missing"], k=1)


def test_corpus_rejects_invalid_doc_ids_offsets_and_mode():
    docs, offsets = _tiny_multivector_docs()

    with pytest.raises(ValueError, match="unique"):
        maxsim.Corpus.from_embeddings(["dup", "dup", "mixed"], docs, offsets, mode="binary")
    with pytest.raises(ValueError, match="one more entry"):
        maxsim.Corpus.from_embeddings(["one", "two"], docs, offsets, mode="binary")
    with pytest.raises(ValueError, match="mode"):
        maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="bad")


def test_binary_q40_mode_searches_through_sdk():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary_q40")
    reranker = maxsim.Reranker.from_corpus(corpus)

    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=2)

    assert corpus.mode == "binary_q40"
    assert corpus.storage_bytes > 3
    assert len(results) == 2
    assert results[0].doc_id == "positive"


def test_int4_mode_searches_through_sdk():
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="int4")
    reranker = maxsim.Reranker.from_corpus(corpus)

    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=2)

    assert corpus.mode == "int4"
    assert corpus.storage_bytes == 3 * 8 // 2 + 4
    assert results[0].doc_id == "positive"


def test_corpus_save_load_roundtrips_binary_q40_scores_and_metadata(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(
        ["positive", "negative", "mixed"],
        docs,
        offsets,
        mode="binary_q40",
        metadata={"source": "tiny"},
    )
    query = np.ones((1, 8), dtype=np.float32)
    expected = maxsim.Reranker.from_corpus(corpus).search(query, k=3)

    corpus.save(tmp_path / "tiny.maxsim.npz")
    loaded = maxsim.Corpus.load(tmp_path / "tiny.maxsim.npz")
    actual = maxsim.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.doc_ids == corpus.doc_ids
    assert loaded.mode == "binary_q40"
    assert loaded.metadata == {"source": "tiny"}
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)


def test_reranker_load_constructs_runtime_from_saved_corpus(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    corpus.save(tmp_path / "tiny.maxsim.npz")

    reranker = maxsim.Reranker.load(tmp_path / "tiny.maxsim.npz")
    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=1)

    assert results[0].doc_id == "positive"


def test_corpus_save_load_roundtrips_int4_scores(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = maxsim.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="int4")
    query = np.ones((1, 8), dtype=np.float32)
    expected = maxsim.Reranker.from_corpus(corpus).search(query, k=3)

    corpus.save(tmp_path / "tiny-int4.maxsim.npz")
    loaded = maxsim.Corpus.load(tmp_path / "tiny-int4.maxsim.npz")
    actual = maxsim.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.mode == "int4"
    assert loaded.storage_bytes == corpus.storage_bytes
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)


def test_auto_mode_picks_token_scales_on_small_corpora():
    import maxsim

    rng = np.random.default_rng(211)
    docs = rng.standard_normal((12, 16)).astype(np.float32)
    offsets = np.arange(0, 13, 2, dtype=np.int64)
    corpus = maxsim.Corpus.from_embeddings([f"d{i}" for i in range(6)], docs, offsets)
    assert corpus.mode == "binary_token_scale"


def test_auto_mode_picks_int4_on_large_corpora():
    import maxsim
    from maxsim.sdk import AUTO_SMALL_CORPUS_DOCS

    num_docs = AUTO_SMALL_CORPUS_DOCS + 1
    rng = np.random.default_rng(223)
    docs = rng.standard_normal((num_docs, 16)).astype(np.float32)
    offsets = np.arange(num_docs + 1, dtype=np.int64)
    corpus = maxsim.Corpus.from_embeddings([f"d{i}" for i in range(num_docs)], docs, offsets, mode="auto")
    assert corpus.mode == "int4"


def test_index_is_the_product_name_for_corpus():
    import maxsim

    assert maxsim.Index is maxsim.Corpus
