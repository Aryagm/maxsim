import numpy as np
import pytest

import maxsim
from maxsim.experimental import Int4PackedDocs, int4_maxsim, pack_int4_symmetric, topk_int4_maxsim


REDUCERS = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")


def _fixture():
    rng = np.random.default_rng(731)
    docs = rng.normal(size=(7, 8)).astype(np.float32)
    offsets = np.array([0, 1, 3, 3, 7], dtype=np.int64)
    query = rng.normal(size=(2, 3, 8)).astype(np.float32)
    weights = np.array([[0.25, 1.5, -0.5], [1.0, 0.75, 2.0]], dtype=np.float32)
    return docs, offsets, query, weights


def _reference(query, docs, offsets, reducer, *, weights=None, temperature=1.0, candidates=None):
    batches = query[None, :, :] if query.ndim == 2 else query
    selected = np.arange(offsets.shape[0] - 1) if candidates is None else np.asarray(candidates)
    result = np.empty((batches.shape[0], selected.shape[0]), dtype=np.float32)
    for batch_idx, query_matrix in enumerate(batches):
        for output_idx, doc_idx in enumerate(selected):
            doc = docs[offsets[doc_idx] : offsets[doc_idx + 1]]
            if doc.shape[0] == 0:
                result[batch_idx, output_idx] = 0.0
                continue
            similarities = query_matrix @ doc.T
            if reducer in {"maxsim", "weighted_maxsim"}:
                pooled = similarities.max(axis=1)
                if reducer == "weighted_maxsim":
                    pooled = pooled * weights[batch_idx]
            elif reducer in {"topk2", "topk4"}:
                count = min(2 if reducer == "topk2" else 4, doc.shape[0])
                pooled = np.sort(similarities, axis=1)[:, -count:].mean(axis=1)
            else:
                maxima = similarities.max(axis=1)
                pooled = maxima + temperature * np.log(
                    np.exp((similarities - maxima[:, None]) / temperature).sum(axis=1)
                )
            result[batch_idx, output_idx] = pooled.sum(dtype=np.float32)
    return result[0] if query.ndim == 2 else result


@pytest.mark.parametrize("reducer", REDUCERS)
def test_binary_cpu_reducers_and_candidates_match_reference(reducer):
    docs, offsets, query, weights = _fixture()
    packed = maxsim.pack_signs(docs, offsets)
    unpacked = np.where(docs >= 0, 1.0, -1.0).astype(np.float32)
    candidates = np.array([3, 0, 2], dtype=np.int64)
    kwargs = {"reducer": reducer, "candidate_indices": candidates, "temperature": 0.7}
    if reducer == "weighted_maxsim":
        kwargs["query_weights"] = weights

    actual = maxsim.maxsim(query, packed, **kwargs)
    expected = _reference(
        query,
        unpacked,
        offsets,
        reducer,
        weights=weights,
        temperature=0.7,
        candidates=candidates,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (2, 3)


@pytest.mark.parametrize("scale_granularity", ["tensor", "token"])
@pytest.mark.parametrize("reducer", REDUCERS)
def test_int4_cpu_reducers_and_candidates_match_reference(reducer, scale_granularity):
    docs, offsets, query, weights = _fixture()
    packed = pack_int4_symmetric(docs, offsets, scale_granularity=scale_granularity)
    dequantized = packed.values.astype(np.float32) * np.float32(packed.scale)
    if packed.token_scale is not None:
        dequantized = dequantized * packed.token_scale[:, None]
    candidates = np.array([1, 3, 2], dtype=np.int64)
    kwargs = {"reducer": reducer, "candidate_indices": candidates, "temperature": 0.35}
    if reducer == "weighted_maxsim":
        kwargs["query_weights"] = weights

    actual = int4_maxsim(query, packed, **kwargs)
    expected = _reference(
        query,
        dequantized,
        offsets,
        reducer,
        weights=weights,
        temperature=0.35,
        candidates=candidates,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_score_dispatches_both_compressed_loaders_and_forces_generalized_cpu_semantics():
    docs, offsets, query, _ = _fixture()
    binary = maxsim.pack_signs(docs, offsets)
    int4 = pack_int4_symmetric(docs, offsets)

    np.testing.assert_allclose(
        maxsim.score(query, binary, reducer="topk2"),
        maxsim.maxsim(query, binary, reducer="topk2"),
    )
    np.testing.assert_allclose(
        maxsim.score(query, int4, reducer="topk4"),
        int4_maxsim(query, int4, reducer="topk4"),
    )
    with pytest.raises(ValueError, match="cannot be overridden"):
        maxsim.score(query, int4, scale=2.0)


def test_weight_broadcasting_topk_ranking_and_smoothsim_stability():
    docs, offsets, query, _ = _fixture()
    packed = maxsim.pack_signs(docs, offsets)
    shared_weights = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    weighted = maxsim.maxsim(query, packed, reducer="weighted_maxsim", query_weights=shared_weights)
    expected = _reference(
        query,
        np.where(docs >= 0, 1.0, -1.0).astype(np.float32),
        offsets,
        "weighted_maxsim",
        weights=np.broadcast_to(shared_weights, (query.shape[0], query.shape[1])),
    )
    np.testing.assert_allclose(weighted, expected, rtol=1e-5, atol=1e-5)

    top_scores, top_indices = maxsim.topk_maxsim(query, packed, 2, reducer="topk2")
    full_scores = maxsim.maxsim(query, packed, reducer="topk2")
    expected_indices = np.stack(
        [np.lexsort((np.arange(packed.num_docs), -row))[:2] for row in full_scores]
    )
    np.testing.assert_array_equal(top_indices, expected_indices)
    np.testing.assert_allclose(top_scores, np.take_along_axis(full_scores, expected_indices, axis=1))

    large_query = query * np.float32(1e6)
    smooth = maxsim.maxsim(large_query, packed, reducer="smoothsim", temperature=0.01)
    assert np.all(np.isfinite(smooth))


def test_binary_cpu_smoothsim_applies_token_and_document_scales_to_similarities():
    docs, offsets, query, _ = _fixture()
    token_scales = np.linspace(0.25, 1.75, docs.shape[0], dtype=np.float32)
    doc_scales = np.array([0.5, 1.25, 2.0, 0.75], dtype=np.float32)
    packed = maxsim.pack_signs(docs, offsets, scale=doc_scales, token_scale=token_scales)
    reconstructed = np.where(docs >= 0, 1.0, -1.0).astype(np.float32) * token_scales[:, None]
    for doc_idx, doc_scale in enumerate(doc_scales):
        reconstructed[offsets[doc_idx] : offsets[doc_idx + 1]] *= doc_scale

    actual = maxsim.maxsim(query, packed, reducer="smoothsim", temperature=0.6)
    expected = _reference(query, reconstructed, offsets, "smoothsim", temperature=0.6)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("packing", ["binary", "int4"])
def test_generalized_scoring_handles_empty_batches_and_zero_query_tokens(packing):
    docs, offsets, query, _ = _fixture()
    packed = maxsim.pack_signs(docs, offsets) if packing == "binary" else pack_int4_symmetric(docs, offsets)

    empty_batch = maxsim.score(query[:0], packed, reducer="smoothsim")
    zero_tokens = maxsim.score(query[:, :0], packed, reducer="smoothsim")

    assert empty_batch.shape == (0, packed.num_docs)
    np.testing.assert_array_equal(zero_tokens, np.zeros((query.shape[0], packed.num_docs), dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reducer": "unknown"}, "reducer must be one of"),
        ({"reducer": "weighted_maxsim"}, "query_weights is required"),
        ({"query_weights": np.ones(3)}, "only supported"),
        ({"reducer": "smoothsim", "temperature": 0.0}, "greater than 0"),
        ({"candidate_indices": np.array([0.5])}, "integer array"),
        ({"candidate_indices": np.array([-1])}, "between 0"),
    ],
)
def test_reducer_options_are_validated(kwargs, message):
    docs, offsets, query, _ = _fixture()
    packed = maxsim.pack_signs(docs, offsets)
    with pytest.raises(ValueError, match=message):
        maxsim.maxsim(query, packed, **kwargs)


def test_binary_fake_cuda_preserves_legacy_arity_and_uses_generalized_methods():
    calls = []

    class FakeHandle:
        def maxsim_batch(self, query, scale):
            calls.append(("legacy", query.shape, scale))
            return np.array([[3.0, 2.0, 1.0]], dtype=np.float32)

        def score_batch(self, query, reducer, weights, temperature, scale, use_scale_vector, use_token_scale):
            calls.append(("score", reducer, weights, temperature, scale, use_scale_vector, use_token_scale))
            return np.array([[6.0, 5.0, 4.0]], dtype=np.float32)

        def score_candidates_batch(
            self, query, candidates, reducer, weights, temperature, scale, use_scale_vector, use_token_scale
        ):
            calls.append(("candidates", candidates.copy(), reducer, weights, temperature, scale))
            return np.array([[8.0, 7.0]], dtype=np.float32)

    packed = maxsim.PackedDocs(
        data=FakeHandle(),
        doc_offsets=np.arange(4, dtype=np.int64),
        dim=8,
        num_docs=3,
        device="cuda",
    )
    query = np.ones((1, 8), dtype=np.float32)

    np.testing.assert_array_equal(maxsim.maxsim(query, packed), np.array([3.0, 2.0, 1.0]))
    np.testing.assert_array_equal(maxsim.score(query, packed), np.array([6.0, 5.0, 4.0]))
    np.testing.assert_array_equal(
        maxsim.maxsim(query, packed, candidate_indices=[2, 0]),
        np.array([8.0, 7.0]),
    )
    assert calls[0] == ("legacy", (1, 1, 8), 1.0)
    assert calls[1][0:2] == ("score", "maxsim")
    assert calls[2][0] == "candidates"
    np.testing.assert_array_equal(calls[2][1], np.array([2, 0], dtype=np.int64))


def test_binary_fake_cuda_smoothsim_passes_per_call_scale_override_atomically():
    calls = []

    class FakeHandle:
        def maxsim_batch(self, query, scale):  # pragma: no cover - validation compatibility only
            raise AssertionError("legacy path should not be used")

        def score_batch(
            self,
            query,
            reducer,
            weights,
            temperature,
            scale,
            use_scale_vector,
            use_token_scale,
            scale_vector_override,
        ):
            calls.append(
                (
                    "score",
                    reducer,
                    scale,
                    use_scale_vector,
                    np.asarray(scale_vector_override).copy(),
                )
            )
            return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    packed = maxsim.PackedDocs(
        data=FakeHandle(),
        doc_offsets=np.arange(4, dtype=np.int64),
        dim=8,
        num_docs=3,
        device="cuda",
    )
    override = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    actual = maxsim.score(
        np.ones((1, 8), dtype=np.float32),
        packed,
        reducer="smoothsim",
        scale=override,
    )

    np.testing.assert_array_equal(actual, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert calls[0][:4] == ("score", "smoothsim", 1.0, False)
    np.testing.assert_array_equal(calls[0][4], override)


def test_int4_fake_cuda_uses_generalized_full_and_candidate_methods():
    calls = []

    class FakeHandle:
        def maxsim_batch(self, query):
            calls.append(("legacy", query.shape))
            return np.array([[3.0, 2.0, 1.0]], dtype=np.float32)

        def score_batch(self, query, reducer, weights, temperature):
            calls.append(("score", reducer, weights, temperature))
            return np.array([[6.0, 5.0, 4.0]], dtype=np.float32)

        def score_candidates_batch(self, query, candidates, reducer, weights, temperature):
            calls.append(("candidates", candidates.copy(), reducer, weights, temperature))
            return np.array([[8.0, 7.0]], dtype=np.float32)

    packed = Int4PackedDocs(
        data=FakeHandle(),
        values=None,
        doc_offsets=np.arange(4, dtype=np.int64),
        dim=8,
        num_docs=3,
        scale=0.5,
        device="cuda",
    )
    query = np.ones((1, 8), dtype=np.float32)

    np.testing.assert_array_equal(int4_maxsim(query, packed), np.array([3.0, 2.0, 1.0]))
    np.testing.assert_array_equal(maxsim.score(query, packed), np.array([6.0, 5.0, 4.0]))
    np.testing.assert_array_equal(
        int4_maxsim(query, packed, reducer="topk2", candidate_indices=[1, 2]),
        np.array([8.0, 7.0]),
    )
    assert calls[0] == ("legacy", (1, 1, 8))
    assert calls[1][0:2] == ("score", "maxsim")
    assert calls[2][0] == "candidates"
    np.testing.assert_array_equal(calls[2][1], np.array([1, 2], dtype=np.int64))
    assert calls[2][2] == "topk2"


def test_sdk_rerank_scores_only_candidate_indices_and_preserves_candidate_order(monkeypatch):
    import maxsim.sdk as sdk

    docs, offsets, query, weights = _fixture()
    corpus = maxsim.Corpus.from_embeddings(["zero", "one", "empty", "three"], docs, offsets, mode="binary")
    reranker = maxsim.Reranker.from_corpus(corpus)
    calls = []

    def fake_maxsim(query_arg, packed_arg, **kwargs):
        calls.append(kwargs)
        return np.array([[4.0, 9.0], [4.0, 9.0]], dtype=np.float32)

    monkeypatch.setattr(sdk, "maxsim", fake_maxsim)
    results = reranker.rerank(
        query,
        ["three", "one", "three"],
        k=2,
        reducer="weighted_maxsim",
        query_weights=weights,
    )

    assert [[item.doc_id for item in row] for row in results] == [["one", "three"], ["one", "three"]]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["candidate_indices"], np.array([3, 1], dtype=np.int64))
    assert calls[0]["reducer"] == "weighted_maxsim"


def test_int4_topk_accepts_reducer_options_on_cpu():
    docs, offsets, query, weights = _fixture()
    packed = pack_int4_symmetric(docs, offsets)
    top_scores, top_indices = topk_int4_maxsim(
        query,
        packed,
        2,
        reducer="weighted_maxsim",
        query_weights=weights,
    )
    full_scores = int4_maxsim(query, packed, reducer="weighted_maxsim", query_weights=weights)
    expected_indices = np.stack([np.lexsort((np.arange(packed.num_docs), -row))[:2] for row in full_scores])
    np.testing.assert_array_equal(top_indices, expected_indices)
    np.testing.assert_allclose(top_scores, np.take_along_axis(full_scores, expected_indices, axis=1))
