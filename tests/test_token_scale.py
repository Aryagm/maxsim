import numpy as np
import pytest

import bitmax
from bitmax.io import load_packed, save_packed


def _reference_token_scale_scores(docs, offsets, query, token_scales):
    signs = np.where(docs >= 0, 1.0, -1.0).astype(np.float32)
    scaled = signs * token_scales[:, np.newaxis].astype(np.float32)
    out = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        doc = scaled[int(start) : int(end)]
        out.append(float(np.max(query @ doc.T, axis=1).sum(dtype=np.float32)) if doc.shape[0] else 0.0)
    return np.asarray(out, dtype=np.float32)


def _ragged_fixture(num_docs, dim, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 9, size=num_docs)
    lengths[min(2, num_docs - 1)] = 0
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    docs = rng.standard_normal((int(offsets[-1]), dim)).astype(np.float32)
    return docs, offsets


def test_pack_signs_token_scale_mean_abs_matches_reference():
    docs, offsets = _ragged_fixture(5, 16, seed=7)
    rng = np.random.default_rng(3)
    query = rng.standard_normal((4, 16)).astype(np.float32)

    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    expected_scales = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(packed.token_scale, expected_scales)

    scores = bitmax.maxsim(query, packed, device="cpu")
    reference = _reference_token_scale_scores(docs, offsets, query, expected_scales)
    np.testing.assert_allclose(scores, reference, rtol=0, atol=1e-5)


def test_pack_signs_token_scale_fp16_quantizes_scales():
    docs, offsets = _ragged_fixture(4, 16, seed=13)
    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs_fp16")
    full = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(packed.token_scale, full.astype(np.float16).astype(np.float32))


def test_pack_signs_token_scale_rejects_bad_shape():
    docs, offsets = _ragged_fixture(4, 16, seed=17)
    with pytest.raises(ValueError):
        bitmax.pack_signs(docs, offsets, token_scale=np.ones(3, dtype=np.float32))


def test_token_scale_round_trips_through_io(tmp_path):
    docs, offsets = _ragged_fixture(4, 16, seed=19)
    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    path = tmp_path / "packed.npz"
    save_packed(path, packed)
    bundle = load_packed(path)
    np.testing.assert_array_equal(bundle.packed.token_scale, packed.token_scale)

    rng = np.random.default_rng(23)
    query = rng.standard_normal((3, 16)).astype(np.float32)
    np.testing.assert_allclose(
        bitmax.maxsim(query, bundle.packed, device="cpu"),
        bitmax.maxsim(query, packed, device="cpu"),
        rtol=0,
        atol=0,
    )


@pytest.mark.cuda
def test_cuda_token_scale_maxsim_matches_cpu_reference_small_corpus():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(6, 128, seed=29)
    rng = np.random.default_rng(31)
    query = rng.standard_normal((2, 5, 128)).astype(np.float32)

    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    cpu_scores = bitmax.maxsim(query, packed, device="cpu")
    cuda_packed = bitmax.to_device(packed, "cuda")
    assert cuda_packed.data.has_token_scale_vector
    cuda_scores = bitmax.maxsim(query, cuda_packed, device="cuda")
    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-3)


@pytest.mark.cuda
def test_cuda_token_scale_maxsim_matches_cpu_reference_large_corpus():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(200, 128, seed=37)
    rng = np.random.default_rng(41)
    query = rng.standard_normal((3, 4, 128)).astype(np.float32)

    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    cpu_scores = bitmax.maxsim(query, packed, device="cpu")
    cuda_scores = bitmax.maxsim(query, bitmax.to_device(packed, "cuda"), device="cuda")
    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-3)


@pytest.mark.cuda
def test_cuda_token_scale_topk_matches_cpu_scores():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(200, 128, seed=43)
    rng = np.random.default_rng(47)
    query = rng.standard_normal((2, 4, 128)).astype(np.float32)

    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    cpu_scores = bitmax.maxsim(query, packed, device="cpu")
    cuda_packed = bitmax.to_device(packed, "cuda")
    top_scores, top_indices = bitmax.topk_maxsim(query, cuda_packed, 10, device="cuda")

    for row_scores, row_indices, row_reference in zip(top_scores, top_indices, cpu_scores):
        order = np.lexsort((np.arange(row_reference.shape[0], dtype=np.int64), -row_reference))[:10]
        np.testing.assert_array_equal(row_indices, order)
        np.testing.assert_allclose(row_scores, row_reference[order], rtol=0, atol=1e-3)


@pytest.mark.cuda
def test_cuda_token_scale_lut_topk_matches_generic_topk_for_int8_queries():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(200, 128, seed=53)
    rng = np.random.default_rng(59)
    query = rng.integers(-4, 5, size=(2, 4, 128)).astype(np.int8)

    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    cuda_packed = bitmax.to_device(packed, "cuda")
    lut_scores, lut_indices = bitmax.topk_maxsim(query, cuda_packed, 10, device="cuda")
    generic_scores, generic_indices = bitmax.topk_maxsim(query.astype(np.float32), cuda_packed, 10, device="cuda")
    np.testing.assert_array_equal(lut_indices, generic_indices)
    np.testing.assert_allclose(lut_scores, generic_scores, rtol=0, atol=1e-3)


@pytest.mark.cuda
def test_cuda_token_scale_empty_doc_scores_zero():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(6, 128, seed=61)
    empty_doc = int(np.flatnonzero(np.diff(offsets) == 0)[0])
    rng = np.random.default_rng(67)
    query = rng.standard_normal((1, 3, 128)).astype(np.float32)

    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs")
    cuda_scores = bitmax.maxsim(query, bitmax.to_device(packed, "cuda"), device="cuda")
    assert cuda_scores[0, empty_doc] == 0.0


def test_pack_signs_token_scale_u8_uses_log_levels():
    docs, offsets = _ragged_fixture(5, 16, seed=101)
    packed = bitmax.pack_signs(docs, offsets, token_scale="mean_abs_u8")
    full = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
    assert packed.token_scale.shape == full.shape
    ratio = packed.token_scale / full
    assert np.all(ratio > 0.99) and np.all(ratio < 1.01)


def test_sdk_binary_token_scale_corpus_round_trip(tmp_path):
    from bitmax.sdk import Corpus, Reranker

    docs, offsets = _ragged_fixture(6, 16, seed=103)
    doc_ids = [f"doc-{i}" for i in range(6)]
    corpus = Corpus.from_embeddings(doc_ids, docs, offsets, mode="binary_token_scale")
    assert corpus.packed.token_scale is not None
    plain = Corpus.from_embeddings(doc_ids, docs, offsets, mode="binary")
    assert corpus.storage_bytes == plain.storage_bytes + docs.shape[0] * 2

    path = tmp_path / "corpus.npz"
    corpus.save(path)
    loaded = Corpus.load(path)
    assert loaded.mode == "binary_token_scale"
    np.testing.assert_array_equal(loaded.packed.token_scale, corpus.packed.token_scale)

    rng = np.random.default_rng(107)
    query = rng.standard_normal((4, 16)).astype(np.float32)
    results = Reranker.from_corpus(loaded).search(query, k=3)
    reference = bitmax.maxsim(query, corpus.packed, device="cpu")
    order = np.lexsort((np.arange(reference.shape[0], dtype=np.int64), -reference))[:3]
    assert [r.doc_id for r in results] == [doc_ids[int(i)] for i in order]
