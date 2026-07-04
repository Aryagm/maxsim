import numpy as np
import pytest
from importlib import import_module

import maxsim
from maxsim.experimental import (
    fit_dim_centroid_calibration,
    pack_dim_centroid_signs,
    pack_int4_symmetric,
    topk_dim_centroid_maxsim,
    topk_int4_maxsim,
)


@pytest.mark.cuda
def test_cuda_maxsim_matches_cpu_reference_for_small_fixture():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
        ],
        dtype=np.float32,
    )
    query = np.array([[3, -1, 2, -4, 5, -6, 7, -8]], dtype=np.float32)
    packed = maxsim.pack_signs(docs)

    cpu_scores = maxsim.maxsim(query, packed, device="cpu")
    cuda_scores = maxsim.maxsim(query, packed, device="cuda")

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)


@pytest.mark.cuda
def test_cuda_batched_maxsim_matches_cpu_reference_for_ragged_docs():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
            [1, 2, -3, -4, 5, 6, -7, -8],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.array(
        [
            [[3, -1, 2, -4, 5, -6, 7, -8], [0, 0, 0, 0, 0, 0, 0, 0]],
            [[-2, 4, -6, 8, -1, 3, -5, 7], [2, -4, 6, -8, 1, -3, 5, -7]],
        ],
        dtype=np.float32,
    )
    packed = maxsim.pack_signs(docs, offsets)

    cpu_scores = maxsim.maxsim(query, packed, device="cpu")
    cuda_scores = maxsim.maxsim(query, packed, device="cuda")

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)


@pytest.mark.cuda
def test_cuda_resident_packed_docs_match_host_cuda_scores():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260630)
    docs = rng.normal(size=(12, 128)).astype(np.float32)
    offsets = np.array([0, 3, 7, 12], dtype=np.int64)
    query = rng.normal(size=(4, 5, 128)).astype(np.float32)
    packed = maxsim.pack_signs(docs, offsets, scale="global")
    cuda_packed = maxsim.to_device(packed, "cuda")

    host_scores = maxsim.maxsim(query, packed, device="cuda")
    resident_scores = maxsim.maxsim(query, cuda_packed)

    assert cuda_packed.device == "cuda"
    np.testing.assert_allclose(resident_scores, host_scores, rtol=0, atol=1e-5)


@pytest.mark.cuda
def test_cuda_resident_topk_matches_cpu_topk_and_tie_breaking():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    query = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )
    packed = maxsim.pack_signs(docs)
    cuda_packed = maxsim.to_device(packed, "cuda")

    cpu_scores, cpu_indices = maxsim.topk_maxsim(query, packed, k=3)
    cuda_scores, cuda_indices = maxsim.topk_maxsim(query, cuda_packed, k=3)

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(cuda_indices, cpu_indices)


@pytest.mark.cuda
def test_cuda_resident_reused_buffers_match_reference_after_resize():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260701)
    docs = rng.normal(size=(30, 128)).astype(np.float32)
    offsets = np.array([0, 4, 11, 19, 30], dtype=np.int64)
    packed = maxsim.pack_signs(docs, offsets)
    cuda_packed = maxsim.to_device(packed, "cuda")
    queries = [
        rng.normal(size=(1, 3, 128)).astype(np.float32),
        rng.normal(size=(3, 7, 128)).astype(np.float32),
        rng.normal(size=(1, 3, 128)).astype(np.float32),
    ]

    for query in queries:
        reference_scores = maxsim.maxsim(query, cuda_packed)
        resident_scores = maxsim.maxsim(query, cuda_packed)
        np.testing.assert_allclose(resident_scores, reference_scores, rtol=0, atol=1e-5)

        reference_top_indices = np.stack(
            [np.lexsort((np.arange(row.shape[0], dtype=np.int64), -row))[:2].astype(np.int64) for row in reference_scores],
            axis=0,
        )
        reference_top_scores = np.take_along_axis(reference_scores, reference_top_indices, axis=1)
        resident_top_scores, resident_top_indices = maxsim.topk_maxsim(query, cuda_packed, k=2)
        np.testing.assert_allclose(resident_top_scores, reference_top_scores, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(resident_top_indices, reference_top_indices)


@pytest.mark.cuda
def test_cuda_resident_dim128_uses_specialized_scoring_variant():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260702)
    docs = rng.normal(size=(18, 128)).astype(np.float32)
    offsets = np.array([0, 5, 11, 18], dtype=np.int64)
    query = rng.normal(size=(2, 4, 128)).astype(np.float32)
    packed = maxsim.pack_signs(docs, offsets)
    cuda_packed = maxsim.to_device(packed, "cuda")

    assert cuda_packed.data.maxsim_kernel_variant == "dim128_unrolled"

    host_scores = maxsim.maxsim(query, packed, device="cuda")
    resident_scores = maxsim.maxsim(query, cuda_packed)
    np.testing.assert_allclose(resident_scores, host_scores, rtol=0, atol=1e-5)

    reference_top_indices = np.stack(
        [np.lexsort((np.arange(row.shape[0], dtype=np.int64), -row))[:2].astype(np.int64) for row in resident_scores],
        axis=0,
    )
    reference_top_scores = np.take_along_axis(resident_scores, reference_top_indices, axis=1)
    top_scores, top_indices = maxsim.topk_maxsim(query, cuda_packed, k=2)
    np.testing.assert_allclose(top_scores, reference_top_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(top_indices, reference_top_indices)

    large_docs = rng.normal(size=(129, 128)).astype(np.float32)
    large_packed = maxsim.to_device(maxsim.pack_signs(large_docs), "cuda")
    assert large_packed.data.maxsim_kernel_variant == "generic"


@pytest.mark.cuda
def test_cuda_resident_doc_scale_matches_cpu_scores_and_topk():
    pytest.importorskip("maxsim._maxsim_cuda")
    signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
    docs = np.stack([signs, signs * 10.0, -signs], axis=0).astype(np.float32)
    query = signs.reshape(1, 1, 8).astype(np.float32)
    packed = maxsim.pack_signs(docs, scale="doc")
    cuda_packed = maxsim.to_device(packed, "cuda")

    assert cuda_packed.data.has_scale_vector is True
    assert cuda_packed.data.scale_vector_size == packed.num_docs

    cpu_scores = maxsim.maxsim(query, packed, device="cpu")
    cuda_scores = maxsim.maxsim(query, cuda_packed)
    cpu_top_scores, cpu_top_indices = maxsim.topk_maxsim(query, packed, k=2)
    cuda_top_scores, cuda_top_indices = maxsim.topk_maxsim(query, cuda_packed, k=2)

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)
    np.testing.assert_allclose(cuda_top_scores, cpu_top_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(cuda_top_indices, cpu_top_indices)
    np.testing.assert_array_equal(cuda_top_indices, np.array([[1, 0]], dtype=np.int64))


@pytest.mark.cuda
def test_cuda_streaming_topk_matches_resident_topk_and_tie_breaking():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260703)
    docs = rng.normal(size=(20, 128)).astype(np.float32)
    docs[0] = 1.0
    docs[1] = 1.0
    offsets = np.array([0, 2, 7, 13, 20], dtype=np.int64)
    query = rng.normal(size=(3, 5, 128)).astype(np.float32)
    query[0, 0] = 1.0
    packed = maxsim.pack_signs(docs, offsets)
    cuda_packed = maxsim.to_device(packed, "cuda")

    reference_scores, reference_indices = maxsim.topk_maxsim(query, cuda_packed, k=3)
    streaming_scores, streaming_indices = cuda_packed.data.streaming_topk_batch(query, 3, 1.0, False)

    np.testing.assert_allclose(streaming_scores, reference_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(streaming_indices, reference_indices)


@pytest.mark.cuda
def test_cuda_dim128_lut_topk_matches_resident_topk_for_int8_queries():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260704)
    docs = rng.normal(size=(1600, 128)).astype(np.float32)
    offsets = np.arange(0, 1601, 10, dtype=np.int64)
    query = rng.integers(-8, 9, size=(4, 7, 128), dtype=np.int8)
    packed = maxsim.pack_signs(docs, offsets)
    cuda_packed = maxsim.to_device(packed, "cuda")
    query_float = query.astype(np.float32)

    reference_scores, reference_indices = cuda_packed.data.topk_batch(query_float, 5, 1.0, False)
    lut_scores, lut_indices = cuda_packed.data.topk_lut_batch(query_float, 5, 1.0, False)
    api_scores, api_indices = maxsim.topk_maxsim(query, cuda_packed, k=5)

    np.testing.assert_allclose(lut_scores, reference_scores, rtol=0, atol=0)
    np.testing.assert_array_equal(lut_indices, reference_indices)
    np.testing.assert_allclose(api_scores, reference_scores, rtol=0, atol=0)
    np.testing.assert_array_equal(api_indices, reference_indices)


@pytest.mark.cuda
def test_cuda_centroid_topk_matches_cpu_centroid_reference():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260705)
    docs = rng.normal(size=(96, 128)).astype(np.float32)
    offsets = np.array([0, 17, 41, 64, 96], dtype=np.int64)
    query = rng.normal(size=(3, 6, 128)).astype(np.float32)
    calibration = fit_dim_centroid_calibration(docs)
    packed, calibration = pack_dim_centroid_signs(docs, offsets, calibration=calibration)
    cuda_packed = maxsim.to_device(packed, "cuda")

    assert hasattr(cuda_packed.data, "topk_centroid_batch")

    cpu_scores, cpu_indices = topk_dim_centroid_maxsim(query, packed, calibration, k=3)
    cuda_scores, cuda_indices = topk_dim_centroid_maxsim(query, cuda_packed, calibration, k=3)

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(cuda_indices, cpu_indices)


@pytest.mark.cuda
def test_cuda_int4_topk_matches_cpu_int4_reference():
    pytest.importorskip("maxsim._maxsim_cuda")
    from maxsim.experimental import int4_to_device

    rng = np.random.default_rng(20260706)
    docs = rng.normal(size=(96, 128)).astype(np.float32)
    offsets = np.array([0, 17, 41, 64, 96], dtype=np.int64)
    query = rng.normal(size=(3, 6, 128)).astype(np.float32)
    packed = pack_int4_symmetric(docs, offsets)
    cuda_packed = int4_to_device(packed)

    cpu_scores, cpu_indices = topk_int4_maxsim(query, packed, k=3)
    cuda_scores, cuda_indices = topk_int4_maxsim(query, cuda_packed, k=3)

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(cuda_indices, cpu_indices)


@pytest.mark.cuda
def test_cuda_sdk_binary_and_q40_search_match_cpu():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260707)
    docs = rng.normal(size=(64, 128)).astype(np.float32)
    offsets = np.array([0, 13, 29, 47, 64], dtype=np.int64)
    query = rng.normal(size=(3, 5, 128)).astype(np.float32)
    doc_ids = [f"doc-{idx}" for idx in range(4)]

    for mode in ("binary", "binary_q40"):
        corpus = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode=mode)
        cpu_results = maxsim.Reranker.from_corpus(corpus, device="cpu").search(query, k=3)
        cuda_results = maxsim.Reranker.from_corpus(corpus, device="cuda").search(query, k=3)

        assert [[result.doc_id for result in row] for row in cuda_results] == [[result.doc_id for result in row] for row in cpu_results]
        for cuda_row, cpu_row in zip(cuda_results, cpu_results):
            np.testing.assert_allclose([r.score for r in cuda_row], [r.score for r in cpu_row], rtol=0, atol=1e-4)


@pytest.mark.cuda
def test_cuda_sdk_int4_search_and_rerank_match_cpu():
    pytest.importorskip("maxsim._maxsim_cuda")
    rng = np.random.default_rng(20260708)
    docs = rng.normal(size=(64, 128)).astype(np.float32)
    offsets = np.array([0, 13, 29, 47, 64], dtype=np.int64)
    query = rng.normal(size=(2, 5, 128)).astype(np.float32)
    doc_ids = [f"doc-{idx}" for idx in range(4)]
    corpus = maxsim.Corpus.from_embeddings(doc_ids, docs, offsets, mode="int4")

    cpu = maxsim.Reranker.from_corpus(corpus, device="cpu")
    cuda = maxsim.Reranker.from_corpus(corpus, device="cuda")
    cpu_search = cpu.search(query, k=3)
    cuda_search = cuda.search(query, k=3)
    cpu_rerank = cpu.rerank(query, ["doc-3", "doc-1", "doc-2"], k=2)
    cuda_rerank = cuda.rerank(query, ["doc-3", "doc-1", "doc-2"], k=2)

    assert [[result.doc_id for result in row] for row in cuda_search] == [[result.doc_id for result in row] for row in cpu_search]
    assert [[result.doc_id for result in row] for row in cuda_rerank] == [[result.doc_id for result in row] for row in cpu_rerank]
    for cuda_row, cpu_row in zip(cuda_search, cpu_search):
        np.testing.assert_allclose([r.score for r in cuda_row], [r.score for r in cpu_row], rtol=0, atol=1e-4)
    for cuda_row, cpu_row in zip(cuda_rerank, cpu_rerank):
        np.testing.assert_allclose([r.score for r in cuda_row], [r.score for r in cpu_row], rtol=0, atol=1e-4)


def test_cuda_device_request_fails_clearly_without_cuda_build():
    try:
        import_module("maxsim._maxsim_cuda")
    except ImportError:
        docs = np.ones((1, 8), dtype=np.float32)
        query = np.ones((1, 8), dtype=np.float32)
        packed = maxsim.pack_signs(docs)

        with pytest.raises(NotImplementedError, match="CUDA maxsim is not available"):
            maxsim.maxsim(query, packed, device="cuda")


@pytest.mark.cuda
def test_cuda_qtile_kernel_matches_unrolled_kernel():
    pytest.importorskip("maxsim._maxsim_cuda")
    from maxsim import _maxsim_cuda

    rng = np.random.default_rng(139)
    lengths = rng.integers(0, 12, size=300)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    docs = rng.standard_normal((int(offsets[-1]), 128)).astype(np.float32)
    query = rng.standard_normal((3, 19, 128)).astype(np.float32)

    packed = maxsim.pack_signs(docs, offsets, token_scale="mean_abs_fp16")
    cuda_packed = maxsim.to_device(packed, "cuda")
    previous = _maxsim_cuda.get_dim128_qtile_min_packed_bytes()
    try:
        _maxsim_cuda.set_dim128_qtile_min_packed_bytes(1 << 62)
        baseline_scores, baseline_indices = maxsim.topk_maxsim(query, cuda_packed, 10, device="cuda")
        _maxsim_cuda.set_dim128_qtile_min_packed_bytes(1)
        assert cuda_packed.data.maxsim_kernel_variant == "dim128_qtile"
        qtile_scores, qtile_indices = maxsim.topk_maxsim(query, cuda_packed, 10, device="cuda")
    finally:
        _maxsim_cuda.set_dim128_qtile_min_packed_bytes(previous)

    np.testing.assert_array_equal(qtile_indices, baseline_indices)
    np.testing.assert_allclose(qtile_scores, baseline_scores, rtol=1e-5, atol=1e-3)
