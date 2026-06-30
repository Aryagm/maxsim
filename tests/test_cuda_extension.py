import numpy as np
import pytest
from importlib import import_module

import bitmax


@pytest.mark.cuda
def test_cuda_maxsim_matches_cpu_reference_for_small_fixture():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
        ],
        dtype=np.float32,
    )
    query = np.array([[3, -1, 2, -4, 5, -6, 7, -8]], dtype=np.float32)
    packed = bitmax.pack_signs(docs)

    cpu_scores = bitmax.maxsim(query, packed, device="cpu")
    cuda_scores = bitmax.maxsim(query, packed, device="cuda")

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)


@pytest.mark.cuda
def test_cuda_batched_maxsim_matches_cpu_reference_for_ragged_docs():
    pytest.importorskip("bitmax._bitmax_cuda")
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
    packed = bitmax.pack_signs(docs, offsets)

    cpu_scores = bitmax.maxsim(query, packed, device="cpu")
    cuda_scores = bitmax.maxsim(query, packed, device="cuda")

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)


@pytest.mark.cuda
def test_cuda_resident_packed_docs_match_host_cuda_scores():
    pytest.importorskip("bitmax._bitmax_cuda")
    rng = np.random.default_rng(20260630)
    docs = rng.normal(size=(12, 128)).astype(np.float32)
    offsets = np.array([0, 3, 7, 12], dtype=np.int64)
    query = rng.normal(size=(4, 5, 128)).astype(np.float32)
    packed = bitmax.pack_signs(docs, offsets, scale="global")
    cuda_packed = bitmax.to_device(packed, "cuda")

    host_scores = bitmax.maxsim(query, packed, device="cuda")
    resident_scores = bitmax.maxsim(query, cuda_packed)

    assert cuda_packed.device == "cuda"
    np.testing.assert_allclose(resident_scores, host_scores, rtol=0, atol=1e-5)


@pytest.mark.cuda
def test_cuda_resident_topk_matches_cpu_topk_and_tie_breaking():
    pytest.importorskip("bitmax._bitmax_cuda")
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
    packed = bitmax.pack_signs(docs)
    cuda_packed = bitmax.to_device(packed, "cuda")

    cpu_scores, cpu_indices = bitmax.topk_maxsim(query, packed, k=3)
    cuda_scores, cuda_indices = bitmax.topk_maxsim(query, cuda_packed, k=3)

    np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(cuda_indices, cpu_indices)


@pytest.mark.cuda
def test_cuda_resident_reused_buffers_match_reference_after_resize():
    pytest.importorskip("bitmax._bitmax_cuda")
    rng = np.random.default_rng(20260701)
    docs = rng.normal(size=(30, 128)).astype(np.float32)
    offsets = np.array([0, 4, 11, 19, 30], dtype=np.int64)
    packed = bitmax.pack_signs(docs, offsets)
    cuda_packed = bitmax.to_device(packed, "cuda")
    queries = [
        rng.normal(size=(1, 3, 128)).astype(np.float32),
        rng.normal(size=(3, 7, 128)).astype(np.float32),
        rng.normal(size=(1, 3, 128)).astype(np.float32),
    ]

    for query in queries:
        reference_scores = bitmax.maxsim(query, cuda_packed)
        resident_scores = bitmax.maxsim(query, cuda_packed)
        np.testing.assert_allclose(resident_scores, reference_scores, rtol=0, atol=1e-5)

        reference_top_indices = np.stack(
            [np.lexsort((np.arange(row.shape[0], dtype=np.int64), -row))[:2].astype(np.int64) for row in reference_scores],
            axis=0,
        )
        reference_top_scores = np.take_along_axis(reference_scores, reference_top_indices, axis=1)
        resident_top_scores, resident_top_indices = bitmax.topk_maxsim(query, cuda_packed, k=2)
        np.testing.assert_allclose(resident_top_scores, reference_top_scores, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(resident_top_indices, reference_top_indices)


@pytest.mark.cuda
def test_cuda_resident_dim128_uses_specialized_scoring_variant():
    pytest.importorskip("bitmax._bitmax_cuda")
    rng = np.random.default_rng(20260702)
    docs = rng.normal(size=(18, 128)).astype(np.float32)
    offsets = np.array([0, 5, 11, 18], dtype=np.int64)
    query = rng.normal(size=(2, 4, 128)).astype(np.float32)
    packed = bitmax.pack_signs(docs, offsets)
    cuda_packed = bitmax.to_device(packed, "cuda")

    assert cuda_packed.data.maxsim_kernel_variant == "dim128_unrolled"

    host_scores = bitmax.maxsim(query, packed, device="cuda")
    resident_scores = bitmax.maxsim(query, cuda_packed)
    np.testing.assert_allclose(resident_scores, host_scores, rtol=0, atol=1e-5)

    reference_top_indices = np.stack(
        [np.lexsort((np.arange(row.shape[0], dtype=np.int64), -row))[:2].astype(np.int64) for row in resident_scores],
        axis=0,
    )
    reference_top_scores = np.take_along_axis(resident_scores, reference_top_indices, axis=1)
    top_scores, top_indices = bitmax.topk_maxsim(query, cuda_packed, k=2)
    np.testing.assert_allclose(top_scores, reference_top_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(top_indices, reference_top_indices)

    large_docs = rng.normal(size=(129, 128)).astype(np.float32)
    large_packed = bitmax.to_device(bitmax.pack_signs(large_docs), "cuda")
    assert large_packed.data.maxsim_kernel_variant == "generic"


def test_cuda_device_request_fails_clearly_without_cuda_build():
    try:
        import_module("bitmax._bitmax_cuda")
    except ImportError:
        docs = np.ones((1, 8), dtype=np.float32)
        query = np.ones((1, 8), dtype=np.float32)
        packed = bitmax.pack_signs(docs)

        with pytest.raises(NotImplementedError, match="CUDA maxsim is not available"):
            bitmax.maxsim(query, packed, device="cuda")
