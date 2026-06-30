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


def test_cuda_device_request_fails_clearly_without_cuda_build():
    try:
        import_module("bitmax._bitmax_cuda")
    except ImportError:
        docs = np.ones((1, 8), dtype=np.float32)
        query = np.ones((1, 8), dtype=np.float32)
        packed = bitmax.pack_signs(docs)

        with pytest.raises(NotImplementedError, match="CUDA maxsim is not available"):
            bitmax.maxsim(query, packed, device="cuda")
