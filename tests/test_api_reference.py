import math

import numpy as np
import pytest

import bitmax


def unpack_reference(packed, dim):
    signs = np.empty((packed.shape[0], dim), dtype=np.float32)
    for row_idx, row in enumerate(packed):
        for bit_idx in range(dim):
            byte = int(row[bit_idx // 8])
            bit = (byte >> (bit_idx % 8)) & 1
            signs[row_idx, bit_idx] = 1.0 if bit else -1.0
    return signs


def reference_maxsim(query, doc_embeddings, offsets, scale=None):
    query = np.asarray(query, dtype=np.float32)
    docs = np.asarray(doc_embeddings, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)

    if query.ndim == 2:
        batches = query[None, :, :]
        squeeze = True
    else:
        batches = query
        squeeze = False

    result = np.empty((batches.shape[0], offsets.shape[0] - 1), dtype=np.float32)
    for batch_idx, q in enumerate(batches):
        for doc_idx in range(offsets.shape[0] - 1):
            doc = docs[offsets[doc_idx] : offsets[doc_idx + 1]]
            dots = q @ doc.T
            score = dots.max(axis=1).sum()
            if scale is not None:
                score *= float(scale)
            result[batch_idx, doc_idx] = score
    return result[0] if squeeze else result


def test_pack_signs_uses_little_endian_bits_and_zero_is_positive():
    docs = np.array([[-1, 0, 2, -3, 4, -5, 6, -7]], dtype=np.float32)

    packed = bitmax.pack_signs(docs)

    assert packed.dim == 8
    assert packed.num_docs == 1
    assert packed.doc_offsets.tolist() == [0, 1]
    assert packed.data.dtype == np.uint8
    assert packed.data.tolist() == [[0b01010110]]


def test_maxsim_matches_reference_for_ragged_docs():
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, -2, 3, 4, -5, -6, 7, 8],
            [1, 2, -3, -4, 5, 6, -7, -8],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.array(
        [
            [3, -1, 2, -4, 5, -6, 7, -8],
            [-2, 4, -6, 8, -1, 3, -5, 7],
        ],
        dtype=np.int8,
    )
    signs = np.where(docs >= 0, 1.0, -1.0)

    packed = bitmax.pack_signs(docs, offsets)
    scores = bitmax.maxsim(query, packed)

    np.testing.assert_allclose(scores, reference_maxsim(query, signs, offsets), rtol=0, atol=1e-5)


def test_maxsim_supports_batched_queries_and_global_scale():
    docs = np.array(
        [
            [1.0, -3.0, 5.0, -7.0, 9.0, -11.0, 13.0, -15.0],
            [-2.0, 4.0, -6.0, 8.0, -10.0, 12.0, -14.0, 16.0],
            [3.0, 5.0, -7.0, -9.0, 11.0, 13.0, -15.0, -17.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 3], dtype=np.int64)
    query = np.array(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [[-8, -7, -6, -5, -4, -3, -2, -1]],
        ],
        dtype=np.float16,
    )

    packed = bitmax.pack_signs(docs, offsets, scale="global")
    scores = bitmax.maxsim(query, packed)

    expected_scale = float(np.mean(np.abs(docs)))
    signs = np.where(docs >= 0, 1.0, -1.0)
    assert math.isclose(packed.scale, expected_scale, rel_tol=1e-6)
    np.testing.assert_allclose(
        scores,
        reference_maxsim(query, signs, offsets, scale=expected_scale),
        rtol=1e-3,
        atol=1e-3,
    )


def test_pack_signs_doc_scale_stores_one_scale_per_document_and_scores_with_it():
    docs = np.array(
        [
            [2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0],
            [8.0, -8.0, 8.0, -8.0, 8.0, -8.0, 8.0, -8.0],
            [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 3], dtype=np.int64)
    query = np.array([[1, -1, 1, -1, 1, -1, 1, -1]], dtype=np.float32)

    packed = bitmax.pack_signs(docs, offsets, scale="doc")
    scores = bitmax.maxsim(query, packed)

    expected_scale = np.array([2.0, 4.5], dtype=np.float32)
    signs = np.where(docs >= 0, 1.0, -1.0)
    expected_scores = reference_maxsim(query, signs, offsets) * expected_scale
    assert isinstance(packed.scale, np.ndarray)
    np.testing.assert_allclose(packed.scale, expected_scale, rtol=0, atol=1e-6)
    np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-5)


def test_topk_maxsim_sorts_after_doc_scale_restoration():
    docs = np.array(
        [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0],
            [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    query = np.array([[1, -1, 1, -1, 1, -1, 1, -1]], dtype=np.float32)
    packed = bitmax.pack_signs(docs, scale="doc")

    scores, indices = bitmax.topk_maxsim(query, packed, k=2)

    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_allclose(scores, np.array([80.0, 8.0], dtype=np.float32), rtol=0, atol=1e-5)


def test_cuda_batched_queries_dispatch_once_when_batch_kernel_is_available(monkeypatch):
    import bitmax._api as api

    docs = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=np.float32,
    )
    query = np.array(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [[-1, -2, -3, -4, -5, -6, -7, -8]],
        ],
        dtype=np.float32,
    )
    packed = bitmax.pack_signs(docs)
    calls = []

    class FakeCuda:
        @staticmethod
        def maxsim_cuda_batch(query_arg, packed_arg, offsets_arg, dim_arg, scale_arg):
            calls.append((query_arg.shape, packed_arg.shape, offsets_arg.tolist(), dim_arg, scale_arg))
            assert query_arg.flags.c_contiguous
            return np.array([[36.0, -36.0], [-36.0, 36.0]], dtype=np.float32)

    monkeypatch.setattr(api, "_bitmax_cuda", FakeCuda())

    scores = bitmax.maxsim(query, packed, device="cuda")

    assert len(calls) == 1
    assert calls[0] == ((2, 1, 8), (2, 1), [0, 1, 2], 8, 1.0)
    np.testing.assert_allclose(scores, np.array([[36.0, -36.0], [-36.0, 36.0]], dtype=np.float32))


def test_to_device_uploads_cpu_packed_docs_to_cuda_handle(monkeypatch):
    import bitmax._api as api

    docs = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=np.float32,
    )
    packed = bitmax.pack_signs(docs, scale="global")
    calls = []

    class FakeCuda:
        class CudaPackedDocs:
            def __init__(self, data, offsets, dim):
                calls.append((data.copy(), offsets.copy(), dim))

    monkeypatch.setattr(api, "_bitmax_cuda", FakeCuda())

    cuda_packed = bitmax.to_device(packed, "cuda")

    assert cuda_packed.device == "cuda"
    assert cuda_packed.dim == packed.dim
    assert cuda_packed.num_docs == packed.num_docs
    assert cuda_packed.scale == packed.scale
    np.testing.assert_array_equal(cuda_packed.doc_offsets, packed.doc_offsets)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], packed.data)
    np.testing.assert_array_equal(calls[0][1], packed.doc_offsets)
    assert calls[0][2] == packed.dim


def test_to_device_preserves_doc_scale_vector_for_later_host_side_restoration(monkeypatch):
    import bitmax._api as api

    docs = np.array(
        [
            [2, -2, 2, -2, 2, -2, 2, -2],
            [4, -4, 4, -4, 4, -4, 4, -4],
        ],
        dtype=np.float32,
    )
    packed = bitmax.pack_signs(docs, scale="doc")

    class FakeCuda:
        class CudaPackedDocs:
            def __init__(self, data, offsets, dim):
                pass

    monkeypatch.setattr(api, "_bitmax_cuda", FakeCuda())

    cuda_packed = bitmax.to_device(packed, "cuda")

    assert isinstance(cuda_packed.scale, np.ndarray)
    np.testing.assert_array_equal(cuda_packed.scale, packed.scale)


def test_maxsim_auto_uses_cuda_resident_packed_docs_without_host_packed_copy():
    query = np.array(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [[-1, -2, -3, -4, -5, -6, -7, -8]],
        ],
        dtype=np.float32,
    )
    calls = []

    class FakeCudaPacked:
        def maxsim_batch(self, query_arg, scale_arg):
            calls.append((query_arg.copy(), scale_arg))
            return np.array([[36.0, -36.0], [-36.0, 36.0]], dtype=np.float32)

    cuda_packed = bitmax.PackedDocs(
        data=FakeCudaPacked(),
        doc_offsets=np.array([0, 1, 2], dtype=np.int64),
        dim=8,
        num_docs=2,
        scale=2.0,
        device="cuda",
    )

    scores = bitmax.maxsim(query, cuda_packed)

    assert len(calls) == 1
    assert calls[0][0].shape == (2, 1, 8)
    assert calls[0][1] == 2.0
    np.testing.assert_allclose(scores, np.array([[36.0, -36.0], [-36.0, 36.0]], dtype=np.float32))


def test_topk_maxsim_uses_cuda_resident_fused_topk_when_available():
    query = np.array(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8]],
            [[-1, -2, -3, -4, -5, -6, -7, -8]],
        ],
        dtype=np.float32,
    )
    calls = []

    class FakeCudaPacked:
        def topk_batch(self, query_arg, k_arg, scale_arg):
            calls.append((query_arg.copy(), k_arg, scale_arg))
            return (
                np.array([[9.0, 7.0], [8.0, 6.0]], dtype=np.float32),
                np.array([[3, 1], [2, 0]], dtype=np.int64),
            )

    cuda_packed = bitmax.PackedDocs(
        data=FakeCudaPacked(),
        doc_offsets=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        dim=8,
        num_docs=4,
        scale=1.5,
        device="cuda",
    )

    scores, indices = bitmax.topk_maxsim(query, cuda_packed, k=2)

    assert len(calls) == 1
    assert calls[0][0].shape == (2, 1, 8)
    assert calls[0][1] == 2
    assert calls[0][2] == 1.5
    np.testing.assert_array_equal(scores, np.array([[9.0, 7.0], [8.0, 6.0]], dtype=np.float32))
    np.testing.assert_array_equal(indices, np.array([[3, 1], [2, 0]], dtype=np.int64))


def test_topk_maxsim_uses_cuda_resident_fused_topk_for_large_doc_counts():
    query = np.ones((1, 8), dtype=np.float32)
    calls = []

    class FakeCudaPacked:
        def maxsim_batch(self, query_arg, scale_arg):
            raise AssertionError("CUDA-resident top-k should use fused topk_batch when available")

        def topk_batch(self, query_arg, k_arg, scale_arg):
            calls.append((query_arg.copy(), k_arg, scale_arg))
            return (
                np.array([[511.0, 510.0, 509.0]], dtype=np.float32),
                np.array([[511, 510, 509]], dtype=np.int64),
            )

    cuda_packed = bitmax.PackedDocs(
        data=FakeCudaPacked(),
        doc_offsets=np.arange(513, dtype=np.int64),
        dim=8,
        num_docs=512,
        device="cuda",
    )

    scores, indices = bitmax.topk_maxsim(query, cuda_packed, k=3)

    assert len(calls) == 1
    assert calls[0][1] == 3
    np.testing.assert_array_equal(indices, np.array([511, 510, 509], dtype=np.int64))
    np.testing.assert_array_equal(scores, np.array([511.0, 510.0, 509.0], dtype=np.float32))


def test_topk_maxsim_sorts_by_score_descending_then_lower_doc_id():
    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )
    query = np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
    packed = bitmax.pack_signs(docs)

    scores, indices = bitmax.topk_maxsim(query, packed, k=2)

    assert indices.tolist() == [0, 1]
    assert scores.tolist() == [8.0, 8.0]


@pytest.mark.parametrize("bad_dim", [7, 9])
def test_pack_signs_rejects_dimensions_not_divisible_by_eight(bad_dim):
    docs = np.ones((2, bad_dim), dtype=np.float32)

    with pytest.raises(ValueError, match="dim must be divisible by 8"):
        bitmax.pack_signs(docs)
