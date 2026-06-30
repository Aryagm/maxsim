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
