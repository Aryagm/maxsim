import numpy as np
import pytest

import bitmax


def _reference_binary_int8q_scores(docs, offsets, query, token_scales=None):
    """Numpy reference with the exact binary dp4a kernel semantics."""
    signs = np.where(docs >= 0, 1.0, -1.0).astype(np.float32)
    scores = np.empty((query.shape[0], offsets.shape[0] - 1), dtype=np.float32)
    for batch_idx, query_matrix in enumerate(query):
        max_abs = np.max(np.abs(query_matrix), axis=1)
        q_scales = np.where(max_abs == 0.0, 1.0, max_abs / 127.0).astype(np.float32)
        q_int = np.clip(np.rint(query_matrix / q_scales[:, np.newaxis]), -127, 127).astype(np.float32)
        for doc_idx in range(offsets.shape[0] - 1):
            start = int(offsets[doc_idx])
            end = int(offsets[doc_idx + 1])
            if start == end:
                scores[batch_idx, doc_idx] = 0.0
                continue
            dots = q_int @ signs[start:end].T
            if token_scales is not None:
                dots = dots * token_scales[start:end][np.newaxis, :]
            best = dots.max(axis=1).astype(np.float64)
            scores[batch_idx, doc_idx] = np.float32(float((best * q_scales).sum()))
    return scores


def _ragged_fixture(num_docs, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 9, size=num_docs)
    lengths[min(2, num_docs - 1)] = 0
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    docs = rng.standard_normal((int(offsets[-1]), 128)).astype(np.float32)
    return docs, offsets


@pytest.mark.cuda
def test_cuda_binary_int8q_matches_numpy_reference():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(150, seed=201)
    rng = np.random.default_rng(203)
    query = rng.standard_normal((2, 5, 128)).astype(np.float32)

    packed = bitmax.to_device(bitmax.pack_signs(docs, offsets), "cuda")
    kernel_scores = packed.data.maxsim_batch_int8q(np.ascontiguousarray(query))
    reference = _reference_binary_int8q_scores(docs, offsets, query)
    np.testing.assert_allclose(kernel_scores, reference, rtol=1e-5, atol=1e-3)


@pytest.mark.cuda
def test_cuda_binary_int8q_token_scale_matches_numpy_reference():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(150, seed=211)
    rng = np.random.default_rng(213)
    query = rng.standard_normal((2, 4, 128)).astype(np.float32)
    query[0, -1] = 0.0  # padded row

    packed = bitmax.to_device(bitmax.pack_signs(docs, offsets, token_scale="mean_abs_fp16"), "cuda")
    kernel_scores = packed.data.maxsim_batch_int8q(np.ascontiguousarray(query), 1.0, False, True)
    reference = _reference_binary_int8q_scores(docs, offsets, query, token_scales=packed.token_scale)
    empty_doc = int(np.flatnonzero(np.diff(offsets) == 0)[0])
    assert kernel_scores[0, empty_doc] == 0.0
    np.testing.assert_allclose(kernel_scores, reference, rtol=1e-5, atol=1e-3)


@pytest.mark.cuda
def test_cuda_binary_int8q_topk_matches_maxsim_ranking():
    pytest.importorskip("bitmax._bitmax_cuda")
    docs, offsets = _ragged_fixture(150, seed=223)
    rng = np.random.default_rng(227)
    query = rng.standard_normal((2, 4, 128)).astype(np.float32)

    packed = bitmax.to_device(bitmax.pack_signs(docs, offsets), "cuda")
    scores = packed.data.maxsim_batch_int8q(np.ascontiguousarray(query))
    top_scores, top_indices = packed.data.topk_batch_int8q(np.ascontiguousarray(query), 10)

    for row_scores, row_indices, reference_row in zip(top_scores, top_indices, np.asarray(scores)):
        order = np.lexsort((np.arange(reference_row.shape[0], dtype=np.int64), -reference_row))[:10]
        np.testing.assert_array_equal(row_indices, order)
        np.testing.assert_allclose(row_scores, reference_row[order], rtol=0, atol=1e-3)
