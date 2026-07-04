import numpy as np
import pytest

from maxsim.experimental import int4_maxsim_int8q, int4_to_device, pack_int4_symmetric


def _reference_int8q_scores(packed, query):
    """Numpy reference with the exact dp4a kernel semantics."""
    values = packed.values.astype(np.float32)
    scores = np.empty((query.shape[0], packed.num_docs), dtype=np.float32)
    for batch_idx, query_matrix in enumerate(query):
        max_abs = np.max(np.abs(query_matrix), axis=1)
        q_scales = np.where(max_abs == 0.0, 1.0, max_abs / 127.0).astype(np.float32)
        q_int = np.clip(np.rint(query_matrix / q_scales[:, np.newaxis]), -127, 127).astype(np.float32)
        for doc_idx in range(packed.num_docs):
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            if start == end:
                scores[batch_idx, doc_idx] = 0.0
                continue
            dots = q_int @ values[start:end].T
            best = dots.max(axis=1).astype(np.float64)
            scores[batch_idx, doc_idx] = np.float32(float((best * q_scales).sum()) * packed.scale)
    return scores


def _ragged_fixture(num_docs, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 9, size=num_docs)
    lengths[min(2, num_docs - 1)] = 0
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    docs = rng.standard_normal((int(offsets[-1]), 128)).astype(np.float32)
    return docs, offsets


@pytest.mark.cuda
def test_cuda_int4_int8q_matches_numpy_reference():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets = _ragged_fixture(150, seed=71)
    rng = np.random.default_rng(73)
    query = rng.standard_normal((2, 5, 128)).astype(np.float32)

    packed = pack_int4_symmetric(docs, offsets)
    cuda_packed = int4_to_device(packed)
    kernel_scores = int4_maxsim_int8q(query, cuda_packed, device="cuda")
    reference = _reference_int8q_scores(packed, query)
    np.testing.assert_allclose(kernel_scores, reference, rtol=1e-5, atol=1e-3)


@pytest.mark.cuda
def test_cuda_int4_int8q_zero_query_rows_and_empty_docs():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets = _ragged_fixture(10, seed=79)
    empty_doc = int(np.flatnonzero(np.diff(offsets) == 0)[0])
    rng = np.random.default_rng(83)
    query = rng.standard_normal((1, 4, 128)).astype(np.float32)
    query[0, -1] = 0.0  # padded row

    packed = pack_int4_symmetric(docs, offsets)
    cuda_packed = int4_to_device(packed)
    kernel_scores = int4_maxsim_int8q(query, cuda_packed, device="cuda")
    reference = _reference_int8q_scores(packed, query)
    assert kernel_scores[0, empty_doc] == 0.0
    np.testing.assert_allclose(kernel_scores, reference, rtol=1e-5, atol=1e-3)


@pytest.mark.cuda
def test_cuda_int4_int8q_topk_matches_maxsim_ranking():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets = _ragged_fixture(150, seed=89)
    rng = np.random.default_rng(97)
    query = rng.standard_normal((2, 4, 128)).astype(np.float32)

    packed = pack_int4_symmetric(docs, offsets)
    cuda_packed = int4_to_device(packed)
    scores = int4_maxsim_int8q(query, cuda_packed, device="cuda")
    top_scores, top_indices = cuda_packed.data.topk_batch_int8q(np.ascontiguousarray(query), 10)

    for row_scores, row_indices, reference_row in zip(top_scores, top_indices, scores):
        order = np.lexsort((np.arange(reference_row.shape[0], dtype=np.int64), -reference_row))[:10]
        np.testing.assert_array_equal(row_indices, order)
        np.testing.assert_allclose(row_scores, reference_row[order], rtol=0, atol=1e-3)
