import numpy as np


def test_pack_int4_symmetric_packs_signed_nibbles_and_reports_storage():
    from maxsim.experimental import pack_int4_symmetric

    docs = np.array(
        [
            [-2.0, -1.1, -0.2, 0.0, 0.4, 1.0, 1.6, 2.2],
            [2.1, 1.4, 0.9, 0.1, -0.6, -1.2, -1.8, -2.4],
        ],
        dtype=np.float32,
    )

    packed = pack_int4_symmetric(docs)

    assert packed.dim == 8
    assert packed.num_docs == 2
    assert packed.scale > 0.0
    assert packed.storage_bytes == 8
    assert packed.data.dtype == np.uint8
    assert packed.data.shape == (2, 4)
    assert np.all(packed.values >= -7)
    assert np.all(packed.values <= 7)


def test_int4_maxsim_matches_dequantized_reference():
    from maxsim.experimental import int4_maxsim, pack_int4_symmetric

    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
            [2, 3, -4, -5, 6, 7, -8, -9],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.array(
        [
            [1, 1, 0, 0, 1, 1, 0, 0],
            [-1, 1, -1, 1, 0, 0, 1, -1],
        ],
        dtype=np.float32,
    )
    packed = pack_int4_symmetric(docs, offsets)

    scores = int4_maxsim(query, packed)
    dequantized = packed.values.astype(np.float32) * packed.scale
    expected = np.array(
        [
            np.max(query @ dequantized[0:2].T, axis=1).sum(),
            np.max(query @ dequantized[2:3].T, axis=1).sum(),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-5)


def test_topk_int4_maxsim_sorts_by_score_then_lower_doc_id():
    from maxsim.experimental import pack_int4_symmetric, topk_int4_maxsim

    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )
    query = np.ones((1, 8), dtype=np.float32)
    packed = pack_int4_symmetric(docs)

    scores, indices = topk_int4_maxsim(query, packed, k=2)

    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(scores[0], scores[1], rtol=0, atol=1e-6)
