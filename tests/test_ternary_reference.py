import numpy as np


def test_pack_ternary_uses_threshold_and_two_bit_storage():
    from bitmax.experimental import pack_ternary

    docs = np.array(
        [
            [-2.0, -0.2, 0.0, 0.3, 4.0, -5.0, 0.49, 0.51],
            [1.0, -1.0, 0.1, -0.1, 2.0, -2.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )

    packed = pack_ternary(docs, threshold=0.5)

    assert packed.dim == 8
    assert packed.num_docs == 2
    assert packed.doc_offsets.tolist() == [0, 1, 2]
    assert packed.storage_bytes == 4
    np.testing.assert_array_equal(
        packed.values,
        np.array(
            [
                [-1, 0, 0, 0, 1, -1, 0, 1],
                [1, -1, 0, 0, 1, -1, 0, 1],
            ],
            dtype=np.int8,
        ),
    )


def test_ternary_maxsim_and_topk_match_dense_ternary_reference():
    from bitmax.experimental import pack_ternary, ternary_maxsim, topk_ternary_maxsim

    docs = np.array(
        [
            [2, -2, 0.1, -0.1, 3, -3, 0, 4],
            [-2, 2, 0.2, -0.2, -3, 3, 0, -4],
            [0.1, 0.1, 5, -5, 0.2, -0.2, 6, -6],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.array(
        [
            [1, -1, 2, -2, 3, -3, 4, -4],
            [-1, 1, -2, 2, -3, 3, -4, 4],
        ],
        dtype=np.float32,
    )
    packed = pack_ternary(docs, offsets, threshold=0.5)

    scores = ternary_maxsim(query, packed)
    top_scores, top_indices = topk_ternary_maxsim(query, packed, k=2)

    ternary_docs = packed.values.astype(np.float32)
    expected = np.array(
        [
            np.max(query @ ternary_docs[0:2].T, axis=1).sum(),
            np.max(query @ ternary_docs[2:3].T, axis=1).sum(),
        ],
        dtype=np.float32,
    )
    expected_indices = np.lexsort((np.arange(expected.shape[0]), -expected))[:2].astype(np.int64)

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(top_indices, expected_indices)
    np.testing.assert_allclose(top_scores, expected[expected_indices], rtol=0, atol=1e-6)
