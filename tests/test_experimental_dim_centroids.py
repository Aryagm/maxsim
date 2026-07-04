import numpy as np

import maxsim


def test_dim_centroid_transform_restores_centroid_approximation_scores():
    from maxsim.experimental import (
        dim_centroid_maxsim,
        fit_dim_centroid_calibration,
        pack_dim_centroid_signs,
    )

    docs = np.array(
        [
            [-1, 1, -1, -1, -1, -1, -1, -1],
            [10, -1, -1, -1, -1, -1, -1, -1],
            [-2, -2, 3, -3, 4, -4, 5, -5],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 1, -1, 0, 0],
        ],
        dtype=np.float32,
    )

    calibration = fit_dim_centroid_calibration(docs)
    packed, returned_calibration = pack_dim_centroid_signs(docs, offsets, calibration=calibration)
    scores = dim_centroid_maxsim(query, packed, returned_calibration)

    approximated_docs = np.where(
        docs >= calibration.thresholds[np.newaxis, :],
        calibration.positive_centroids[np.newaxis, :],
        calibration.negative_centroids[np.newaxis, :],
    ).astype(np.float32)
    expected = np.array(
        [
            np.max(query @ approximated_docs[0:2].T, axis=1).sum(),
            np.max(query @ approximated_docs[2:3].T, axis=1).sum(),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-5)


def test_dim_centroid_topk_keeps_one_bit_doc_storage_plus_small_metadata():
    from maxsim.experimental import (
        fit_dim_centroid_calibration,
        pack_dim_centroid_signs,
        topk_dim_centroid_maxsim,
    )

    docs = np.array(
        [
            [-1, 1, -1, -1, -1, -1, -1, -1],
            [10, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )
    query = np.array([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    calibration = fit_dim_centroid_calibration(docs)
    packed, calibration = pack_dim_centroid_signs(docs, calibration=calibration)
    scores, indices = topk_dim_centroid_maxsim(query, packed, calibration, k=2)

    assert packed.data.shape == (2, 1)
    assert calibration.metadata_bytes == 3 * 8 * 4
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    assert scores[0] > scores[1]


def test_dim_centroid_topk_uses_cuda_fused_centroid_method_when_available():
    from maxsim.experimental import DimCentroidCalibration, topk_dim_centroid_maxsim

    calls = []

    class FakeCudaHandle:
        def topk_centroid_batch(self, query, weights, k, scale, use_scale_vector):
            calls.append((query.copy(), weights.copy(), k, scale, use_scale_vector))
            return (
                np.array([[10.0, 2.0]], dtype=np.float32),
                np.array([[1, 0]], dtype=np.int64),
            )

    calibration = DimCentroidCalibration(
        thresholds=np.zeros(8, dtype=np.float32),
        negative_centroids=np.full(8, -2.0, dtype=np.float32),
        positive_centroids=np.full(8, 3.0, dtype=np.float32),
    )
    packed = maxsim.PackedDocs(
        data=FakeCudaHandle(),
        doc_offsets=np.array([0, 1, 2], dtype=np.int64),
        dim=8,
        num_docs=2,
        device="cuda",
    )
    query = np.ones((1, 8), dtype=np.float32)

    scores, indices = topk_dim_centroid_maxsim(query, packed, calibration, k=2)

    assert len(calls) == 1
    sent_query, sent_weights, sent_k, sent_scale, sent_use_scale = calls[0]
    np.testing.assert_array_equal(sent_query, query[np.newaxis, :, :])
    np.testing.assert_array_equal(sent_weights, np.full(8, 5.0, dtype=np.float32))
    assert sent_k == 2
    assert sent_scale == 1.0
    assert sent_use_scale is False
    np.testing.assert_array_equal(indices, np.array([1, 0], dtype=np.int64))
    np.testing.assert_allclose(scores, np.array([9.0, 5.0], dtype=np.float32))
