import numpy as np
import pytest

import bitmax
from bitmax.experimental import dim_centroid_maxsim, fit_dim_centroid_calibration, pack_dim_centroid_signs


def test_save_load_packed_docs_roundtrips_scores_scale_and_metadata(tmp_path):
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
        ],
        dtype=np.float32,
    )
    query = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
    packed = bitmax.pack_signs(docs, scale="doc")

    bitmax.save_packed(tmp_path / "docs.npz", packed, metadata={"name": "tiny"})
    bundle = bitmax.load_packed(tmp_path / "docs.npz")

    assert bundle.metadata == {"name": "tiny"}
    assert bundle.centroid_calibration is None
    assert bundle.packed.dim == packed.dim
    assert bundle.packed.num_docs == packed.num_docs
    np.testing.assert_array_equal(bundle.packed.data, packed.data)
    np.testing.assert_array_equal(bundle.packed.doc_offsets, packed.doc_offsets)
    np.testing.assert_allclose(bundle.packed.scale, packed.scale)
    np.testing.assert_allclose(bitmax.maxsim(query, bundle.packed), bitmax.maxsim(query, packed))


def test_save_load_centroid_bundle_roundtrips_scores(tmp_path):
    docs = np.array(
        [
            [1, -2, 3, -4, 5, -6, 7, -8],
            [-1, 2, -3, 4, -5, 6, -7, 8],
            [2, 3, -4, -5, 6, 7, -8, -9],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.array([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.float32)
    calibration = fit_dim_centroid_calibration(docs)
    packed, calibration = pack_dim_centroid_signs(docs, offsets, calibration=calibration)

    bitmax.save_packed(tmp_path / "centroid.npz", packed, calibration=calibration, metadata={"format": "centroid"})
    bundle = bitmax.load_packed(tmp_path / "centroid.npz")

    assert bundle.metadata == {"format": "centroid"}
    assert bundle.centroid_calibration is not None
    np.testing.assert_allclose(bundle.centroid_calibration.thresholds, calibration.thresholds)
    np.testing.assert_allclose(bundle.centroid_calibration.negative_centroids, calibration.negative_centroids)
    np.testing.assert_allclose(bundle.centroid_calibration.positive_centroids, calibration.positive_centroids)
    np.testing.assert_allclose(
        dim_centroid_maxsim(query, bundle.packed, bundle.centroid_calibration),
        dim_centroid_maxsim(query, packed, calibration),
    )


def test_save_packed_rejects_cuda_handles(tmp_path):
    packed = bitmax.PackedDocs(
        data=object(),
        doc_offsets=np.array([0, 1], dtype=np.int64),
        dim=8,
        num_docs=1,
        device="cuda",
    )

    with pytest.raises(ValueError, match="CPU PackedDocs"):
        bitmax.save_packed(tmp_path / "cuda.npz", packed)
