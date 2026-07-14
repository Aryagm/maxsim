import numpy as np
import pytest


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
    assert packed.storage_bytes == 8 + 4
    assert packed.data.dtype == np.uint8
    assert packed.data.shape == (2, 4)
    assert np.all(packed.values >= -7)
    assert np.all(packed.values <= 7)


def test_pack_int4_symmetric_per_token_uses_row_scales_and_handles_zero_rows():
    from maxsim.experimental import pack_int4_symmetric

    docs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-14.0, -7.0, 0.0, 2.0, 4.0, 6.0, 10.0, 14.0],
            [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7],
        ],
        dtype=np.float32,
    )

    packed = pack_int4_symmetric(docs, scale_granularity="token")

    expected_scales = np.array([1.0, 2.0, 0.1], dtype=np.float32)
    expected_values = np.clip(np.rint(docs / expected_scales[:, None]), -7, 7).astype(np.int8)
    assert packed.scale == 1.0
    assert packed.token_scale.dtype == np.float32
    assert packed.token_scale.flags.c_contiguous
    np.testing.assert_allclose(packed.token_scale, expected_scales, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(packed.values, expected_values)
    assert packed.storage_bytes == packed.data.nbytes + 4 + packed.token_scale.nbytes


@pytest.mark.parametrize("bad_granularity", ["doc", "channel", ""])
def test_pack_int4_symmetric_rejects_unknown_scale_granularity(bad_granularity):
    from maxsim.experimental import pack_int4_symmetric

    with pytest.raises(ValueError, match="scale_granularity"):
        pack_int4_symmetric(np.ones((1, 8), dtype=np.float32), scale_granularity=bad_granularity)


def test_pack_int4_symmetric_rejects_explicit_scale_in_token_mode():
    from maxsim.experimental import pack_int4_symmetric

    with pytest.raises(ValueError, match="scale cannot be provided"):
        pack_int4_symmetric(
            np.ones((1, 8), dtype=np.float32),
            scale=0.5,
            scale_granularity="token",
        )


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


def test_int4_per_token_scale_is_applied_before_max_reduction():
    from maxsim.experimental import int4_maxsim, pack_int4_symmetric

    docs = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.ones((1, 8), dtype=np.float32)
    packed = pack_int4_symmetric(
        docs,
        np.array([0, 2], dtype=np.int64),
        scale_granularity="token",
    )

    integer_dots = query @ packed.values.astype(np.float32).T
    reconstructed_dots = integer_dots * packed.token_scale[None, :]
    assert int(integer_dots.argmax(axis=1)[0]) == 0
    assert int(reconstructed_dots.argmax(axis=1)[0]) == 1
    np.testing.assert_allclose(int4_maxsim(query, packed), np.array([10.0], dtype=np.float32))


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


def test_topk_int4_maxsim_supports_an_empty_query_batch():
    from maxsim.experimental import pack_int4_symmetric, topk_int4_maxsim

    packed = pack_int4_symmetric(
        np.ones((3, 8), dtype=np.float32),
        np.arange(4, dtype=np.int64),
        scale_granularity="token",
    )

    scores, indices = topk_int4_maxsim(
        np.empty((0, 2, 8), dtype=np.float32), packed, k=2
    )

    assert scores.shape == (0, 2)
    assert indices.shape == (0, 2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"scale": np.nan}, "finite scalar"),
        ({"token_scale": np.array([1.0], dtype=np.float32)}, "num_doc_tokens"),
        ({"token_scale": np.array([1.0, 0.0], dtype=np.float32)}, "finite and > 0"),
        ({"token_scale": np.array([1.0, np.nan], dtype=np.float32)}, "finite and > 0"),
        ({"token_scale": np.ones(2, dtype=np.float64)}, "dtype float32"),
        ({"doc_offsets": np.array([1, 2], dtype=np.int64)}, "span all"),
        ({"doc_offsets": np.array([0, 3], dtype=np.int64)}, "span all"),
        ({"doc_offsets": np.array([0.0, 2.0], dtype=np.float32)}, "contain integers"),
    ],
)
def test_int4_scoring_rejects_malformed_packed_metadata(mutation, message):
    from dataclasses import replace

    from maxsim.experimental import int4_maxsim, pack_int4_symmetric

    docs = np.ones((2, 8), dtype=np.float32)
    packed = pack_int4_symmetric(
        docs,
        np.array([0, 2], dtype=np.int64),
        scale_granularity="token",
    )
    malformed = replace(packed, **mutation)

    with pytest.raises(ValueError, match=message):
        int4_maxsim(np.ones((1, 8), dtype=np.float32), malformed)


def test_int4_scoring_rejects_nonmonotonic_offsets():
    from dataclasses import replace

    from maxsim.experimental import int4_maxsim, pack_int4_symmetric

    docs = np.ones((2, 8), dtype=np.float32)
    packed = pack_int4_symmetric(
        docs,
        np.array([0, 1, 2], dtype=np.int64),
        scale_granularity="token",
    )
    malformed = replace(
        packed,
        num_docs=3,
        doc_offsets=np.array([0, 2, 1, 2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="monotonically nondecreasing"):
        int4_maxsim(np.ones((1, 8), dtype=np.float32), malformed)


def test_int4_to_device_uploads_token_scales_without_retaining_values(monkeypatch):
    import maxsim
    from maxsim.experimental import int4_to_device, pack_int4_symmetric

    calls = []

    class FakeHandle:
        def __init__(self, data, offsets, dim, scale):
            self.packed_size = data.nbytes
            calls.append((data.copy(), offsets.copy(), dim, scale))

        def set_token_scale_vector(self, token_scale):
            calls.append(token_scale.copy())

    class FakeCudaModule:
        CudaInt4PackedDocs = FakeHandle

    monkeypatch.setattr(maxsim, "_maxsim_cuda", FakeCudaModule(), raising=False)
    packed = pack_int4_symmetric(
        np.arange(16, dtype=np.float32).reshape(2, 8),
        scale_granularity="token",
    )

    cuda_packed = int4_to_device(packed)

    assert cuda_packed.device == "cuda"
    assert cuda_packed.values is None
    np.testing.assert_array_equal(cuda_packed.token_scale, packed.token_scale)
    np.testing.assert_array_equal(calls[1], packed.token_scale)
    assert cuda_packed.storage_bytes == packed.storage_bytes


def test_int4_to_device_rejects_mismatched_packed_and_unpacked_values():
    from dataclasses import replace

    from maxsim.experimental import int4_to_device, pack_int4_symmetric

    packed = pack_int4_symmetric(
        np.arange(16, dtype=np.float32).reshape(2, 8),
        scale_granularity="token",
    )
    corrupt_data = packed.data.copy()
    corrupt_data[0, 0] ^= np.uint8(1)

    with pytest.raises(ValueError, match="does not encode"):
        int4_to_device(replace(packed, data=corrupt_data))


def test_int4_scoring_rejects_asymmetric_negative_eight_code():
    from dataclasses import replace

    from maxsim.experimental import int4_maxsim, pack_int4_symmetric

    packed = pack_int4_symmetric(np.ones((1, 8), dtype=np.float32))
    values = packed.values.copy()
    values[0, 0] = -8

    with pytest.raises(ValueError, match="symmetric int4"):
        int4_maxsim(
            np.ones((1, 8), dtype=np.float32),
            replace(packed, values=values),
        )
