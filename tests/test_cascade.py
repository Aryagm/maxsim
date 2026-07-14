from dataclasses import replace

import numpy as np
import pytest

from maxsim.cascade import (
    cascade_topk,
    pack_residual_int4,
    prefix_score,
    prefix_topk,
    residual_int4_to_device,
    residual_score,
)


REDUCERS = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")


def _fixture(seed=1701, dim=8, query_tokens=3):
    rng = np.random.default_rng(seed)
    lengths = np.array([1, 3, 0, 5, 2, 4], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    docs = rng.normal(size=(int(offsets[-1]), dim)).astype(np.float32)
    query = rng.normal(size=(2, query_tokens, dim)).astype(np.float32)
    weights = rng.normal(size=(2, query_tokens)).astype(np.float32)
    return docs, offsets, query, weights


def _reconstruct(values, scale, token_scale):
    return (
        values.astype(np.float32)
        * np.float32(scale)
        * token_scale[:, None]
    )


def _fused_docs(packed):
    return _reconstruct(
        packed.prefix_values,
        packed.prefix_scale,
        packed.prefix_token_scale,
    ) + _reconstruct(
        packed.residual_values,
        packed.residual_scale,
        packed.residual_token_scale,
    )


def _reference(
    query,
    docs,
    offsets,
    reducer,
    *,
    weights=None,
    temperature=1.0,
    candidates=None,
):
    batches = query[None, None, :] if query.ndim == 1 else query[None, :, :] if query.ndim == 2 else query
    selected = np.arange(offsets.shape[0] - 1) if candidates is None else np.asarray(candidates)
    result = np.empty((batches.shape[0], selected.shape[0]), dtype=np.float32)
    if weights is not None:
        weight_values = np.asarray(weights, dtype=np.float32)
        if weight_values.ndim == 1:
            weight_values = np.broadcast_to(weight_values[None, :], (batches.shape[0], batches.shape[1]))
    else:
        weight_values = None
    for batch_idx, query_matrix in enumerate(batches):
        for output_idx, doc_idx in enumerate(selected):
            start = int(offsets[int(doc_idx)])
            end = int(offsets[int(doc_idx) + 1])
            if start == end:
                result[batch_idx, output_idx] = 0.0
                continue
            similarities = query_matrix @ docs[start:end].T
            if reducer == "maxsim":
                pooled = similarities.max(axis=1)
            elif reducer == "weighted_maxsim":
                pooled = similarities.max(axis=1) * weight_values[batch_idx]
            elif reducer in {"topk2", "topk4"}:
                requested = 2 if reducer == "topk2" else 4
                count = min(requested, end - start)
                pooled = np.sort(similarities, axis=1)[:, -count:].mean(axis=1)
            else:
                maximum = similarities.max(axis=1)
                pooled = maximum + temperature * np.log(
                    np.exp((similarities - maximum[:, None]) / temperature).sum(axis=1)
                )
            result[batch_idx, output_idx] = pooled.sum(dtype=np.float32)
    return result[0] if query.ndim in {1, 2} else result


def test_pack_residual_int4_builds_two_token_scaled_streams_with_shared_offsets():
    docs, offsets, _query, _weights = _fixture()
    docs[0] = 0.0

    packed = pack_residual_int4(docs, offsets)

    assert packed.device == "cpu"
    assert packed.prefix_data.shape == (docs.shape[0], docs.shape[1] // 2)
    assert packed.residual_data.shape == packed.prefix_data.shape
    assert packed.prefix_data.dtype == np.uint8
    assert packed.residual_data.dtype == np.uint8
    np.testing.assert_array_equal(packed.doc_offsets, offsets)
    assert packed.prefix_scale == 1.0
    assert packed.residual_scale == 1.0
    assert packed.prefix_token_scale[0] == 1.0
    assert packed.residual_token_scale[0] == 1.0

    expected_bytes = (
        packed.prefix_data.nbytes
        + packed.residual_data.nbytes
        + packed.prefix_token_scale.nbytes
        + packed.residual_token_scale.nbytes
        + 8
    )
    assert packed.storage_bytes == expected_bytes
    prefix = _reconstruct(
        packed.prefix_values,
        packed.prefix_scale,
        packed.prefix_token_scale,
    )
    fused = _fused_docs(packed)
    assert np.mean((docs - fused) ** 2) <= np.mean((docs - prefix) ** 2)


@pytest.mark.parametrize("reducer", REDUCERS)
def test_residual_cpu_reducers_and_candidates_match_fused_reference(reducer):
    docs, offsets, query, weights = _fixture()
    packed = pack_residual_int4(docs, offsets)
    candidates = np.array([5, 1, 5, 2], dtype=np.int64)
    kwargs = {
        "reducer": reducer,
        "candidate_indices": candidates,
        "temperature": 0.45,
    }
    if reducer == "weighted_maxsim":
        kwargs["query_weights"] = weights

    actual = residual_score(query, packed, **kwargs)
    expected = _reference(
        query,
        _fused_docs(packed),
        offsets,
        reducer,
        weights=weights,
        temperature=0.45,
        candidates=candidates,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (2, 4)


def test_prefix_score_matches_prefix_reconstruction_for_vector_query():
    docs, offsets, query, _weights = _fixture()
    packed = pack_residual_int4(docs, offsets)
    prefix = _reconstruct(
        packed.prefix_values,
        packed.prefix_scale,
        packed.prefix_token_scale,
    )

    actual = prefix_score(query[0, 0], packed)
    expected = _reference(query[0, 0], prefix, offsets, "maxsim")

    assert actual.shape == (packed.num_docs,)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_prefix_topk_matches_stable_topk_of_prefix_scores():
    docs, offsets, query, _weights = _fixture(seed=1753)
    packed = pack_residual_int4(docs, offsets)

    scores, indices = prefix_topk(query, packed, 3)
    full = prefix_score(query, packed)
    expected_indices = np.stack(
        [
            np.lexsort((np.arange(packed.num_docs, dtype=np.int64), -row))[:3]
            for row in full
        ]
    )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        scores,
        np.take_along_axis(full, expected_indices, axis=1),
        rtol=0,
        atol=0,
    )

    vector_scores, vector_indices = prefix_topk(query[0, 0], packed, 2)
    vector_full = prefix_score(query[0, 0], packed)
    expected_vector = np.lexsort(
        (np.arange(packed.num_docs, dtype=np.int64), -vector_full)
    )[:2]
    np.testing.assert_array_equal(vector_indices, expected_vector)
    np.testing.assert_allclose(
        vector_scores,
        vector_full[expected_vector],
        rtol=0,
        atol=0,
    )


def test_prefix_topk_large_budget_uses_stable_partial_selection(monkeypatch):
    docs = np.zeros((40, 8), dtype=np.float32)
    offsets = np.arange(41, dtype=np.int64)
    packed = pack_residual_int4(docs, offsets)

    def fail_native_selector(*_args, **_kwargs):
        raise AssertionError("large prefix budget used the native O(num_docs * k) selector")

    monkeypatch.setattr(
        "maxsim.cascade.topk_int4_maxsim", fail_native_selector
    )
    scores, indices = prefix_topk(
        np.ones((2, 3, 8), dtype=np.float32), packed, 33
    )
    empty_scores, empty_indices = prefix_topk(
        np.empty((0, 3, 8), dtype=np.float32), packed, 33
    )

    np.testing.assert_array_equal(scores, np.zeros((2, 33), dtype=np.float32))
    np.testing.assert_array_equal(
        indices,
        np.broadcast_to(np.arange(33, dtype=np.int64), (2, 33)),
    )
    assert empty_scores.shape == (0, 33)
    assert empty_indices.shape == (0, 33)


@pytest.mark.parametrize("k", [0, 41, True, 1.5])
def test_prefix_topk_validates_k(k):
    docs = np.zeros((40, 8), dtype=np.float32)
    packed = pack_residual_int4(docs, np.arange(41, dtype=np.int64))

    with pytest.raises(ValueError):
        prefix_topk(np.ones((2, 8), dtype=np.float32), packed, k)


def test_fused_residual_reduction_is_not_sum_of_separately_reduced_streams():
    docs = np.array(
        [
            [0, 20, 13, 12, 8, 5, -7, 20],
            [-1, -12, 14, -14, 15, 5, -16, -19],
            [-2, -19, -15, 1, 19, -1, 13, 17],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 3], dtype=np.int64)
    query = np.array([[2, 1, 0, 0, -2, 0, -1, -2]], dtype=np.float32)
    packed = pack_residual_int4(docs, offsets)
    prefix = _reconstruct(
        packed.prefix_values,
        packed.prefix_scale,
        packed.prefix_token_scale,
    )
    residual = _reconstruct(
        packed.residual_values,
        packed.residual_scale,
        packed.residual_token_scale,
    )

    fused = residual_score(query, packed)
    separately_reduced = _reference(query, prefix, offsets, "maxsim") + _reference(
        query,
        residual,
        offsets,
        "maxsim",
    )
    expected = _reference(query, prefix + residual, offsets, "maxsim")

    np.testing.assert_allclose(fused, expected, rtol=1e-5, atol=1e-5)
    assert not np.allclose(fused, separately_reduced, rtol=0, atol=1e-6)


def test_residual_int4_to_device_uploads_both_streams_and_dispatches_cuda(monkeypatch):
    import maxsim

    calls = []

    class FakeHandle:
        def __init__(self, data, offsets, dim, scale):
            self.packed_size = data.nbytes
            self.dim = dim
            self.num_docs = offsets.shape[0] - 1
            self.num_tokens = data.shape[0]
            self.has_token_scale_vector = False
            self.has_residual_int4 = False
            calls.append(("init", data.copy(), offsets.copy(), dim, scale))

        def set_token_scale_vector(self, token_scales):
            self.has_token_scale_vector = True
            calls.append(("prefix_scale", token_scales.copy()))

        def set_residual_int4(self, data, scale, token_scales):
            self.has_residual_int4 = True
            calls.append(("residual", data.copy(), scale, token_scales.copy()))

        def maxsim_batch(self, query):
            calls.append(("prefix_score", query.copy()))
            return np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32)

        def score_residual_batch(self, query, reducer, weights, temperature):
            calls.append(("score", reducer, weights, temperature))
            return np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)

        def score_residual_candidates_batch(
            self,
            query,
            candidates,
            reducer,
            weights,
            temperature,
        ):
            calls.append(("candidates", candidates.copy(), reducer, weights, temperature))
            return np.array([[9.0, 8.0]], dtype=np.float32)

    class FakeCudaModule:
        CudaInt4PackedDocs = FakeHandle

    monkeypatch.setattr(maxsim, "_maxsim_cuda", FakeCudaModule(), raising=False)
    docs, offsets, query, _weights = _fixture()
    cpu_packed = pack_residual_int4(docs, offsets)

    packed = residual_int4_to_device(cpu_packed)
    full = residual_score(query[0], packed, reducer="topk4")
    selected = residual_score(
        query[0],
        packed,
        reducer="smoothsim",
        temperature=0.7,
        candidate_indices=np.array([4, 1], dtype=np.int64),
    )
    coarse = prefix_score(query[0], packed)

    assert packed.device == "cuda"
    assert packed.prefix_values is None
    assert packed.residual_values is None
    assert packed.storage_bytes == cpu_packed.storage_bytes
    assert calls[0][0] == "init"
    assert calls[1][0] == "prefix_scale"
    assert calls[2][0] == "residual"
    np.testing.assert_array_equal(calls[2][1], cpu_packed.residual_data)
    np.testing.assert_array_equal(full, np.arange(1, 7, dtype=np.float32))
    np.testing.assert_array_equal(selected, np.array([9.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(coarse, np.arange(6, 0, -1, dtype=np.float32))
    assert calls[3][0:2] == ("score", "topk4")
    assert calls[4][0] == "candidates"
    np.testing.assert_array_equal(calls[4][1], np.array([4, 1], dtype=np.int64))


def test_cascade_topk_matches_manual_coarse_then_fused_rescore_for_batches():
    docs, offsets, query, _weights = _fixture(seed=1811)
    packed = pack_residual_int4(docs, offsets)
    k = 2
    candidate_count = 4

    actual_scores, actual_indices = cascade_topk(
        query,
        packed,
        k,
        candidates=candidate_count,
    )

    coarse = prefix_score(query, packed)
    expected_scores = []
    expected_indices = []
    doc_ids = np.arange(packed.num_docs, dtype=np.int64)
    coarse_sets = []
    for batch_idx in range(query.shape[0]):
        candidate_ids = np.lexsort((doc_ids, -coarse[batch_idx]))[:candidate_count]
        coarse_sets.append(candidate_ids)
        fine = residual_score(
            query[batch_idx],
            packed,
            candidate_indices=candidate_ids,
        )
        order = np.lexsort((candidate_ids, -fine))[:k]
        expected_scores.append(fine[order])
        expected_indices.append(candidate_ids[order])

    np.testing.assert_allclose(actual_scores, np.stack(expected_scores), rtol=0, atol=0)
    np.testing.assert_array_equal(actual_indices, np.stack(expected_indices))
    assert not np.array_equal(coarse_sets[0], coarse_sets[1])


def test_cascade_topk_supports_vector_queries_and_stable_lower_index_ties():
    docs = np.zeros((5, 8), dtype=np.float32)
    offsets = np.arange(6, dtype=np.int64)
    packed = pack_residual_int4(docs, offsets)

    scores, indices = cascade_topk(
        np.ones(8, dtype=np.float32),
        packed,
        2,
        candidates=3,
    )

    np.testing.assert_array_equal(scores, np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))


@pytest.mark.parametrize("reducer", REDUCERS)
def test_cascade_full_budget_matches_exact_full_residual_topk(reducer):
    docs, offsets, query, weights = _fixture(seed=1817)
    packed = pack_residual_int4(docs, offsets)
    kwargs = {"reducer": reducer, "temperature": 0.55}
    if reducer == "weighted_maxsim":
        kwargs["query_weights"] = weights

    full = residual_score(query, packed, **kwargs)
    scores, indices = cascade_topk(
        query,
        packed,
        3,
        candidates=packed.num_docs,
        **kwargs,
    )
    doc_ids = np.arange(packed.num_docs, dtype=np.int64)
    expected_indices = np.stack(
        [np.lexsort((doc_ids, -row))[:3] for row in full]
    )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(
        scores,
        np.take_along_axis(full, expected_indices, axis=1),
        rtol=0,
        atol=0,
    )


def test_cascade_full_budget_vector_query_preserves_stable_ties():
    docs = np.zeros((5, 8), dtype=np.float32)
    packed = pack_residual_int4(docs, np.arange(6, dtype=np.int64))

    scores, indices = cascade_topk(
        np.ones(8, dtype=np.float32),
        packed,
        3,
        candidates=packed.num_docs,
    )

    np.testing.assert_array_equal(scores, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(indices, np.arange(3, dtype=np.int64))


@pytest.mark.parametrize(
    ("k", "candidates"),
    [(0, 1), (2, 1), (1, 7), (True, 2), (1, False)],
)
def test_cascade_topk_validates_candidate_bounds(k, candidates):
    docs, offsets, query, _weights = _fixture()
    packed = pack_residual_int4(docs, offsets)

    with pytest.raises(ValueError):
        cascade_topk(query[0], packed, k, candidates=candidates)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"residual_scale": np.nan}, "finite scalar"),
        ({"residual_token_scale": np.ones(1, dtype=np.float32)}, "num_doc_tokens"),
        ({"residual_token_scale": np.zeros(15, dtype=np.float32)}, "finite and > 0"),
        ({"residual_token_scale": np.ones(15, dtype=np.float64)}, "dtype float32"),
    ],
)
def test_residual_score_rejects_malformed_residual_metadata(mutation, message):
    docs, offsets, query, _weights = _fixture()
    packed = pack_residual_int4(docs, offsets)
    malformed = replace(packed, **mutation)

    with pytest.raises(ValueError, match=message):
        residual_score(query[0], malformed)


def test_residual_upload_rejects_mismatched_packed_and_unpacked_values():
    docs, offsets, _query, _weights = _fixture()
    packed = pack_residual_int4(docs, offsets)
    corrupt_data = packed.residual_data.copy()
    corrupt_data[0, 0] ^= np.uint8(1)

    with pytest.raises(ValueError, match="does not encode"):
        residual_int4_to_device(replace(packed, residual_data=corrupt_data))


@pytest.mark.parametrize(
    ("handle_mutation", "message"),
    [
        ({"dim": 16}, "dim does not match"),
        ({"has_token_scale_vector": False}, "missing prefix token scales"),
        ({"has_residual_int4": False}, "missing residual int4 data"),
    ],
)
def test_residual_score_validates_cuda_handle_metadata(handle_mutation, message):
    docs, offsets, query, _weights = _fixture()
    cpu = pack_residual_int4(docs, offsets)

    class FakeHandle:
        dim = cpu.dim
        num_docs = cpu.num_docs
        num_tokens = cpu.num_tokens
        has_token_scale_vector = True
        has_residual_int4 = True

    handle = FakeHandle()
    for name, value in handle_mutation.items():
        setattr(handle, name, value)
    packed = replace(
        cpu,
        prefix_data=handle,
        prefix_values=None,
        residual_data=handle,
        residual_values=None,
        device="cuda",
    )

    with pytest.raises(ValueError, match=message):
        residual_score(query[0], packed, device="cuda")


@pytest.mark.cuda
@pytest.mark.parametrize("reducer", REDUCERS)
def test_cuda_residual_int4_matches_cpu_for_full_and_candidate_scoring(reducer):
    pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, query, weights = _fixture(seed=1901)
    cpu_packed = pack_residual_int4(docs, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)
    candidates = np.array([5, 1, 2, 5], dtype=np.int64)
    kwargs = {"reducer": reducer, "temperature": 0.6}
    if reducer == "weighted_maxsim":
        kwargs["query_weights"] = weights

    expected_full = residual_score(query, cpu_packed, device="cpu", **kwargs)
    expected_candidates = residual_score(
        query,
        cpu_packed,
        device="cpu",
        candidate_indices=candidates,
        **kwargs,
    )
    actual_full = residual_score(query, cuda_packed, device="cuda", **kwargs)
    actual_candidates = residual_score(
        query,
        cuda_packed,
        device="cuda",
        candidate_indices=candidates,
        **kwargs,
    )

    assert cuda_packed.prefix_data.has_residual_int4 is True
    np.testing.assert_allclose(actual_full, expected_full, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(
        actual_candidates, expected_candidates, rtol=2e-5, atol=2e-3
    )


@pytest.mark.cuda
@pytest.mark.parametrize("residual_reducer_warps", [0, 4, 8])
def test_cuda_residual_int4_launch_modes_match_cpu(residual_reducer_warps):
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, query, weights = _fixture(
        seed=1903, dim=128, query_tokens=8
    )
    cpu_packed = pack_residual_int4(docs, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)
    candidates = np.array([5, 1, 5, 0, 2], dtype=np.int64)
    prior_mode = cuda_extension.get_residual_reducer_warps()
    prior_force_mode = cuda_extension.get_residual_reducer_force_warps()
    try:
        cuda_extension.set_residual_reducer_force_warps(-1)
        cuda_extension.set_residual_reducer_warps(residual_reducer_warps)
        assert (
            cuda_extension.get_residual_reducer_warps()
            == residual_reducer_warps
        )
        for reducer in REDUCERS:
            kwargs = {"reducer": reducer, "temperature": 0.6}
            if reducer == "weighted_maxsim":
                kwargs["query_weights"] = weights
            expected = residual_score(
                query,
                cpu_packed,
                device="cpu",
                candidate_indices=candidates,
                **kwargs,
            )
            actual = residual_score(
                query,
                cuda_packed,
                device="cuda",
                candidate_indices=candidates,
                **kwargs,
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-3)
    finally:
        cuda_extension.set_residual_reducer_warps(prior_mode)
        cuda_extension.set_residual_reducer_force_warps(prior_force_mode)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("query_tokens", "reference_mode"), [(3, 4), (4, 4), (7, 4), (8, 8)]
)
def test_cuda_residual_int4_default_steps_down_for_short_queries(
    query_tokens, reference_mode
):
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, query, _weights = _fixture(
        seed=1905, dim=128, query_tokens=query_tokens
    )
    cuda_packed = residual_int4_to_device(pack_residual_int4(docs, offsets))
    prior_mode = cuda_extension.get_residual_reducer_warps()
    prior_force_mode = cuda_extension.get_residual_reducer_force_warps()
    try:
        cuda_extension.set_residual_reducer_force_warps(-1)
        cuda_extension.set_residual_reducer_warps(reference_mode)
        expected = residual_score(query, cuda_packed, device="cuda")
        cuda_extension.set_residual_reducer_warps(8)
        assert (
            cuda_extension.get_effective_residual_reducer_warps(query_tokens)
            == reference_mode
        )
        actual = residual_score(query, cuda_packed, device="cuda")
        np.testing.assert_array_equal(actual, expected)
    finally:
        cuda_extension.set_residual_reducer_warps(prior_mode)
        cuda_extension.set_residual_reducer_force_warps(prior_force_mode)


@pytest.mark.cuda
def test_cuda_residual_force_route_overrides_adaptive_policy():
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, query, _weights = _fixture(
        seed=1906, dim=128, query_tokens=3
    )
    cpu_packed = pack_residual_int4(docs, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)
    expected = residual_score(query, cpu_packed, device="cpu")
    prior_mode = cuda_extension.get_residual_reducer_warps()
    prior_force_mode = cuda_extension.get_residual_reducer_force_warps()
    try:
        cuda_extension.set_residual_reducer_warps(8)
        for force_mode in (0, 4, 8):
            cuda_extension.set_residual_reducer_force_warps(force_mode)
            assert cuda_extension.get_residual_reducer_force_warps() == force_mode
            for query_tokens in (3, 4, 7, 8):
                assert (
                    cuda_extension.get_effective_residual_reducer_warps(
                        query_tokens
                    )
                    == force_mode
                )
            actual = residual_score(query, cuda_packed, device="cuda")
            np.testing.assert_allclose(
                actual, expected, rtol=2e-5, atol=2e-3
            )
    finally:
        cuda_extension.set_residual_reducer_warps(prior_mode)
        cuda_extension.set_residual_reducer_force_warps(prior_force_mode)


@pytest.mark.cuda
def test_cuda_residual_force_route_rejects_invalid_values_without_mutation():
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    prior_force_mode = cuda_extension.get_residual_reducer_force_warps()
    try:
        for invalid_mode in (-2, 1, 2, 16):
            with pytest.raises(
                ValueError, match="one of -1, 0, 4, or 8"
            ):
                cuda_extension.set_residual_reducer_force_warps(invalid_mode)
            assert (
                cuda_extension.get_residual_reducer_force_warps()
                == prior_force_mode
            )
        with pytest.raises(ValueError, match="query tokens must be positive"):
            cuda_extension.get_effective_residual_reducer_warps(0)
    finally:
        cuda_extension.set_residual_reducer_force_warps(prior_force_mode)


@pytest.mark.cuda
def test_cuda_residual_launch_mode_rejects_invalid_values_without_mutation():
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    prior_mode = cuda_extension.get_residual_reducer_warps()
    try:
        for invalid_mode in (-1, 1, 2, 16):
            with pytest.raises(ValueError, match="one of 0, 4, or 8"):
                cuda_extension.set_residual_reducer_warps(invalid_mode)
            assert cuda_extension.get_residual_reducer_warps() == prior_mode
    finally:
        cuda_extension.set_residual_reducer_warps(prior_mode)


@pytest.mark.cuda
def test_cuda_sdk_residual_cascade_matches_cpu():
    pytest.importorskip("maxsim._maxsim_cuda")
    import maxsim

    docs, offsets, query, _weights = _fixture(seed=1907)
    doc_ids = [f"doc-{idx}" for idx in range(offsets.shape[0] - 1)]
    corpus = maxsim.Index.from_embeddings(
        doc_ids, docs, offsets, mode="int4_residual"
    )
    cpu = maxsim.Reranker.from_corpus(corpus)
    cuda = maxsim.Reranker.from_corpus(corpus, device="cuda")

    expected = cpu.search(query, k=2, rescore_candidates=4)
    actual = cuda.search(query, k=2, rescore_candidates=4)

    assert [[item.doc_id for item in row] for row in actual] == [
        [item.doc_id for item in row] for row in expected
    ]
    np.testing.assert_allclose(
        [[item.score for item in row] for row in actual],
        [[item.score for item in row] for row in expected],
        rtol=2e-5,
        atol=2e-3,
    )


@pytest.mark.cuda
def test_cuda_cascade_full_budget_matches_exact_cpu_topk():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, query, _weights = _fixture(seed=1911, dim=128)
    cpu_packed = pack_residual_int4(docs, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)

    expected_scores, expected_indices = cascade_topk(
        query,
        cpu_packed,
        3,
        candidates=cpu_packed.num_docs,
        device="cpu",
    )
    actual_scores, actual_indices = cascade_topk(
        query,
        cuda_packed,
        3,
        candidates=cuda_packed.num_docs,
        device="cuda",
    )

    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(
        actual_scores, expected_scores, rtol=2e-5, atol=2e-3
    )


@pytest.mark.cuda
def test_cuda_residual_int4_dim128_grows_workspace_and_matches_cpu():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, _query, _weights = _fixture(seed=1913, dim=128)
    cpu_packed = pack_residual_int4(docs, offsets)
    cuda_packed = residual_int4_to_device(cpu_packed)
    candidates = np.array([5, 1, 5, 0, 2], dtype=np.int64)
    rng = np.random.default_rng(1919)

    workspace_sizes = []
    queries = []
    for shape in ((1, 5, 128), (3, 17, 128)):
        query = rng.normal(size=shape).astype(np.float32)
        queries.append(query)
        expected = residual_score(
            query,
            cpu_packed,
            device="cpu",
            candidate_indices=candidates,
        )
        actual = residual_score(
            query,
            cuda_packed,
            device="cuda",
            candidate_indices=candidates,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-3)
        workspace_sizes.append(cuda_packed.prefix_data.workspace_bytes)

    assert workspace_sizes[1] >= workspace_sizes[0]
    expected_scores, expected_indices = cascade_topk(
        queries[-1],
        cpu_packed,
        k=2,
        candidates=5,
        device="cpu",
    )
    actual_scores, actual_indices = cascade_topk(
        queries[-1],
        cuda_packed,
        k=2,
        candidates=5,
        device="cuda",
    )
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=2e-5, atol=2e-3)


@pytest.mark.cuda
def test_cuda_residual_int4_supports_all_empty_documents():
    pytest.importorskip("maxsim._maxsim_cuda")
    docs = np.empty((0, 8), dtype=np.float32)
    offsets = np.array([0, 0, 0], dtype=np.int64)
    cpu = pack_residual_int4(docs, offsets)
    packed = residual_int4_to_device(cpu)

    scores = residual_score(
        np.ones((2, 8), dtype=np.float32),
        packed,
        device="cuda",
    )

    np.testing.assert_array_equal(scores, np.zeros(2, dtype=np.float32))
