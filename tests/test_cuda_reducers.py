import numpy as np
import pytest

import maxsim


REDUCERS = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")


def _require_cuda():
    pytest.importorskip("maxsim._maxsim_cuda")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA reducer tests require a CUDA-capable PyTorch build")
    return torch


def _fixture(dim=128):
    rng = np.random.default_rng(20260713)
    lengths = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    docs = rng.normal(size=(int(offsets[-1]), dim)).astype(np.float32)
    query = rng.normal(size=(2, 5, dim)).astype(np.float32)
    query_weights = np.array(
        [[1.0, 0.25, 1.5, 0.0, 0.75], [0.5, 2.0, 0.125, 1.0, 0.25]],
        dtype=np.float32,
    )
    candidates = np.array([5, 1, 5, 0, 2], dtype=np.int64)
    return docs, offsets, query, query_weights, candidates


def _torch_reference(
    query,
    docs,
    offsets,
    *,
    reducer,
    query_weights=None,
    temperature=1.0,
    candidate_indices=None,
):
    torch = __import__("torch")
    query_tensor = torch.as_tensor(np.asarray(query), dtype=torch.float32)
    if query_tensor.ndim == 2:
        query_tensor = query_tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    docs_tensor = torch.as_tensor(np.asarray(docs), dtype=torch.float32)
    offsets = np.asarray(offsets, dtype=np.int64)

    if query_weights is None:
        weights = None
    else:
        weights = torch.as_tensor(np.asarray(query_weights), dtype=torch.float32)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if weights.shape[0] == 1 and query_tensor.shape[0] != 1:
            weights = weights.expand(query_tensor.shape[0], -1)

    if candidate_indices is None:
        doc_indices = np.arange(offsets.shape[0] - 1, dtype=np.int64)
    else:
        doc_indices = np.asarray(candidate_indices, dtype=np.int64)

    rows = []
    for doc_idx in doc_indices:
        start = int(offsets[int(doc_idx)])
        end = int(offsets[int(doc_idx) + 1])
        if start == end:
            rows.append(torch.zeros(query_tensor.shape[0], dtype=torch.float32))
            continue

        similarities = torch.einsum("bqd,td->bqt", query_tensor, docs_tensor[start:end])
        if reducer == "maxsim":
            per_query = similarities.amax(dim=-1)
        elif reducer == "weighted_maxsim":
            per_query = similarities.amax(dim=-1) * weights
        elif reducer in {"topk2", "topk4"}:
            requested_k = 2 if reducer == "topk2" else 4
            actual_k = min(requested_k, end - start)
            per_query = similarities.topk(actual_k, dim=-1).values.mean(dim=-1)
        elif reducer == "smoothsim":
            per_query = float(temperature) * torch.logsumexp(similarities / float(temperature), dim=-1)
        else:  # pragma: no cover - test helper guard
            raise AssertionError(f"unknown reducer: {reducer}")
        rows.append(per_query.sum(dim=-1))

    result = torch.stack(rows, dim=-1).cpu().numpy().astype(np.float32, copy=False)
    return result[0] if squeeze else result


def _reducer_kwargs(reducer, query_weights, *, temperature=0.7):
    if reducer == "weighted_maxsim":
        return {"query_weights": query_weights}
    if reducer == "smoothsim":
        return {"temperature": temperature}
    return {}


@pytest.mark.cuda
@pytest.mark.parametrize("reducer", REDUCERS)
def test_cuda_binary_reducers_match_torch_for_full_and_candidate_scoring(reducer):
    _require_cuda()
    docs, offsets, query, query_weights, candidates = _fixture()
    packed = maxsim.to_device(maxsim.pack_signs(docs, offsets), "cuda")
    reconstructed = np.where(docs >= 0.0, 1.0, -1.0).astype(np.float32)
    kwargs = _reducer_kwargs(reducer, query_weights)

    full = maxsim.score(query, packed, reducer=reducer, device="cuda", **kwargs)
    candidate = maxsim.score(
        query,
        packed,
        reducer=reducer,
        candidate_indices=candidates,
        device="cuda",
        **kwargs,
    )
    expected_full = _torch_reference(query, reconstructed, offsets, reducer=reducer, **kwargs)
    expected_candidate = _torch_reference(
        query,
        reconstructed,
        offsets,
        reducer=reducer,
        candidate_indices=candidates,
        **kwargs,
    )

    np.testing.assert_allclose(full, expected_full, rtol=1e-5, atol=3e-4)
    np.testing.assert_allclose(candidate, expected_candidate, rtol=1e-5, atol=3e-4)
    np.testing.assert_allclose(candidate, np.take(full, candidates, axis=-1), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("reducer", REDUCERS)
def test_cuda_int4_reducers_match_torch_for_full_and_candidate_scoring(reducer):
    _require_cuda()
    from maxsim.experimental import int4_to_device, pack_int4_symmetric

    docs, offsets, query, query_weights, candidates = _fixture()
    cpu_packed = pack_int4_symmetric(docs, offsets)
    packed = int4_to_device(cpu_packed)
    reconstructed = cpu_packed.values.astype(np.float32) * np.float32(cpu_packed.scale)
    kwargs = _reducer_kwargs(reducer, query_weights)

    full = maxsim.score(query, packed, reducer=reducer, device="cuda", **kwargs)
    candidate = maxsim.score(
        query,
        packed,
        reducer=reducer,
        candidate_indices=candidates,
        device="cuda",
        **kwargs,
    )
    expected_full = _torch_reference(query, reconstructed, offsets, reducer=reducer, **kwargs)
    expected_candidate = _torch_reference(
        query,
        reconstructed,
        offsets,
        reducer=reducer,
        candidate_indices=candidates,
        **kwargs,
    )

    np.testing.assert_allclose(full, expected_full, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(candidate, expected_candidate, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(candidate, np.take(full, candidates, axis=-1), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("reducer", ["maxsim", "smoothsim"])
def test_cuda_binary_reducers_apply_token_and_document_scales_before_candidate_selection(reducer):
    _require_cuda()
    docs, offsets, query, _query_weights, candidates = _fixture()
    token_scales = np.linspace(0.25, 1.75, docs.shape[0], dtype=np.float32)
    cpu_packed = maxsim.pack_signs(docs, offsets, scale="doc", token_scale=token_scales)
    packed = maxsim.to_device(cpu_packed, "cuda")
    reconstructed = np.where(docs >= 0.0, 1.0, -1.0).astype(np.float32) * token_scales[:, None]
    for doc_idx, doc_scale in enumerate(cpu_packed.scale):
        start = int(offsets[doc_idx])
        end = int(offsets[doc_idx + 1])
        reconstructed[start:end] *= np.float32(doc_scale)

    kwargs = {"temperature": 3.0} if reducer == "smoothsim" else {}
    actual = maxsim.score(
        query,
        packed,
        reducer=reducer,
        candidate_indices=candidates,
        device="cuda",
        **kwargs,
    )
    expected = _torch_reference(
        query,
        reconstructed,
        offsets,
        reducer=reducer,
        candidate_indices=candidates,
        **kwargs,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-3)


@pytest.mark.cuda
def test_cuda_binary_smoothsim_per_call_scale_override_restores_resident_scale():
    _require_cuda()
    docs, offsets, query, _query_weights, candidates = _fixture()
    cpu_packed = maxsim.pack_signs(docs, offsets, scale="doc")
    packed = maxsim.to_device(cpu_packed, "cuda")
    signs = np.where(docs >= 0.0, 1.0, -1.0).astype(np.float32)
    override = np.linspace(0.4, 1.4, packed.num_docs, dtype=np.float32)

    overridden = maxsim.score(
        query,
        packed,
        reducer="smoothsim",
        temperature=0.6,
        scale=override,
        candidate_indices=candidates,
        device="cuda",
    )
    restored = maxsim.score(
        query,
        packed,
        reducer="smoothsim",
        temperature=0.6,
        candidate_indices=candidates,
        device="cuda",
    )

    override_docs = signs.copy()
    stored_docs = signs.copy()
    for doc_idx in range(packed.num_docs):
        start = int(offsets[doc_idx])
        end = int(offsets[doc_idx + 1])
        override_docs[start:end] *= override[doc_idx]
        stored_docs[start:end] *= cpu_packed.scale[doc_idx]
    expected_override = _torch_reference(
        query,
        override_docs,
        offsets,
        reducer="smoothsim",
        temperature=0.6,
        candidate_indices=candidates,
    )
    expected_restored = _torch_reference(
        query,
        stored_docs,
        offsets,
        reducer="smoothsim",
        temperature=0.6,
        candidate_indices=candidates,
    )

    np.testing.assert_allclose(overridden, expected_override, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(restored, expected_restored, rtol=2e-5, atol=2e-3)


@pytest.mark.cuda
@pytest.mark.parametrize("packing", ["binary", "int4"])
def test_cuda_smoothsim_is_finite_for_logits_that_overflow_naive_exp(packing):
    _require_cuda()
    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 3], dtype=np.int64)
    query = np.full((2, 8), 1.0e5, dtype=np.float32)
    candidates = np.array([1, 0], dtype=np.int64)
    temperature = 0.125

    if packing == "binary":
        packed = maxsim.to_device(maxsim.pack_signs(docs, offsets), "cuda")
        reconstructed = docs
    else:
        from maxsim.experimental import int4_to_device, pack_int4_symmetric

        cpu_packed = pack_int4_symmetric(docs, offsets)
        packed = int4_to_device(cpu_packed)
        reconstructed = cpu_packed.values.astype(np.float32) * np.float32(cpu_packed.scale)

    actual = maxsim.score(
        query,
        packed,
        reducer="smoothsim",
        temperature=temperature,
        candidate_indices=candidates,
        device="cuda",
    )
    expected = _torch_reference(
        query,
        reconstructed,
        offsets,
        reducer="smoothsim",
        temperature=temperature,
        candidate_indices=candidates,
    )

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1.0)


@pytest.mark.cuda
def test_cuda_reducer_single_query_preserves_candidate_order_and_squeezes_output():
    _require_cuda()
    docs, offsets, query, _query_weights, candidates = _fixture()
    packed = maxsim.to_device(maxsim.pack_signs(docs, offsets), "cuda")

    full = maxsim.score(query[0], packed, reducer="topk4", device="cuda")
    selected = maxsim.score(
        query[0],
        packed,
        reducer="topk4",
        candidate_indices=candidates,
        device="cuda",
    )

    assert full.shape == (packed.num_docs,)
    assert selected.shape == (candidates.shape[0],)
    np.testing.assert_allclose(selected, full[candidates], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("packing", ["binary", "int4"])
@pytest.mark.parametrize("shared_reducer_warps", [0, 4, 8])
def test_cuda_shared_reducer_launch_modes_match_torch_for_all_policies_and_scopes(
    packing,
    shared_reducer_warps,
):
    _require_cuda()
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    docs, offsets, query, query_weights, candidates = _fixture()

    if packing == "binary":
        packed = maxsim.to_device(maxsim.pack_signs(docs, offsets), "cuda")
        reconstructed = np.where(docs >= 0.0, 1.0, -1.0).astype(np.float32)
        rtol, atol = 1e-5, 3e-4
    else:
        from maxsim.experimental import int4_to_device, pack_int4_symmetric

        cpu_packed = pack_int4_symmetric(docs, offsets)
        packed = int4_to_device(cpu_packed)
        reconstructed = cpu_packed.values.astype(np.float32) * np.float32(cpu_packed.scale)
        rtol, atol = 2e-5, 2e-3

    prior_mode = cuda_extension.get_shared_reducer_warps()
    try:
        cuda_extension.set_shared_reducer_warps(shared_reducer_warps)
        assert cuda_extension.get_shared_reducer_warps() == shared_reducer_warps

        for reducer in REDUCERS:
            kwargs = _reducer_kwargs(reducer, query_weights)
            full = maxsim.score(query, packed, reducer=reducer, device="cuda", **kwargs)
            candidate = maxsim.score(
                query,
                packed,
                reducer=reducer,
                candidate_indices=candidates,
                device="cuda",
                **kwargs,
            )
            expected_full = _torch_reference(query, reconstructed, offsets, reducer=reducer, **kwargs)
            expected_candidate = _torch_reference(
                query,
                reconstructed,
                offsets,
                reducer=reducer,
                candidate_indices=candidates,
                **kwargs,
            )
            context = f"packing={packing}, warps={shared_reducer_warps}, reducer={reducer}"
            np.testing.assert_allclose(full, expected_full, rtol=rtol, atol=atol, err_msg=context)
            np.testing.assert_allclose(
                candidate,
                expected_candidate,
                rtol=rtol,
                atol=atol,
                err_msg=context,
            )
            np.testing.assert_allclose(
                candidate,
                np.take(full, candidates, axis=-1),
                rtol=0,
                atol=0,
                err_msg=context,
            )
    finally:
        cuda_extension.set_shared_reducer_warps(prior_mode)


@pytest.mark.cuda
def test_cuda_shared_reducer_launch_mode_rejects_invalid_values_without_mutation():
    cuda_extension = pytest.importorskip("maxsim._maxsim_cuda")
    prior_mode = cuda_extension.get_shared_reducer_warps()
    try:
        for invalid_mode in (-1, 1, 2, 16):
            with pytest.raises(ValueError, match="one of 0, 4, or 8"):
                cuda_extension.set_shared_reducer_warps(invalid_mode)
            assert cuda_extension.get_shared_reducer_warps() == prior_mode
    finally:
        cuda_extension.set_shared_reducer_warps(prior_mode)


@pytest.mark.cuda
@pytest.mark.parametrize("packing", ["binary", "int4"])
def test_cuda_native_reducer_handles_empty_batches_and_zero_query_tokens(packing):
    _require_cuda()
    docs, offsets, _query, _query_weights, _candidates = _fixture()
    if packing == "binary":
        packed = maxsim.to_device(maxsim.pack_signs(docs, offsets), "cuda")
    else:
        from maxsim.experimental import int4_to_device, pack_int4_symmetric

        packed = int4_to_device(pack_int4_symmetric(docs, offsets))

    empty_batch = packed.data.score_batch(
        np.empty((0, 3, packed.dim), dtype=np.float32)
    )
    zero_query_tokens = packed.data.score_batch(
        np.empty((2, 0, packed.dim), dtype=np.float32)
    )

    assert empty_batch.shape == (0, packed.num_docs)
    np.testing.assert_array_equal(
        zero_query_tokens,
        np.zeros((2, packed.num_docs), dtype=np.float32),
    )
