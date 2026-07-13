from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from maxsim import _maxsim_cpp
except ImportError:  # pragma: no cover - exercised only in pure Python builds
    _maxsim_cpp = None

try:
    from maxsim import _maxsim_cuda
except ImportError:  # pragma: no cover - exercised only in CUDA builds
    _maxsim_cuda = None


Reducer = Literal["maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim"]
_REDUCERS = frozenset(("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim"))


@dataclass(frozen=True)
class PackedDocs:
    data: object
    doc_offsets: np.ndarray
    dim: int
    num_docs: int
    scale: object = None
    device: Literal["cpu", "cuda"] = "cpu"
    token_scale: object = None


def pack_signs(doc_embeddings, doc_offsets=None, *, dim=None, scale=None, token_scale=None, device="cpu") -> PackedDocs:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")

    docs = _as_numpy(doc_embeddings)
    if docs.ndim != 2:
        raise ValueError("doc_embeddings must have shape [num_doc_tokens, dim]")

    actual_dim = int(docs.shape[1])
    if dim is not None and int(dim) != actual_dim:
        raise ValueError(f"dim={dim} does not match doc_embeddings dim={actual_dim}")
    if actual_dim % 8 != 0:
        raise ValueError("dim must be divisible by 8")

    offsets = _normalize_offsets(doc_offsets, docs.shape[0])
    byte_dim = actual_dim // 8
    packed = np.zeros((docs.shape[0], byte_dim), dtype=np.uint8)
    positive = docs >= 0

    for bit_idx in range(actual_dim):
        packed[:, bit_idx // 8] |= positive[:, bit_idx].astype(np.uint8) << (bit_idx % 8)

    stored_scale: object
    if isinstance(scale, str) and scale == "global":
        stored_scale = float(np.mean(np.abs(docs), dtype=np.float64))
    elif isinstance(scale, str) and scale in {"doc", "per_doc"}:
        stored_scale = _doc_mean_abs_scale(docs, offsets)
    elif scale is None:
        stored_scale = None
    else:
        stored_scale = _normalize_explicit_scale(scale, int(offsets.shape[0] - 1))

    stored_token_scale = _resolve_pack_token_scale(token_scale, docs)

    packed_docs = PackedDocs(
        data=packed,
        doc_offsets=offsets,
        dim=actual_dim,
        num_docs=int(offsets.shape[0] - 1),
        scale=stored_scale,
        device="cpu",
        token_scale=stored_token_scale,
    )
    return packed_docs if device == "cpu" else to_device(packed_docs, "cuda")


def _resolve_pack_token_scale(token_scale, docs: np.ndarray):
    if token_scale is None:
        return None
    if isinstance(token_scale, str):
        if token_scale == "mean_abs":
            return np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
        if token_scale == "l2":
            return (np.linalg.norm(docs.astype(np.float64), axis=1) / np.sqrt(docs.shape[1])).astype(np.float32)
        if token_scale == "mean_abs_fp16":
            scales = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
            return scales.astype(np.float16).astype(np.float32)
        if token_scale == "mean_abs_u8":
            scales = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
            return _log_quantized_scales(scales, 256)
        if token_scale == "mean_abs_u4":
            scales = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
            return _log_quantized_scales(scales, 16)
        raise ValueError(
            "token_scale must be None, 'mean_abs', 'mean_abs_fp16', 'mean_abs_u8', 'mean_abs_u4', 'l2', or a [num_doc_tokens] vector"
        )
    values = np.ascontiguousarray(_as_numpy(token_scale), dtype=np.float32)
    if values.ndim != 1 or values.shape[0] != docs.shape[0]:
        raise ValueError("token_scale vector must have shape [num_doc_tokens]")
    return values


def _log_quantized_scales(scales: np.ndarray, levels: int) -> np.ndarray:
    """Quantize positive scales to `levels` log-spaced values (log2(levels) bits/token on disk)."""
    positive = np.maximum(scales.astype(np.float64), 1e-12)
    log_values = np.log(positive)
    lo = float(log_values.min())
    hi = float(log_values.max())
    if hi <= lo:
        return scales.astype(np.float32)
    steps = float(levels - 1)
    codes = np.clip(np.rint((log_values - lo) * (steps / (hi - lo))), 0, steps)
    return np.exp(lo + codes * ((hi - lo) / steps)).astype(np.float32)


def to_device(packed: PackedDocs, device: Literal["cpu", "cuda"] = "cuda") -> PackedDocs:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    _validate_packed(packed)
    if device == packed.device:
        return packed
    if device == "cpu":
        raise NotImplementedError("copying CUDA PackedDocs back to CPU is not implemented")
    if _maxsim_cuda is None or not hasattr(_maxsim_cuda, "CudaPackedDocs"):
        raise NotImplementedError("CUDA PackedDocs are not available in this build")

    packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
    offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
    handle = _maxsim_cuda.CudaPackedDocs(packed_data, offsets, packed.dim)
    stored_scale = packed.scale.copy() if isinstance(packed.scale, np.ndarray) else packed.scale
    if isinstance(stored_scale, np.ndarray) and hasattr(handle, "set_scale_vector"):
        handle.set_scale_vector(np.ascontiguousarray(stored_scale, dtype=np.float32))
    stored_token_scale = packed.token_scale.copy() if isinstance(packed.token_scale, np.ndarray) else packed.token_scale
    if isinstance(stored_token_scale, np.ndarray):
        if not hasattr(handle, "set_token_scale_vector"):
            raise NotImplementedError("token scale vectors require a CUDA build with set_token_scale_vector")
        handle.set_token_scale_vector(np.ascontiguousarray(stored_token_scale, dtype=np.float32))
    return PackedDocs(
        data=handle,
        doc_offsets=offsets.copy(),
        dim=packed.dim,
        num_docs=packed.num_docs,
        scale=stored_scale,
        device="cuda",
        token_scale=stored_token_scale,
    )


def score(
    query_tokens,
    packed,
    *,
    reducer: Reducer = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
    candidate_indices=None,
    scale=None,
    device="auto",
):
    """Score compressed multi-vector documents with a selected reducer.

    Unlike :func:`maxsim`, this generalized entry point always dispatches
    CUDA-resident inputs through the reducer-policy kernel, including when
    ``reducer="maxsim"``. This keeps the legacy MaxSim fast path stable while
    making the shared reducer implementation directly accessible.
    """
    if isinstance(packed, PackedDocs):
        return _score_binary(
            query_tokens,
            packed,
            scale=scale,
            device=device,
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
            candidate_indices=candidate_indices,
            force_generalized=True,
        )

    # Import lazily because maxsim.experimental imports helpers from this
    # module. The public score API can still dispatch both resident loaders.
    from maxsim.experimental import Int4PackedDocs, _int4_score

    if isinstance(packed, Int4PackedDocs):
        if scale is not None:
            raise ValueError("scale is stored by Int4PackedDocs and cannot be overridden")
        return _int4_score(
            query_tokens,
            packed,
            device=device,
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
            candidate_indices=candidate_indices,
            force_generalized=True,
        )
    raise TypeError("packed must be a PackedDocs or Int4PackedDocs instance")


def maxsim(
    query_tokens,
    packed: PackedDocs,
    *,
    scale=None,
    device="auto",
    reducer: Reducer = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
    candidate_indices=None,
):
    return _score_binary(
        query_tokens,
        packed,
        scale=scale,
        device=device,
        reducer=reducer,
        query_weights=query_weights,
        temperature=temperature,
        candidate_indices=candidate_indices,
        force_generalized=False,
    )


def _score_binary(
    query_tokens,
    packed: PackedDocs,
    *,
    scale,
    device,
    reducer,
    query_weights,
    temperature,
    candidate_indices,
    force_generalized: bool,
):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    effective_device = "cuda" if device == "auto" and isinstance(packed, PackedDocs) and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cpu" and _maxsim_cuda is None:
        raise NotImplementedError("CUDA maxsim is not available in this build")
    _validate_packed(packed)
    if packed.device == "cuda" and effective_device in ("auto", "cpu"):
        raise NotImplementedError("CPU scoring for CUDA PackedDocs is not implemented")

    query = _as_numpy(query_tokens)
    if query.ndim == 2:
        batches = query[np.newaxis, :, :]
        squeeze = True
    elif query.ndim == 3:
        batches = query
        squeeze = False
    else:
        raise ValueError("query_tokens must have shape [query_tokens, dim] or [batch, query_tokens, dim]")
    if batches.shape[2] != packed.dim:
        raise ValueError(f"query dim={batches.shape[2]} does not match packed dim={packed.dim}")

    query_float = batches.astype(np.float32, copy=False)
    reducer_value, weight_values, temperature_value = _normalize_reducer_options(
        reducer,
        query_weights,
        temperature,
        batch=query_float.shape[0],
        query_tokens=query_float.shape[1],
    )
    candidates = _normalize_candidate_indices(candidate_indices, packed.num_docs)
    num_targets = packed.num_docs if candidates is None else candidates.shape[0]
    result_shape = (query_float.shape[0], num_targets)
    resolved_scale = _resolve_scale(scale, packed)
    kernel_scale = _kernel_scale(resolved_scale)
    use_token_scale = isinstance(packed.token_scale, np.ndarray)
    use_generalized = force_generalized or reducer_value != "maxsim" or candidates is not None

    if num_targets == 0 or query_float.shape[0] == 0:
        result = np.empty(result_shape, dtype=np.float32)
        return result[0] if squeeze else result
    if query_float.shape[1] == 0:
        result = np.zeros(result_shape, dtype=np.float32)
        return result[0] if squeeze else result

    if effective_device == "cuda" and packed.device == "cuda":
        use_resident_vector_scale = _can_use_cuda_resident_vector_scale(resolved_scale, packed)
        if use_generalized:
            method_name = "score_batch" if candidates is None else "score_candidates_batch"
            method = getattr(packed.data, method_name, None)
            if method is None:
                raise NotImplementedError(f"this CUDA build does not provide {method_name}")
            contiguous_query = np.ascontiguousarray(query_float, dtype=np.float32)
            temporary_scale_vector = (
                reducer_value == "smoothsim"
                and _scale_is_vector(resolved_scale)
                and not use_resident_vector_scale
            )
            kernel_uses_scale_vector = use_resident_vector_scale or temporary_scale_vector
            common_args = (
                reducer_value,
                weight_values,
                float(temperature_value),
                float(kernel_scale),
                kernel_uses_scale_vector,
                use_token_scale,
            )
            if temporary_scale_vector:
                override_args = (*common_args[:-2], False, common_args[-1], resolved_scale)
            else:
                override_args = common_args
            if candidates is None:
                batch_result = method(contiguous_query, *override_args)
            else:
                batch_result = method(contiguous_query, candidates, *override_args)
            if _scale_is_vector(resolved_scale) and not kernel_uses_scale_vector:
                selected_scale = resolved_scale if candidates is None else resolved_scale[candidates]
                batch_result = _apply_vector_scale(batch_result, selected_scale)
            return batch_result[0] if squeeze else batch_result

        batch_result = _cuda_resident_maxsim_batch(
            packed.data,
            np.ascontiguousarray(query_float, dtype=np.float32),
            float(kernel_scale),
            use_resident_vector_scale,
            use_token_scale,
        )
        batch_result = _apply_vector_scale(batch_result, None if use_resident_vector_scale else resolved_scale)
        return batch_result[0] if squeeze else batch_result

    if effective_device == "cuda":
        if use_generalized:
            upload_packed = packed
            if resolved_scale is not packed.scale and _scale_is_vector(resolved_scale):
                upload_packed = PackedDocs(
                    data=packed.data,
                    doc_offsets=packed.doc_offsets,
                    dim=packed.dim,
                    num_docs=packed.num_docs,
                    scale=resolved_scale,
                    device="cpu",
                    token_scale=packed.token_scale,
                )
                scale = None
            return _score_binary(
                query_tokens,
                to_device(upload_packed, "cuda"),
                scale=scale,
                device="cuda",
                reducer=reducer_value,
                query_weights=weight_values,
                temperature=temperature_value,
                candidate_indices=candidates,
                force_generalized=True,
            )
        if use_token_scale:
            raise NotImplementedError("token scale scoring on CUDA requires CUDA-resident PackedDocs; call maxsim.to_device(packed, 'cuda') first")
        packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
        offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
        batch_kernel = getattr(_maxsim_cuda, "maxsim_cuda_batch", None)
        if batch_kernel is not None:
            batch_result = batch_kernel(
                np.ascontiguousarray(query_float, dtype=np.float32),
                packed_data,
                offsets,
                packed.dim,
                float(kernel_scale),
            )
            batch_result = _apply_vector_scale(batch_result, resolved_scale)
            return batch_result[0] if squeeze else batch_result

        result = np.empty(result_shape, dtype=np.float32)
        for batch_idx, query in enumerate(query_float):
            result[batch_idx] = _maxsim_cuda.maxsim_cuda(
                np.ascontiguousarray(query, dtype=np.float32),
                packed_data,
                offsets,
                packed.dim,
                float(kernel_scale),
            )
        result = _apply_vector_scale(result, resolved_scale)
        return result[0] if squeeze else result

    if _maxsim_cpp is not None and not use_token_scale and not use_generalized:
        packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
        offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
        result = np.empty(result_shape, dtype=np.float32)
        for batch_idx, query in enumerate(query_float):
            result[batch_idx] = _maxsim_cpp.maxsim_lut(
                np.ascontiguousarray(query, dtype=np.float32),
                packed_data,
                offsets,
                packed.dim,
                float(kernel_scale),
            )
        result = _apply_vector_scale(result, resolved_scale)
        return result[0] if squeeze else result

    signs = _unpack_signs(packed.data, packed.dim)
    token_scales = packed.token_scale if use_token_scale else None
    selected_docs = range(packed.num_docs) if candidates is None else candidates
    result = np.empty(result_shape, dtype=np.float32)
    for batch_idx, query_matrix in enumerate(query_float):
        batch_weights = None if weight_values is None else weight_values[batch_idx]
        for output_idx, doc_idx_value in enumerate(selected_docs):
            doc_idx = int(doc_idx_value)
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            doc = signs[start:end]
            if doc.shape[0] == 0:
                result[batch_idx, output_idx] = 0.0
                continue
            dots = query_matrix @ doc.T
            if token_scales is not None:
                dots = dots * token_scales[start:end][np.newaxis, :]
            doc_scale = resolved_scale[doc_idx] if _scale_is_vector(resolved_scale) else np.float32(resolved_scale)
            if reducer_value == "smoothsim":
                dots = dots * np.float32(doc_scale)
                doc_score = _reduce_similarities(dots, reducer_value, batch_weights, temperature_value)
            else:
                doc_score = _reduce_similarities(dots, reducer_value, batch_weights, temperature_value)
                doc_score = np.float32(doc_score * np.float32(doc_scale))
            result[batch_idx, output_idx] = doc_score

    return result[0] if squeeze else result


def topk_maxsim(
    query_tokens,
    packed: PackedDocs,
    k: int,
    *,
    scale=None,
    device="auto",
    reducer: Reducer = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
):
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > packed.num_docs:
        raise ValueError("k cannot exceed packed.num_docs")
    _validate_packed(packed)
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    resolved_scale = _resolve_scale(scale, packed)
    use_resident_vector_scale = _can_use_cuda_resident_vector_scale(resolved_scale, packed)
    use_token_scale = isinstance(packed.token_scale, np.ndarray)
    query = _as_numpy(query_tokens)
    if query.ndim == 2:
        batches = query[np.newaxis, :, :]
        squeeze = True
    elif query.ndim == 3:
        batches = query
        squeeze = False
    else:
        raise ValueError("query_tokens must have shape [query_tokens, dim] or [batch, query_tokens, dim]")
    if batches.shape[2] != packed.dim:
        raise ValueError(f"query dim={batches.shape[2]} does not match packed dim={packed.dim}")
    reducer_value, weight_values, temperature_value = _normalize_reducer_options(
        reducer,
        query_weights,
        temperature,
        batch=batches.shape[0],
        query_tokens=batches.shape[1],
    )
    if (
        reducer_value == "maxsim"
        and effective_device == "cuda"
        and packed.device == "cuda"
        and hasattr(packed.data, "topk_batch")
        and (not _scale_is_vector(resolved_scale) or use_resident_vector_scale)
    ):
        prefer_lut_topk = np.issubdtype(query.dtype, np.integer) and packed.dim == 128 and packed.num_docs > 128
        scores, indices = _cuda_resident_topk_batch(
            packed.data,
            np.ascontiguousarray(batches, dtype=np.float32),
            int(k),
            float(_kernel_scale(resolved_scale)),
            use_resident_vector_scale,
            prefer_lut_topk,
            use_token_scale,
        )
        return (scores[0], indices[0]) if squeeze else (scores, indices)

    scores = maxsim(
        query_tokens,
        packed,
        scale=scale,
        device=device,
        reducer=reducer_value,
        query_weights=weight_values,
        temperature=temperature_value,
    )
    if scores.ndim == 1:
        indices = _topk_indices_1d(scores, k)
        return scores[indices], indices

    all_indices = np.stack([_topk_indices_1d(row, k) for row in scores], axis=0)
    all_scores = np.take_along_axis(scores, all_indices, axis=1)
    return all_scores, all_indices


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    detach = getattr(value, "detach", None)
    if detach is not None:
        cpu = detach().cpu()
        return cpu.numpy()

    return np.asarray(value)


def _normalize_reducer_options(reducer, query_weights, temperature, *, batch: int, query_tokens: int):
    if not isinstance(reducer, str) or reducer not in _REDUCERS:
        choices = ", ".join(sorted(_REDUCERS))
        raise ValueError(f"reducer must be one of: {choices}")

    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a finite value greater than 0") from exc
    if not np.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be a finite value greater than 0")

    if reducer != "weighted_maxsim":
        if query_weights is not None:
            raise ValueError("query_weights is only supported with reducer='weighted_maxsim'")
        return reducer, None, temperature_value
    if query_weights is None:
        raise ValueError("query_weights is required with reducer='weighted_maxsim'")

    weights = _as_numpy(query_weights).astype(np.float32, copy=False)
    if weights.ndim == 1:
        if weights.shape[0] != query_tokens:
            raise ValueError("query_weights must have shape [query_tokens] or [batch, query_tokens]")
        weights = np.broadcast_to(weights[np.newaxis, :], (batch, query_tokens))
    elif weights.ndim == 2:
        if weights.shape != (batch, query_tokens):
            raise ValueError("query_weights must have shape [query_tokens] or [batch, query_tokens]")
    else:
        raise ValueError("query_weights must have shape [query_tokens] or [batch, query_tokens]")
    if not np.all(np.isfinite(weights)):
        raise ValueError("query_weights must contain only finite values")
    return reducer, np.ascontiguousarray(weights, dtype=np.float32), temperature_value


def _normalize_candidate_indices(candidate_indices, num_docs: int) -> np.ndarray | None:
    if candidate_indices is None:
        return None
    values = _as_numpy(candidate_indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("candidate_indices must be a one-dimensional integer array")
    normalized = np.ascontiguousarray(values, dtype=np.int64)
    if np.any(normalized < 0) or np.any(normalized >= num_docs):
        raise ValueError("candidate_indices must be between 0 and packed.num_docs - 1")
    return normalized


def _reduce_similarities(
    similarities: np.ndarray,
    reducer: str,
    query_weights: np.ndarray | None,
    temperature: float,
) -> np.float32:
    if similarities.shape[1] == 0 or similarities.shape[0] == 0:
        return np.float32(0.0)
    values = np.asarray(similarities, dtype=np.float32)
    if reducer == "maxsim":
        return np.float32(np.max(values, axis=1).sum(dtype=np.float32))
    if reducer == "weighted_maxsim":
        maxima = np.max(values, axis=1)
        return np.float32(np.sum(maxima * query_weights, dtype=np.float32))
    if reducer in {"topk2", "topk4"}:
        requested = 2 if reducer == "topk2" else 4
        count = min(requested, values.shape[1])
        top_values = np.partition(values, values.shape[1] - count, axis=1)[:, -count:]
        pooled = np.mean(top_values, axis=1, dtype=np.float32)
        return np.float32(np.sum(pooled, dtype=np.float32))
    if reducer == "smoothsim":
        tau = np.float32(temperature)
        maxima = np.max(values, axis=1)
        shifted = (values - maxima[:, np.newaxis]) / tau
        sum_exp = np.sum(np.exp(shifted), axis=1, dtype=np.float32)
        pooled = maxima + tau * np.log(sum_exp)
        return np.float32(np.sum(pooled, dtype=np.float32))
    raise AssertionError(f"unhandled reducer: {reducer}")


def _normalize_offsets(doc_offsets, num_tokens: int) -> np.ndarray:
    if doc_offsets is None:
        return np.arange(num_tokens + 1, dtype=np.int64)

    offsets = _as_numpy(doc_offsets).astype(np.int64, copy=False)
    if offsets.ndim != 1:
        raise ValueError("doc_offsets must be a 1-D array")
    if offsets.shape[0] < 2:
        raise ValueError("doc_offsets must contain at least [0, num_doc_tokens]")
    if int(offsets[0]) != 0:
        raise ValueError("doc_offsets must start at 0")
    if int(offsets[-1]) != num_tokens:
        raise ValueError("doc_offsets must end at num_doc_tokens")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("doc_offsets must be monotonically nondecreasing")
    return offsets.copy()


def _validate_packed(packed: PackedDocs) -> None:
    if not isinstance(packed, PackedDocs):
        raise TypeError("packed must be a PackedDocs instance")
    if packed.dim % 8 != 0:
        raise ValueError("packed.dim must be divisible by 8")
    _validate_scale_value(packed.scale, packed.num_docs)
    if packed.token_scale is not None:
        if not isinstance(packed.token_scale, np.ndarray):
            raise ValueError("packed.token_scale must be a [num_doc_tokens] vector or None")
        num_tokens = int(packed.doc_offsets[-1])
        if packed.token_scale.ndim != 1 or packed.token_scale.shape[0] != num_tokens:
            raise ValueError("packed.token_scale must have shape [num_doc_tokens]")
    if packed.device == "cuda":
        if not hasattr(packed.data, "maxsim_batch") and not hasattr(packed.data, "topk_batch"):
            raise ValueError("CUDA PackedDocs data must expose maxsim_batch or topk_batch")
        if packed.doc_offsets.ndim != 1 or packed.doc_offsets.shape[0] != packed.num_docs + 1:
            raise ValueError("packed.doc_offsets must have shape [num_docs + 1]")
        return
    if packed.device != "cpu":
        raise ValueError("packed.device must be 'cpu' or 'cuda'")
    if packed.data.ndim != 2 or packed.data.shape[1] != packed.dim // 8:
        raise ValueError("packed.data must have shape [num_doc_tokens, dim / 8]")
    if packed.doc_offsets.ndim != 1 or packed.doc_offsets.shape[0] != packed.num_docs + 1:
        raise ValueError("packed.doc_offsets must have shape [num_docs + 1]")


def _unpack_signs(data: np.ndarray, dim: int) -> np.ndarray:
    signs = np.empty((data.shape[0], dim), dtype=np.float32)
    for bit_idx in range(dim):
        bits = (data[:, bit_idx // 8] >> (bit_idx % 8)) & 1
        signs[:, bit_idx] = np.where(bits == 1, 1.0, -1.0)
    return signs


def _doc_mean_abs_scale(docs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    scales = np.zeros(offsets.shape[0] - 1, dtype=np.float32)
    abs_docs = np.abs(docs).astype(np.float32, copy=False)
    for doc_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if int(start) != int(end):
            scales[doc_idx] = np.float32(np.mean(abs_docs[int(start) : int(end)], dtype=np.float64))
    return scales


def _normalize_explicit_scale(scale, num_docs: int):
    values = _as_numpy(scale)
    if values.ndim == 0:
        return float(values)
    normalized = np.ascontiguousarray(values, dtype=np.float32)
    if normalized.ndim != 1 or normalized.shape[0] != num_docs:
        raise ValueError("scale vector must have shape [num_docs]")
    return normalized


def _validate_scale_value(scale, num_docs: int) -> None:
    if scale is None:
        return
    if isinstance(scale, np.ndarray):
        if scale.ndim != 1 or scale.shape[0] != num_docs:
            raise ValueError("packed.scale vector must have shape [num_docs]")
        return
    float(scale)


def _resolve_scale(scale, packed: PackedDocs):
    if scale is None:
        return 1.0 if packed.scale is None else packed.scale
    if isinstance(scale, str) and scale == "global":
        if packed.scale is None:
            raise ValueError("scale='global' requires a PackedDocs object with stored scale")
        if _scale_is_vector(packed.scale):
            raise ValueError("scale='global' requires a scalar stored scale")
        return float(packed.scale)
    if isinstance(scale, str) and scale in {"doc", "per_doc"}:
        if not _scale_is_vector(packed.scale):
            raise ValueError("scale='doc' requires a PackedDocs object with stored per-document scale")
        return packed.scale
    if isinstance(scale, str):
        raise ValueError("scale must be None, 'global', 'doc', a scalar, or a [num_docs] vector")
    return _normalize_explicit_scale(scale, packed.num_docs)


def _scale_is_vector(scale) -> bool:
    return isinstance(scale, np.ndarray)


def _kernel_scale(scale) -> float:
    return 1.0 if _scale_is_vector(scale) else float(scale)


def _apply_vector_scale(scores: np.ndarray, scale) -> np.ndarray:
    if not _scale_is_vector(scale):
        return scores
    return np.asarray(scores, dtype=np.float32) * scale.astype(np.float32, copy=False)


def _can_use_cuda_resident_vector_scale(scale, packed: PackedDocs) -> bool:
    if not _scale_is_vector(scale) or packed.device != "cuda" or scale is not packed.scale:
        return False
    has_scale_vector = getattr(packed.data, "has_scale_vector", False)
    if callable(has_scale_vector):
        has_scale_vector = has_scale_vector()
    return bool(has_scale_vector)


def _cuda_resident_maxsim_batch(handle, query: np.ndarray, scale: float, use_scale_vector: bool, use_token_scale: bool = False):
    if use_token_scale:
        return handle.maxsim_batch(query, scale, use_scale_vector, True)
    if use_scale_vector:
        return handle.maxsim_batch(query, scale, True)
    return handle.maxsim_batch(query, scale)


def _cuda_resident_topk_batch(handle, query: np.ndarray, k: int, scale: float, use_scale_vector: bool, prefer_lut: bool = False, use_token_scale: bool = False):
    if prefer_lut and hasattr(handle, "topk_lut_batch"):
        if use_token_scale:
            return handle.topk_lut_batch(query, k, scale, use_scale_vector, True)
        if use_scale_vector:
            return handle.topk_lut_batch(query, k, scale, True)
        return handle.topk_lut_batch(query, k, scale)
    if use_token_scale:
        return handle.topk_batch(query, k, scale, use_scale_vector, True)
    if use_scale_vector:
        return handle.topk_batch(query, k, scale, True)
    return handle.topk_batch(query, k, scale)


def _topk_indices_1d(scores: np.ndarray, k: int) -> np.ndarray:
    doc_ids = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((doc_ids, -scores))
    return order[:k].astype(np.int64, copy=False)
