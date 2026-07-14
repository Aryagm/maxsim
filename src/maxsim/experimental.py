from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxsim._api import (
    PackedDocs,
    _as_numpy,
    _normalize_candidate_indices,
    _normalize_offsets,
    _normalize_reducer_options,
    _reduce_similarities,
    maxsim,
    pack_signs,
    topk_maxsim,
)


@dataclass(frozen=True)
class DimCentroidCalibration:
    thresholds: np.ndarray
    negative_centroids: np.ndarray
    positive_centroids: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.thresholds.shape[0])

    @property
    def metadata_bytes(self) -> int:
        return int(self.dim * 3 * 4)


@dataclass(frozen=True)
class TernaryPackedDocs:
    values: np.ndarray
    doc_offsets: np.ndarray
    dim: int
    num_docs: int
    threshold: float

    @property
    def storage_bytes(self) -> int:
        return int(self.values.shape[0] * self.dim * 2 // 8)


@dataclass(frozen=True)
class Int4PackedDocs:
    data: object
    values: np.ndarray | None
    doc_offsets: np.ndarray
    dim: int
    num_docs: int
    scale: float
    device: str = "cpu"
    token_scale: np.ndarray | None = None

    @property
    def storage_bytes(self) -> int:
        if isinstance(self.data, np.ndarray):
            packed_bytes = int(self.data.nbytes)
        else:
            packed_bytes = int(getattr(self.data, "packed_size", 0))
        token_scale_bytes = 0 if self.token_scale is None else int(self.token_scale.nbytes)
        return packed_bytes + 4 + token_scale_bytes


def fit_dim_centroid_calibration(
    doc_embeddings,
    *,
    thresholds=None,
    lloyd_iterations: int = 0,
) -> DimCentroidCalibration:
    docs = _as_numpy(doc_embeddings).astype(np.float32, copy=False)
    if docs.ndim != 2:
        raise ValueError("doc_embeddings must have shape [num_doc_tokens, dim]")
    if docs.shape[1] % 8 != 0:
        raise ValueError("dim must be divisible by 8")
    if lloyd_iterations < 0:
        raise ValueError("lloyd_iterations must be >= 0")

    if thresholds is None:
        threshold_values = np.zeros(docs.shape[1], dtype=np.float32)
    else:
        threshold_values = _normalize_thresholds(thresholds, docs.shape[1])

    negative, positive = _fit_dim_centroids(docs, threshold_values)
    for _ in range(lloyd_iterations):
        next_thresholds = ((negative + positive) * 0.5).astype(np.float32)
        negative, positive = _fit_dim_centroids(docs, next_thresholds)
        if np.allclose(next_thresholds, threshold_values, rtol=0.0, atol=1e-6):
            threshold_values = next_thresholds
            break
        threshold_values = next_thresholds

    return DimCentroidCalibration(
        thresholds=np.ascontiguousarray(threshold_values, dtype=np.float32),
        negative_centroids=np.ascontiguousarray(negative, dtype=np.float32),
        positive_centroids=np.ascontiguousarray(positive, dtype=np.float32),
    )


def pack_dim_centroid_signs(
    doc_embeddings,
    doc_offsets=None,
    *,
    calibration: DimCentroidCalibration | None = None,
    thresholds=None,
    lloyd_iterations: int = 0,
    device: str = "cpu",
) -> tuple[PackedDocs, DimCentroidCalibration]:
    docs = _as_numpy(doc_embeddings).astype(np.float32, copy=False)
    if calibration is None:
        calibration = fit_dim_centroid_calibration(docs, thresholds=thresholds, lloyd_iterations=lloyd_iterations)
    _validate_dim_centroid_calibration(calibration, docs.shape[1])

    thresholded = docs - calibration.thresholds[np.newaxis, :]
    return pack_signs(thresholded, doc_offsets, device=device), calibration


def transform_query_dim_centroids(query_tokens, calibration: DimCentroidCalibration) -> np.ndarray:
    _validate_dim_centroid_calibration(calibration)
    query = _as_numpy(query_tokens).astype(np.float32, copy=False)
    if query.ndim not in {2, 3}:
        raise ValueError("query_tokens must have shape [query_tokens, dim] or [batch, query_tokens, dim]")
    if query.shape[-1] != calibration.dim:
        raise ValueError(f"query dim={query.shape[-1]} does not match calibration dim={calibration.dim}")
    weights = calibration.positive_centroids - calibration.negative_centroids
    return np.ascontiguousarray(query * weights, dtype=np.float32)


def dim_centroid_maxsim(
    query_tokens,
    packed: PackedDocs,
    calibration: DimCentroidCalibration,
    *,
    device="auto",
    restore_scores: bool = True,
    candidate_indices=None,
):
    transformed_query = transform_query_dim_centroids(query_tokens, calibration)
    scores = maxsim(transformed_query, packed, device=device, candidate_indices=candidate_indices)
    if not restore_scores:
        return scores
    return _restore_dim_centroid_scores(scores, query_tokens, calibration)


def topk_dim_centroid_maxsim(
    query_tokens,
    packed: PackedDocs,
    calibration: DimCentroidCalibration,
    k: int,
    *,
    device="auto",
    restore_scores: bool = True,
):
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cuda" and hasattr(packed.data, "topk_centroid_batch"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > packed.num_docs:
            raise ValueError("k cannot exceed packed.num_docs")
        _validate_dim_centroid_calibration(calibration, packed.dim)
        query = _as_numpy(query_tokens).astype(np.float32, copy=False)
        if query.ndim == 2:
            batches = query[np.newaxis, :, :]
            squeeze = True
        elif query.ndim == 3:
            batches = query
            squeeze = False
        else:
            raise ValueError("query_tokens must have shape [query_tokens, dim] or [batch, query_tokens, dim]")
        if batches.shape[2] != calibration.dim:
            raise ValueError(f"query dim={batches.shape[2]} does not match calibration dim={calibration.dim}")
        weights = np.ascontiguousarray(calibration.positive_centroids - calibration.negative_centroids, dtype=np.float32)
        scores, indices = packed.data.topk_centroid_batch(
            np.ascontiguousarray(batches, dtype=np.float32),
            weights,
            int(k),
            1.0,
            False,
        )
        if restore_scores:
            scores = _restore_dim_centroid_scores(scores, query_tokens, calibration)
        return (scores[0], indices[0]) if squeeze else (scores, indices)

    transformed_query = transform_query_dim_centroids(query_tokens, calibration)
    scores, indices = topk_maxsim(transformed_query, packed, k, device=device)
    if restore_scores:
        scores = _restore_dim_centroid_scores(scores, query_tokens, calibration)
    return scores, indices


def pack_ternary(doc_embeddings, doc_offsets=None, *, threshold: float = 0.0) -> TernaryPackedDocs:
    docs = _as_numpy(doc_embeddings)
    if docs.ndim != 2:
        raise ValueError("doc_embeddings must have shape [num_doc_tokens, dim]")
    if docs.shape[1] % 8 != 0:
        raise ValueError("dim must be divisible by 8")
    threshold_value = float(threshold)
    if threshold_value < 0.0:
        raise ValueError("threshold must be >= 0")

    offsets = _normalize_offsets(doc_offsets, docs.shape[0])
    values = np.zeros(docs.shape, dtype=np.int8)
    values[docs > threshold_value] = 1
    values[docs < -threshold_value] = -1
    return TernaryPackedDocs(
        values=np.ascontiguousarray(values, dtype=np.int8),
        doc_offsets=offsets,
        dim=int(docs.shape[1]),
        num_docs=int(offsets.shape[0] - 1),
        threshold=threshold_value,
    )


def pack_int4_symmetric(
    doc_embeddings,
    doc_offsets=None,
    *,
    scale: float | None = None,
    scale_granularity: str = "tensor",
) -> Int4PackedDocs:
    docs = _as_numpy(doc_embeddings).astype(np.float32, copy=False)
    if docs.ndim != 2:
        raise ValueError("doc_embeddings must have shape [num_doc_tokens, dim]")
    if docs.shape[1] <= 0:
        raise ValueError("dim must be positive")
    if docs.shape[1] % 2 != 0:
        raise ValueError("dim must be divisible by 2 for int4 packing")
    if docs.shape[1] % 8 != 0:
        raise ValueError("dim must be divisible by 8")
    if not np.all(np.isfinite(docs)):
        raise ValueError("doc_embeddings must contain only finite values")
    if scale_granularity not in {"tensor", "token"}:
        raise ValueError("scale_granularity must be 'tensor' or 'token'")

    token_scale = None
    if scale_granularity == "token":
        if scale is not None:
            raise ValueError("scale cannot be provided when scale_granularity='token'")
        token_scale = np.max(np.abs(docs), axis=1).astype(np.float32) / np.float32(7.0)
        token_scale[token_scale == 0.0] = 1.0
        scale_value = 1.0
        values = np.clip(np.rint(docs / token_scale[:, np.newaxis]), -7, 7).astype(np.int8)
    else:
        if scale is None:
            max_abs = float(np.max(np.abs(docs))) if docs.size else 0.0
            scale_value = 1.0 if max_abs == 0.0 else max_abs / 7.0
        else:
            scale_value = float(scale)
            if not np.isfinite(scale_value) or scale_value <= 0.0:
                raise ValueError("scale must be finite and > 0")
        values = np.clip(np.rint(docs / scale_value), -7, 7).astype(np.int8)

    packed = _pack_signed_int4_values(values)
    offsets = _normalize_offsets(doc_offsets, docs.shape[0])
    return Int4PackedDocs(
        data=packed,
        values=np.ascontiguousarray(values, dtype=np.int8),
        doc_offsets=offsets,
        dim=int(docs.shape[1]),
        num_docs=int(offsets.shape[0] - 1),
        scale=scale_value,
        device="cpu",
        token_scale=token_scale,
    )


def int4_to_device(packed: Int4PackedDocs, device: str = "cuda") -> Int4PackedDocs:
    if device != "cuda":
        raise ValueError("device must be 'cuda'")
    _validate_int4_packed(packed)
    if packed.device == "cuda":
        return packed
    _validate_int4_data_matches_values(packed)
    try:
        from maxsim import _maxsim_cuda
    except ImportError as exc:  # pragma: no cover - depends on CUDA build
        raise NotImplementedError("CUDA int4 packed docs are not available in this build") from exc
    if not hasattr(_maxsim_cuda, "CudaInt4PackedDocs"):
        raise NotImplementedError("CUDA int4 packed docs are not available in this build")
    handle = _maxsim_cuda.CudaInt4PackedDocs(
        np.ascontiguousarray(packed.data, dtype=np.uint8),
        np.ascontiguousarray(packed.doc_offsets, dtype=np.int64),
        packed.dim,
        float(packed.scale),
    )
    stored_token_scale = None if packed.token_scale is None else packed.token_scale.copy()
    if stored_token_scale is not None:
        setter = getattr(handle, "set_token_scale_vector", None)
        if setter is None:
            raise NotImplementedError("per-token int4 scales require a CUDA build with set_token_scale_vector")
        setter(np.ascontiguousarray(stored_token_scale, dtype=np.float32))
    return Int4PackedDocs(
        data=handle,
        values=None,
        doc_offsets=packed.doc_offsets.copy(),
        dim=packed.dim,
        num_docs=packed.num_docs,
        scale=packed.scale,
        device="cuda",
        token_scale=stored_token_scale,
    )


def int4_maxsim(
    query_tokens,
    packed: Int4PackedDocs,
    *,
    device="auto",
    reducer="maxsim",
    query_weights=None,
    temperature: float = 1.0,
    candidate_indices=None,
) -> np.ndarray:
    return _int4_score(
        query_tokens,
        packed,
        device=device,
        reducer=reducer,
        query_weights=query_weights,
        temperature=temperature,
        candidate_indices=candidate_indices,
        force_generalized=False,
    )


def _int4_score(
    query_tokens,
    packed: Int4PackedDocs,
    *,
    device,
    reducer,
    query_weights,
    temperature,
    candidate_indices,
    force_generalized: bool,
) -> np.ndarray:
    _validate_int4_packed(packed)
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cpu":
        return _int4_score(
            query_tokens,
            int4_to_device(packed),
            device="cuda",
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
            candidate_indices=candidate_indices,
            force_generalized=force_generalized,
        )

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
    if num_targets == 0 or query_float.shape[0] == 0:
        result = np.empty(result_shape, dtype=np.float32)
        return result[0] if squeeze else result
    if query_float.shape[1] == 0:
        result = np.zeros(result_shape, dtype=np.float32)
        return result[0] if squeeze else result

    use_generalized = force_generalized or reducer_value != "maxsim" or candidates is not None
    if effective_device == "cuda" and packed.device == "cuda":
        contiguous_query = np.ascontiguousarray(query_float, dtype=np.float32)
        if use_generalized:
            method_name = "score_batch" if candidates is None else "score_candidates_batch"
            method = getattr(packed.data, method_name, None)
            if method is None:
                raise NotImplementedError(f"this CUDA build does not provide {method_name}")
            if candidates is None:
                result = method(contiguous_query, reducer_value, weight_values, float(temperature_value))
            else:
                result = method(contiguous_query, candidates, reducer_value, weight_values, float(temperature_value))
        else:
            result = packed.data.maxsim_batch(contiguous_query)
        return result[0] if squeeze else result
    if packed.device == "cuda":
        raise NotImplementedError("CPU scoring for CUDA Int4PackedDocs is not implemented")

    docs = packed.values.astype(np.float32, copy=False) * np.float32(packed.scale)
    if packed.token_scale is not None:
        docs = docs * packed.token_scale[:, np.newaxis]
    selected_docs = range(packed.num_docs) if candidates is None else candidates
    result = np.empty(result_shape, dtype=np.float32)
    for batch_idx, query_matrix in enumerate(query_float):
        batch_weights = None if weight_values is None else weight_values[batch_idx]
        for output_idx, doc_idx_value in enumerate(selected_docs):
            doc_idx = int(doc_idx_value)
            start = packed.doc_offsets[doc_idx]
            end = packed.doc_offsets[doc_idx + 1]
            doc = docs[int(start) : int(end)]
            result[batch_idx, output_idx] = (
                _reduce_similarities(query_matrix @ doc.T, reducer_value, batch_weights, temperature_value)
                if doc.shape[0]
                else 0.0
            )
    return result[0] if squeeze else result


def int4_maxsim_int8q(query_tokens, packed: Int4PackedDocs, *, device="auto") -> np.ndarray:
    """Score with the dp4a int8-query x int4-doc CUDA kernel (dim=128 only)."""
    _validate_int4_packed(packed)
    query = _as_numpy(query_tokens).astype(np.float32, copy=False)
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
    if batches.shape[0] == 0:
        return np.empty((0, packed.num_docs), dtype=np.float32)
    if batches.shape[1] == 0:
        result = np.zeros((batches.shape[0], packed.num_docs), dtype=np.float32)
        return result[0] if squeeze else result

    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cpu":
        return int4_maxsim_int8q(
            query_tokens,
            int4_to_device(packed),
            device="cuda",
        )
    if packed.device != "cuda" or not hasattr(packed.data, "maxsim_batch_int8q"):
        raise NotImplementedError("int8-query int4 scoring requires a CUDA build with maxsim_batch_int8q")

    result = packed.data.maxsim_batch_int8q(np.ascontiguousarray(batches, dtype=np.float32))
    return result[0] if squeeze else result


def topk_int4_maxsim(
    query_tokens,
    packed: Int4PackedDocs,
    k: int,
    *,
    device="auto",
    prefer_int8_query: bool = False,
    reducer="maxsim",
    query_weights=None,
    temperature: float = 1.0,
):
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > packed.num_docs:
        raise ValueError("k cannot exceed packed.num_docs")
    _validate_int4_packed(packed)
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cpu":
        return topk_int4_maxsim(
            query_tokens,
            int4_to_device(packed),
            k,
            device="cuda",
            prefer_int8_query=prefer_int8_query,
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
        )

    query = _as_numpy(query_tokens).astype(np.float32, copy=False)
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
    if batches.shape[0] == 0:
        return (
            np.empty((0, k), dtype=np.float32),
            np.empty((0, k), dtype=np.int64),
        )
    if (
        reducer_value == "maxsim"
        and prefer_int8_query
        and effective_device == "cuda"
        and packed.device == "cuda"
        and packed.dim == 128
        and hasattr(packed.data, "topk_batch_int8q")
    ):
        scores, indices = packed.data.topk_batch_int8q(np.ascontiguousarray(batches, dtype=np.float32), int(k))
        return (scores[0], indices[0]) if squeeze else (scores, indices)
    if reducer_value == "maxsim" and effective_device == "cuda" and packed.device == "cuda" and hasattr(packed.data, "topk_batch"):
        scores, indices = packed.data.topk_batch(np.ascontiguousarray(batches, dtype=np.float32), int(k))
        return (scores[0], indices[0]) if squeeze else (scores, indices)

    scores = int4_maxsim(
        query_tokens,
        packed,
        device=device,
        reducer=reducer_value,
        query_weights=weight_values,
        temperature=temperature_value,
    )
    if scores.ndim == 1:
        indices = _topk_indices(scores, k)
        return scores[indices], indices
    all_indices = np.stack([_topk_indices(row, k) for row in scores], axis=0)
    return np.take_along_axis(scores, all_indices, axis=1), all_indices


def ternary_maxsim(query_tokens, packed: TernaryPackedDocs) -> np.ndarray:
    _validate_ternary_packed(packed)
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
    docs = packed.values.astype(np.float32, copy=False)
    result = np.empty((query_float.shape[0], packed.num_docs), dtype=np.float32)
    for batch_idx, query_matrix in enumerate(query_float):
        for doc_idx, (start, end) in enumerate(zip(packed.doc_offsets[:-1], packed.doc_offsets[1:])):
            doc = docs[int(start) : int(end)]
            result[batch_idx, doc_idx] = np.max(query_matrix @ doc.T, axis=1).sum(dtype=np.float32) if doc.shape[0] else 0.0
    return result[0] if squeeze else result


def topk_ternary_maxsim(query_tokens, packed: TernaryPackedDocs, k: int):
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > packed.num_docs:
        raise ValueError("k cannot exceed packed.num_docs")
    scores = ternary_maxsim(query_tokens, packed)
    if scores.ndim == 1:
        indices = _topk_indices(scores, k)
        return scores[indices], indices
    all_indices = np.stack([_topk_indices(row, k) for row in scores], axis=0)
    return np.take_along_axis(scores, all_indices, axis=1), all_indices


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    doc_ids = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((doc_ids, -scores))[:k].astype(np.int64, copy=False)


def _normalize_thresholds(thresholds, dim: int) -> np.ndarray:
    values = _as_numpy(thresholds).astype(np.float32, copy=False)
    if values.ndim != 1 or values.shape[0] != dim:
        raise ValueError("thresholds must have shape [dim]")
    return np.ascontiguousarray(values, dtype=np.float32)


def _fit_dim_centroids(docs: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    negative = np.empty(docs.shape[1], dtype=np.float32)
    positive = np.empty(docs.shape[1], dtype=np.float32)
    for dim_idx in range(docs.shape[1]):
        column = docs[:, dim_idx]
        mask = column >= thresholds[dim_idx]
        positive[dim_idx] = _mean_or_fallback(column[mask], thresholds[dim_idx])
        negative[dim_idx] = _mean_or_fallback(column[~mask], thresholds[dim_idx])
    return negative, positive


def _mean_or_fallback(values: np.ndarray, fallback: np.float32) -> np.float32:
    if values.size == 0:
        return np.float32(fallback)
    return np.float32(np.mean(values, dtype=np.float64))


def _pack_signed_int4_values(values: np.ndarray) -> np.ndarray:
    packed = np.zeros((values.shape[0], values.shape[1] // 2), dtype=np.uint8)
    low = (values[:, 0::2].astype(np.int16) & 0x0F).astype(np.uint8)
    high = (values[:, 1::2].astype(np.int16) & 0x0F).astype(np.uint8)
    packed[:, :] = low | (high << 4)
    return np.ascontiguousarray(packed, dtype=np.uint8)


def _restore_dim_centroid_scores(scores, query_tokens, calibration: DimCentroidCalibration) -> np.ndarray:
    score_values = _as_numpy(scores).astype(np.float32, copy=False)
    query = _as_numpy(query_tokens).astype(np.float32, copy=False)
    centroid_sum = calibration.positive_centroids + calibration.negative_centroids
    if query.ndim == 2:
        constant = np.float32(0.5 * np.sum(query * centroid_sum[np.newaxis, :], dtype=np.float64))
        return np.asarray(score_values * 0.5 + constant, dtype=np.float32)
    if query.ndim == 3:
        constants = 0.5 * np.sum(query * centroid_sum[np.newaxis, np.newaxis, :], axis=(1, 2), dtype=np.float64)
        return np.asarray(score_values * 0.5 + constants[:, np.newaxis].astype(np.float32), dtype=np.float32)
    raise ValueError("query_tokens must have shape [query_tokens, dim] or [batch, query_tokens, dim]")


def _validate_dim_centroid_calibration(calibration: DimCentroidCalibration, dim: int | None = None) -> None:
    if not isinstance(calibration, DimCentroidCalibration):
        raise TypeError("calibration must be a DimCentroidCalibration instance")
    if calibration.thresholds.ndim != 1:
        raise ValueError("calibration thresholds must be one-dimensional")
    expected_dim = calibration.thresholds.shape[0] if dim is None else int(dim)
    for name, values in (
        ("negative_centroids", calibration.negative_centroids),
        ("positive_centroids", calibration.positive_centroids),
    ):
        if values.ndim != 1 or values.shape[0] != expected_dim:
            raise ValueError(f"calibration {name} must have shape [dim]")
    if calibration.thresholds.shape[0] != expected_dim:
        raise ValueError("calibration thresholds must have shape [dim]")


def _validate_ternary_packed(packed: TernaryPackedDocs) -> None:
    if not isinstance(packed, TernaryPackedDocs):
        raise TypeError("packed must be a TernaryPackedDocs instance")
    if packed.values.ndim != 2 or packed.values.shape[1] != packed.dim:
        raise ValueError("packed.values must have shape [num_doc_tokens, dim]")
    if packed.doc_offsets.ndim != 1 or packed.doc_offsets.shape[0] != packed.num_docs + 1:
        raise ValueError("packed.doc_offsets must have shape [num_docs + 1]")


def _validate_int4_packed(packed: Int4PackedDocs) -> None:
    if not isinstance(packed, Int4PackedDocs):
        raise TypeError("packed must be an Int4PackedDocs instance")
    if not isinstance(packed.dim, (int, np.integer)) or packed.dim <= 0 or packed.dim % 8 != 0:
        raise ValueError("packed.dim must be positive and divisible by 8")
    if not isinstance(packed.num_docs, (int, np.integer)) or packed.num_docs < 0:
        raise ValueError("packed.num_docs must be a nonnegative integer")
    if packed.values is not None:
        if not isinstance(packed.values, np.ndarray) or packed.values.ndim != 2 or packed.values.shape[1] != packed.dim:
            raise ValueError("packed.values must have shape [num_doc_tokens, dim]")
        if packed.values.dtype != np.int8:
            raise ValueError("packed.values must have dtype int8")
        if np.any(packed.values < -7) or np.any(packed.values > 7):
            raise ValueError("packed.values must be symmetric int4 values in [-7, 7]")
    if packed.device == "cpu":
        if packed.values is None:
            raise ValueError("CPU Int4PackedDocs must include values")
        if (
            not isinstance(packed.data, np.ndarray)
            or packed.data.dtype != np.uint8
            or packed.data.ndim != 2
            or packed.data.shape != (packed.values.shape[0], packed.dim // 2)
        ):
            raise ValueError("packed.data must have shape [num_doc_tokens, dim / 2]")
    elif packed.device != "cuda":
        raise ValueError("packed.device must be 'cpu' or 'cuda'")
    if not isinstance(packed.doc_offsets, np.ndarray) or packed.doc_offsets.ndim != 1 or packed.doc_offsets.shape[0] != packed.num_docs + 1:
        raise ValueError("packed.doc_offsets must have shape [num_docs + 1]")
    if not np.issubdtype(packed.doc_offsets.dtype, np.integer):
        raise ValueError("packed.doc_offsets must contain integers")
    num_tokens = int(packed.values.shape[0]) if packed.values is not None else int(packed.doc_offsets[-1])
    if int(packed.doc_offsets[0]) != 0 or int(packed.doc_offsets[-1]) != num_tokens:
        raise ValueError("packed.doc_offsets must span all document tokens")
    if np.any(packed.doc_offsets[1:] < packed.doc_offsets[:-1]):
        raise ValueError("packed.doc_offsets must be monotonically nondecreasing")
    try:
        scale_array = np.asarray(packed.scale)
        if scale_array.ndim != 0:
            raise ValueError
        scale_value = float(scale_array)
    except (TypeError, ValueError) as exc:
        raise ValueError("packed.scale must be a finite scalar > 0") from exc
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("packed.scale must be a finite scalar > 0")
    if packed.token_scale is not None:
        if not isinstance(packed.token_scale, np.ndarray) or packed.token_scale.ndim != 1 or packed.token_scale.shape[0] != num_tokens:
            raise ValueError("packed.token_scale must have shape [num_doc_tokens]")
        if packed.token_scale.dtype != np.float32:
            raise ValueError("packed.token_scale must have dtype float32")
        if not np.all(np.isfinite(packed.token_scale)) or np.any(packed.token_scale <= 0.0):
            raise ValueError("packed.token_scale values must be finite and > 0")


def _validate_int4_data_matches_values(packed: Int4PackedDocs) -> None:
    """Reject ambiguous CPU payloads before crossing a persistence/device boundary."""
    if packed.device != "cpu":
        return
    expected = _pack_signed_int4_values(packed.values)
    if not np.array_equal(packed.data, expected):
        raise ValueError("packed.data does not encode packed.values")
