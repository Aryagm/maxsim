from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bitmax._api import (
    PackedDocs,
    _as_numpy,
    _normalize_offsets,
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
):
    transformed_query = transform_query_dim_centroids(query_tokens, calibration)
    scores = maxsim(transformed_query, packed, device=device)
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
