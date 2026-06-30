from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bitmax._api import _as_numpy, _normalize_offsets


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


def _validate_ternary_packed(packed: TernaryPackedDocs) -> None:
    if not isinstance(packed, TernaryPackedDocs):
        raise TypeError("packed must be a TernaryPackedDocs instance")
    if packed.values.ndim != 2 or packed.values.shape[1] != packed.dim:
        raise ValueError("packed.values must have shape [num_doc_tokens, dim]")
    if packed.doc_offsets.ndim != 1 or packed.doc_offsets.shape[0] != packed.num_docs + 1:
        raise ValueError("packed.doc_offsets must have shape [num_docs + 1]")
