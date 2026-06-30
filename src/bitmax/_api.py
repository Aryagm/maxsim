from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from bitmax import _bitmax_cpp
except ImportError:  # pragma: no cover - exercised only in pure Python builds
    _bitmax_cpp = None

try:
    from bitmax import _bitmax_cuda
except ImportError:  # pragma: no cover - exercised only in CUDA builds
    _bitmax_cuda = None


@dataclass(frozen=True)
class PackedDocs:
    data: object
    doc_offsets: np.ndarray
    dim: int
    num_docs: int
    scale: float | None = None
    device: Literal["cpu", "cuda"] = "cpu"


def pack_signs(doc_embeddings, doc_offsets=None, *, dim=None, scale=None, device="cpu") -> PackedDocs:
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

    stored_scale: float | None
    if scale == "global":
        stored_scale = float(np.mean(np.abs(docs), dtype=np.float64))
    elif scale is None:
        stored_scale = None
    else:
        stored_scale = float(scale)

    packed_docs = PackedDocs(
        data=packed,
        doc_offsets=offsets,
        dim=actual_dim,
        num_docs=int(offsets.shape[0] - 1),
        scale=stored_scale,
        device="cpu",
    )
    return packed_docs if device == "cpu" else to_device(packed_docs, "cuda")


def to_device(packed: PackedDocs, device: Literal["cpu", "cuda"] = "cuda") -> PackedDocs:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    _validate_packed(packed)
    if device == packed.device:
        return packed
    if device == "cpu":
        raise NotImplementedError("copying CUDA PackedDocs back to CPU is not implemented")
    if _bitmax_cuda is None or not hasattr(_bitmax_cuda, "CudaPackedDocs"):
        raise NotImplementedError("CUDA PackedDocs are not available in this build")

    packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
    offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
    handle = _bitmax_cuda.CudaPackedDocs(packed_data, offsets, packed.dim)
    return PackedDocs(
        data=handle,
        doc_offsets=offsets.copy(),
        dim=packed.dim,
        num_docs=packed.num_docs,
        scale=packed.scale,
        device="cuda",
    )


def maxsim(query_tokens, packed: PackedDocs, *, scale=None, device="auto"):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    effective_device = "cuda" if device == "auto" and isinstance(packed, PackedDocs) and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cpu" and _bitmax_cuda is None:
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
    result = np.empty((query_float.shape[0], packed.num_docs), dtype=np.float32)
    multiplier = _resolve_scale(scale, packed)

    if effective_device == "cuda" and packed.device == "cuda":
        batch_result = packed.data.maxsim_batch(np.ascontiguousarray(query_float, dtype=np.float32), float(multiplier))
        return batch_result[0] if squeeze else batch_result

    if effective_device == "cuda":
        packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
        offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
        batch_kernel = getattr(_bitmax_cuda, "maxsim_cuda_batch", None)
        if batch_kernel is not None:
            batch_result = batch_kernel(
                np.ascontiguousarray(query_float, dtype=np.float32),
                packed_data,
                offsets,
                packed.dim,
                float(multiplier),
            )
            return batch_result[0] if squeeze else batch_result

        for batch_idx, query in enumerate(query_float):
            result[batch_idx] = _bitmax_cuda.maxsim_cuda(
                np.ascontiguousarray(query, dtype=np.float32),
                packed_data,
                offsets,
                packed.dim,
                float(multiplier),
            )
        return result[0] if squeeze else result

    if _bitmax_cpp is not None:
        packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
        offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
        for batch_idx, query in enumerate(query_float):
            result[batch_idx] = _bitmax_cpp.maxsim_lut(
                np.ascontiguousarray(query, dtype=np.float32),
                packed_data,
                offsets,
                packed.dim,
                float(multiplier),
            )
        return result[0] if squeeze else result

    signs = _unpack_signs(packed.data, packed.dim)
    for batch_idx, query in enumerate(query_float):
        for doc_idx in range(packed.num_docs):
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            doc = signs[start:end]
            if doc.shape[0] == 0:
                result[batch_idx, doc_idx] = 0.0
                continue
            score = np.max(query @ doc.T, axis=1).sum(dtype=np.float32)
            result[batch_idx, doc_idx] = np.float32(score * multiplier)

    return result[0] if squeeze else result


def topk_maxsim(query_tokens, packed: PackedDocs, k: int, *, scale=None, device="auto"):
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > packed.num_docs:
        raise ValueError("k cannot exceed packed.num_docs")
    _validate_packed(packed)
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cuda" and hasattr(packed.data, "topk_batch"):
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
        multiplier = _resolve_scale(scale, packed)
        scores, indices = packed.data.topk_batch(np.ascontiguousarray(batches, dtype=np.float32), int(k), float(multiplier))
        return (scores[0], indices[0]) if squeeze else (scores, indices)

    scores = maxsim(query_tokens, packed, scale=scale, device=device)
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


def _resolve_scale(scale, packed: PackedDocs) -> float:
    if scale is None:
        return 1.0 if packed.scale is None else float(packed.scale)
    if scale == "global":
        if packed.scale is None:
            raise ValueError("scale='global' requires a PackedDocs object with stored scale")
        return float(packed.scale)
    return float(scale)


def _topk_indices_1d(scores: np.ndarray, k: int) -> np.ndarray:
    doc_ids = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((doc_ids, -scores))
    return order[:k].astype(np.int64, copy=False)
