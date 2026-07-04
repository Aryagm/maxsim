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
    if scale == "global":
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


def maxsim(query_tokens, packed: PackedDocs, *, scale=None, device="auto"):
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
    result = np.empty((query_float.shape[0], packed.num_docs), dtype=np.float32)
    resolved_scale = _resolve_scale(scale, packed)
    kernel_scale = _kernel_scale(resolved_scale)
    use_token_scale = isinstance(packed.token_scale, np.ndarray)

    if effective_device == "cuda" and packed.device == "cuda":
        use_resident_vector_scale = _can_use_cuda_resident_vector_scale(resolved_scale, packed)
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

    if _maxsim_cpp is not None and not use_token_scale:
        packed_data = np.ascontiguousarray(packed.data, dtype=np.uint8)
        offsets = np.ascontiguousarray(packed.doc_offsets, dtype=np.int64)
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
    for batch_idx, query in enumerate(query_float):
        for doc_idx in range(packed.num_docs):
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            doc = signs[start:end]
            if doc.shape[0] == 0:
                result[batch_idx, doc_idx] = 0.0
                continue
            dots = query @ doc.T
            if token_scales is not None:
                dots = dots * token_scales[start:end][np.newaxis, :]
            score = np.max(dots, axis=1).sum(dtype=np.float32)
            result[batch_idx, doc_idx] = np.float32(score * kernel_scale)

    result = _apply_vector_scale(result, resolved_scale)
    return result[0] if squeeze else result


def topk_maxsim(query_tokens, packed: PackedDocs, k: int, *, scale=None, device="auto"):
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > packed.num_docs:
        raise ValueError("k cannot exceed packed.num_docs")
    _validate_packed(packed)
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    resolved_scale = _resolve_scale(scale, packed)
    use_resident_vector_scale = _can_use_cuda_resident_vector_scale(resolved_scale, packed)
    use_token_scale = isinstance(packed.token_scale, np.ndarray)
    if (
        effective_device == "cuda"
        and packed.device == "cuda"
        and hasattr(packed.data, "topk_batch")
        and (not _scale_is_vector(resolved_scale) or use_resident_vector_scale)
    ):
        query = _as_numpy(query_tokens)
        prefer_lut_topk = np.issubdtype(query.dtype, np.integer) and packed.dim == 128 and packed.num_docs > 128
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
