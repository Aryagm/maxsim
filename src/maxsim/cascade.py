from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxsim._api import (
    _as_numpy,
    _normalize_candidate_indices,
    _normalize_reducer_options,
    _reduce_similarities,
)
from maxsim.experimental import (
    Int4PackedDocs,
    _pack_signed_int4_values,
    _validate_int4_data_matches_values,
    _validate_int4_packed,
    int4_maxsim,
    int4_to_device,
    pack_int4_symmetric,
    topk_int4_maxsim,
)

_NATIVE_PREFIX_TOPK_LIMIT = 32


@dataclass(frozen=True)
class ResidualInt4PackedDocs:
    prefix_data: object
    prefix_values: np.ndarray | None
    prefix_scale: float
    prefix_token_scale: np.ndarray
    residual_data: object
    residual_values: np.ndarray | None
    residual_scale: float
    residual_token_scale: np.ndarray
    doc_offsets: np.ndarray
    dim: int
    num_docs: int
    device: str = "cpu"

    @property
    def num_tokens(self) -> int:
        return int(self.doc_offsets[-1])

    @property
    def storage_bytes(self) -> int:
        code_bytes = self.num_tokens * self.dim // 2
        prefix_bytes = self.prefix_data.nbytes if isinstance(self.prefix_data, np.ndarray) else code_bytes
        residual_bytes = self.residual_data.nbytes if isinstance(self.residual_data, np.ndarray) else code_bytes
        return int(
            prefix_bytes
            + residual_bytes
            + 8
            + self.prefix_token_scale.nbytes
            + self.residual_token_scale.nbytes
        )


def pack_residual_int4(doc_embeddings, doc_offsets=None) -> ResidualInt4PackedDocs:
    docs = _as_numpy(doc_embeddings).astype(np.float32, copy=False)
    prefix = pack_int4_symmetric(
        docs,
        doc_offsets,
        scale_granularity="token",
    )
    prefix_reconstruction = _reconstruct_int4(
        prefix.values,
        prefix.scale,
        prefix.token_scale,
    )
    residual = np.asarray(docs - prefix_reconstruction, dtype=np.float32)
    residual_packed = pack_int4_symmetric(
        residual,
        prefix.doc_offsets,
        scale_granularity="token",
    )
    result = ResidualInt4PackedDocs(
        prefix_data=prefix.data,
        prefix_values=prefix.values,
        prefix_scale=prefix.scale,
        prefix_token_scale=prefix.token_scale,
        residual_data=residual_packed.data,
        residual_values=residual_packed.values,
        residual_scale=residual_packed.scale,
        residual_token_scale=residual_packed.token_scale,
        doc_offsets=prefix.doc_offsets,
        dim=prefix.dim,
        num_docs=prefix.num_docs,
        device="cpu",
    )
    _validate_residual_packed(result)
    return result


def residual_int4_to_device(
    packed: ResidualInt4PackedDocs,
    device: str = "cuda",
) -> ResidualInt4PackedDocs:
    if device != "cuda":
        raise ValueError("device must be 'cuda'")
    _validate_residual_packed(packed, verify_packed_values=True)
    if packed.device == "cuda":
        return packed

    cuda_prefix = int4_to_device(_prefix_view(packed))
    setter = getattr(cuda_prefix.data, "set_residual_int4", None)
    if setter is None:
        raise NotImplementedError(
            "residual int4 requires a CUDA build with set_residual_int4"
        )
    setter(
        np.ascontiguousarray(packed.residual_data, dtype=np.uint8),
        float(packed.residual_scale),
        np.ascontiguousarray(packed.residual_token_scale, dtype=np.float32),
    )
    return ResidualInt4PackedDocs(
        prefix_data=cuda_prefix.data,
        prefix_values=None,
        prefix_scale=cuda_prefix.scale,
        prefix_token_scale=cuda_prefix.token_scale,
        residual_data=cuda_prefix.data,
        residual_values=None,
        residual_scale=packed.residual_scale,
        residual_token_scale=packed.residual_token_scale.copy(),
        doc_offsets=packed.doc_offsets.copy(),
        dim=packed.dim,
        num_docs=packed.num_docs,
        device="cuda",
    )


def prefix_score(
    query_tokens,
    packed: ResidualInt4PackedDocs,
    *,
    device: str = "auto",
    reducer: str = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
    candidate_indices=None,
) -> np.ndarray:
    _validate_residual_packed(packed)
    batches, squeeze = _normalize_query(query_tokens, packed.dim)
    result = int4_maxsim(
        batches[0] if squeeze else batches,
        _prefix_view(packed),
        device=device,
        reducer=reducer,
        query_weights=query_weights,
        temperature=temperature,
        candidate_indices=candidate_indices,
    )
    return result


def prefix_topk(
    query_tokens,
    packed: ResidualInt4PackedDocs,
    k: int,
    *,
    device: str = "auto",
    reducer: str = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix top-k with stable ties and bounded selector complexity."""
    _validate_residual_packed(packed)
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    if k < 1 or k > packed.num_docs:
        raise ValueError("k must satisfy 1 <= k <= num_docs")
    batches, squeeze = _normalize_query(query_tokens, packed.dim)
    query = batches[0] if squeeze else batches
    if k <= _NATIVE_PREFIX_TOPK_LIMIT:
        return topk_int4_maxsim(
            query,
            _prefix_view(packed),
            k,
            device=device,
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
        )
    scores = prefix_score(
        query,
        packed,
        device=device,
        reducer=reducer,
        query_weights=query_weights,
        temperature=temperature,
    )
    return _stable_topk(scores, k)


def residual_score(
    query_tokens,
    packed: ResidualInt4PackedDocs,
    *,
    device: str = "auto",
    reducer: str = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
    candidate_indices=None,
) -> np.ndarray:
    _validate_residual_packed(packed)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    effective_device = "cuda" if device == "auto" and packed.device == "cuda" else device
    if effective_device == "cuda" and packed.device == "cpu":
        return residual_score(
            query_tokens,
            residual_int4_to_device(packed),
            device="cuda",
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
            candidate_indices=candidate_indices,
        )
    if packed.device == "cuda" and effective_device != "cuda":
        raise NotImplementedError(
            "CPU scoring for CUDA ResidualInt4PackedDocs is not implemented"
        )

    batches, squeeze = _normalize_query(query_tokens, packed.dim)
    reducer_value, weights, temperature_value = _normalize_reducer_options(
        reducer,
        query_weights,
        temperature,
        batch=batches.shape[0],
        query_tokens=batches.shape[1],
    )
    candidates = _normalize_candidate_indices(candidate_indices, packed.num_docs)
    num_targets = packed.num_docs if candidates is None else int(candidates.shape[0])
    result_shape = (batches.shape[0], num_targets)
    if batches.shape[0] == 0 or num_targets == 0:
        result = np.empty(result_shape, dtype=np.float32)
        return result[0] if squeeze else result
    if batches.shape[1] == 0:
        result = np.zeros(result_shape, dtype=np.float32)
        return result[0] if squeeze else result

    if effective_device == "cuda":
        method_name = (
            "score_residual_batch"
            if candidates is None
            else "score_residual_candidates_batch"
        )
        method = getattr(packed.prefix_data, method_name, None)
        if method is None:
            raise NotImplementedError(
                f"this CUDA build does not provide {method_name}"
            )
        query = np.ascontiguousarray(batches, dtype=np.float32)
        if candidates is None:
            result = method(
                query,
                reducer_value,
                weights,
                float(temperature_value),
            )
        else:
            result = method(
                query,
                candidates,
                reducer_value,
                weights,
                float(temperature_value),
            )
        return result[0] if squeeze else result

    docs = _reconstruct_fused(packed)
    selected = range(packed.num_docs) if candidates is None else candidates
    result = np.empty(result_shape, dtype=np.float32)
    for batch_idx, query in enumerate(batches):
        batch_weights = None if weights is None else weights[batch_idx]
        for output_idx, doc_idx_value in enumerate(selected):
            doc_idx = int(doc_idx_value)
            start = int(packed.doc_offsets[doc_idx])
            end = int(packed.doc_offsets[doc_idx + 1])
            if start == end:
                result[batch_idx, output_idx] = 0.0
                continue
            similarities = query @ docs[start:end].T
            result[batch_idx, output_idx] = _reduce_similarities(
                similarities,
                reducer_value,
                batch_weights,
                temperature_value,
            )
    return result[0] if squeeze else result


def cascade_topk(
    query_tokens,
    packed: ResidualInt4PackedDocs,
    k: int,
    *,
    candidates: int,
    device: str = "auto",
    reducer: str = "maxsim",
    query_weights=None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_residual_packed(packed)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    if not isinstance(candidates, (int, np.integer)) or isinstance(
        candidates, (bool, np.bool_)
    ):
        raise ValueError("candidates must be an integer")
    if k < 1 or k > candidates or candidates > packed.num_docs:
        raise ValueError("cascade sizes must satisfy 1 <= k <= candidates <= num_docs")

    runtime_packed = (
        residual_int4_to_device(packed)
        if device == "cuda" and packed.device == "cpu"
        else packed
    )
    batches, squeeze = _normalize_query(query_tokens, runtime_packed.dim)
    _reducer, normalized_weights, _temperature = _normalize_reducer_options(
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

    query = batches[0] if squeeze else batches
    weights = (
        normalized_weights[0]
        if squeeze and normalized_weights is not None
        else normalized_weights
    )
    if candidates == runtime_packed.num_docs:
        fine_scores = residual_score(
            query,
            runtime_packed,
            device=device,
            reducer=reducer,
            query_weights=weights,
            temperature=temperature,
        )
        return _stable_topk(fine_scores, k)

    # The CUDA top-k selector is tuned for small final k. Using it for a large
    # cascade budget makes selection O(num_docs * candidates). Full coarse
    # scores are already copied to the host by the public API, so select stable
    # top-M in expected O(num_docs + M log M) time instead.
    coarse_scores = prefix_score(
        query,
        runtime_packed,
        device=device,
        reducer=reducer,
        query_weights=weights,
        temperature=temperature,
    )
    coarse_indices = _stable_topk(coarse_scores, candidates)[1]
    if squeeze:
        return _cascade_row(
            batches[0],
            np.asarray(coarse_indices, dtype=np.int64),
            runtime_packed,
            k=k,
            device=device,
            reducer=reducer,
            query_weights=weights,
            temperature=temperature,
        )

    all_scores = []
    all_indices = []
    for batch_idx, query in enumerate(batches):
        row_scores, row_indices = _cascade_row(
            query,
            coarse_indices[batch_idx],
            runtime_packed,
            k=k,
            device=device,
            reducer=reducer,
            query_weights=None if normalized_weights is None else normalized_weights[batch_idx],
            temperature=temperature,
        )
        all_scores.append(row_scores)
        all_indices.append(row_indices)
    return np.stack(all_scores), np.stack(all_indices)


def _cascade_row(
    query: np.ndarray,
    coarse_indices: np.ndarray,
    packed: ResidualInt4PackedDocs,
    *,
    k: int,
    device: str,
    reducer: str,
    query_weights,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    fine_scores = residual_score(
        query,
        packed,
        device=device,
        reducer=reducer,
        query_weights=query_weights,
        temperature=temperature,
        candidate_indices=coarse_indices,
    )
    order = _stable_topk_indices(fine_scores, k, tie_ids=coarse_indices)
    return fine_scores[order], coarse_indices[order]


def _stable_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim == 1:
        indices = _stable_topk_indices(values, k)
        return values[indices], indices
    if values.ndim != 2:
        raise ValueError("scores must be one- or two-dimensional")
    if values.shape[0] == 0:
        return (
            np.empty((0, k), dtype=np.float32),
            np.empty((0, k), dtype=np.int64),
        )
    indices = np.stack([_stable_topk_indices(row, k) for row in values])
    return np.take_along_axis(values, indices, axis=1), indices


def _stable_topk_indices(
    scores: np.ndarray,
    k: int,
    *,
    tie_ids: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.arange(scores.shape[0], dtype=np.int64)
    stable_ids = positions if tie_ids is None else np.asarray(tie_ids, dtype=np.int64)
    if k == scores.shape[0]:
        selected = positions
    else:
        cutoff = np.partition(scores, scores.shape[0] - k)[scores.shape[0] - k]
        better = positions[scores > cutoff]
        tied = positions[scores == cutoff]
        tied = tied[np.argsort(stable_ids[tied], kind="stable")]
        selected = np.concatenate((better, tied[: k - better.shape[0]]))
    order = np.lexsort((stable_ids[selected], -scores[selected]))
    return np.ascontiguousarray(selected[order], dtype=np.int64)


def _prefix_view(packed: ResidualInt4PackedDocs) -> Int4PackedDocs:
    return Int4PackedDocs(
        data=packed.prefix_data,
        values=packed.prefix_values,
        doc_offsets=packed.doc_offsets,
        dim=packed.dim,
        num_docs=packed.num_docs,
        scale=packed.prefix_scale,
        device=packed.device,
        token_scale=packed.prefix_token_scale,
    )


def _reconstruct_fused(packed: ResidualInt4PackedDocs) -> np.ndarray:
    prefix = _reconstruct_int4(
        packed.prefix_values,
        packed.prefix_scale,
        packed.prefix_token_scale,
    )
    residual = _reconstruct_int4(
        packed.residual_values,
        packed.residual_scale,
        packed.residual_token_scale,
    )
    return np.asarray(prefix + residual, dtype=np.float32)


def _reconstruct_int4(
    values: np.ndarray | None,
    scale: float,
    token_scale: np.ndarray,
) -> np.ndarray:
    if values is None:
        raise ValueError("CPU reconstruction requires unpacked int4 values")
    return (
        values.astype(np.float32, copy=False)
        * np.float32(scale)
        * token_scale[:, np.newaxis]
    )


def _normalize_query(query_tokens, dim: int) -> tuple[np.ndarray, bool]:
    query = _as_numpy(query_tokens).astype(np.float32, copy=False)
    if query.ndim == 1:
        batches = query[np.newaxis, np.newaxis, :]
        squeeze = True
    elif query.ndim == 2:
        batches = query[np.newaxis, :, :]
        squeeze = True
    elif query.ndim == 3:
        batches = query
        squeeze = False
    else:
        raise ValueError(
            "query_tokens must have shape [dim], [query_tokens, dim], or "
            "[batch, query_tokens, dim]"
        )
    if batches.shape[2] != dim:
        raise ValueError(f"query dim={batches.shape[2]} does not match packed dim={dim}")
    if not np.all(np.isfinite(batches)):
        raise ValueError("query_tokens must contain only finite values")
    return np.ascontiguousarray(batches, dtype=np.float32), squeeze


def _validate_residual_packed(
    packed: ResidualInt4PackedDocs,
    *,
    verify_packed_values: bool = False,
) -> None:
    if not isinstance(packed, ResidualInt4PackedDocs):
        raise TypeError("packed must be a ResidualInt4PackedDocs instance")
    prefix = _prefix_view(packed)
    _validate_int4_packed(prefix)
    if verify_packed_values:
        _validate_int4_data_matches_values(prefix)
    if packed.prefix_token_scale is None:
        raise ValueError("prefix_token_scale must have shape [num_doc_tokens]")
    if packed.device == "cpu":
        if not isinstance(packed.residual_data, np.ndarray):
            raise ValueError("CPU residual_data must be a NumPy array")
        if packed.residual_data.dtype != np.uint8:
            raise ValueError("CPU residual_data must have dtype uint8")
        expected_data_shape = (packed.num_tokens, packed.dim // 2)
        if packed.residual_data.shape != expected_data_shape:
            raise ValueError(
                "residual_data must have shape [num_doc_tokens, dim / 2]"
            )
        if not isinstance(packed.residual_values, np.ndarray):
            raise ValueError("CPU residual_values must be a NumPy array")
        if packed.residual_values.shape != (packed.num_tokens, packed.dim):
            raise ValueError(
                "residual_values must have shape [num_doc_tokens, dim]"
            )
        if packed.residual_values.dtype != np.int8:
            raise ValueError("residual_values must have dtype int8")
        if np.any(packed.residual_values < -7) or np.any(
            packed.residual_values > 7
        ):
            raise ValueError("residual_values must be symmetric int4 values in [-7, 7]")
        if verify_packed_values and not np.array_equal(
            packed.residual_data,
            _pack_signed_int4_values(packed.residual_values),
        ):
            raise ValueError("residual_data does not encode residual_values")
    elif packed.device == "cuda":
        if packed.residual_values is not None:
            raise ValueError("CUDA residual_values must be None")
        if packed.prefix_data is not packed.residual_data:
            raise ValueError("CUDA prefix_data and residual_data must share one handle")
        handle = packed.prefix_data
        for name, expected in (
            ("dim", packed.dim),
            ("num_docs", packed.num_docs),
            ("num_tokens", packed.num_tokens),
        ):
            actual = getattr(handle, name, None)
            if actual is None or int(actual) != int(expected):
                raise ValueError(f"CUDA residual handle {name} does not match metadata")
        if getattr(handle, "has_token_scale_vector", None) is not True:
            raise ValueError("CUDA residual handle is missing prefix token scales")
        if getattr(handle, "has_residual_int4", None) is not True:
            raise ValueError("CUDA residual handle is missing residual int4 data")
    else:
        raise ValueError("device must be 'cpu' or 'cuda'")

    try:
        residual_scale = float(packed.residual_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("residual_scale must be a finite scalar > 0") from exc
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("residual_scale must be a finite scalar > 0")
    if (
        not isinstance(packed.residual_token_scale, np.ndarray)
        or packed.residual_token_scale.ndim != 1
        or packed.residual_token_scale.shape[0] != packed.num_tokens
    ):
        raise ValueError(
            "residual_token_scale must have shape [num_doc_tokens]"
        )
    if packed.residual_token_scale.dtype != np.float32:
        raise ValueError("residual_token_scale must have dtype float32")
    if not np.all(np.isfinite(packed.residual_token_scale)) or np.any(
        packed.residual_token_scale <= 0.0
    ):
        raise ValueError("residual_token_scale values must be finite and > 0")
