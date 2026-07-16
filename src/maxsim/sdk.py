from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from maxsim._api import PackedDocs, _as_numpy, maxsim, pack_signs, to_device, topk_maxsim
from maxsim.cascade import (
    ResidualInt4PackedDocs,
    _stable_topk,
    _validate_residual_packed,
    cascade_topk,
    pack_residual_int4,
    residual_int4_to_device,
    residual_score,
)
from maxsim.experimental import (
    DimCentroidCalibration,
    Int4PackedDocs,
    _validate_int4_data_matches_values,
    _validate_int4_packed,
    dim_centroid_maxsim,
    fit_dim_centroid_calibration,
    int4_maxsim,
    int4_to_device,
    pack_dim_centroid_signs,
    pack_int4_symmetric,
    topk_dim_centroid_maxsim,
    topk_int4_maxsim,
)

CorpusMode = Literal[
    "binary",
    "binary_token_scale",
    "binary_token_scale_u8",
    "binary_token_scale_u4",
    "pooled_binary",
    "binary_q40",
    "int4",
    "int4_per_token",
    "int4_residual",
]

# Named accuracy/size/latency tiers over the measured Pareto frontier.
MODE_PRESETS = {
    "balanced": "binary_token_scale",
    "max_quality": "int4_per_token",
    "max_compression": "pooled_binary",
    "compact": "binary_token_scale_u4",
    "max_speed": "binary",
}

def resolve_auto_mode(num_docs: int, embeddings: np.ndarray | None = None) -> str:
    """Return the per-token int4 convenience preset."""
    del num_docs, embeddings
    return "int4_per_token"


@dataclass(frozen=True)
class CorpusMemoryReport:
    """Byte counts for the encoded corpus and its current runtime residency."""

    encoded_bytes: int
    host_array_bytes: int
    device_index_bytes: int
    device_workspace_bytes: int | None
    serialized_bytes: int | None


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class Corpus:
    doc_ids: tuple[str, ...]
    mode: str
    packed: PackedDocs | None = None
    int4_packed: object | None = None
    calibration: object | None = None
    metadata: dict[str, Any] | None = None
    _source_path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_embeddings(
        cls,
        doc_ids,
        embeddings,
        offsets,
        *,
        mode: str = "auto",
        pool_factor: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Corpus":
        requested_mode = mode
        doc_id_values = _normalize_doc_ids(doc_ids)
        docs = _as_numpy(embeddings).astype(np.float32, copy=False)
        if mode == "auto":
            mode = resolve_auto_mode(len(doc_id_values), docs)
        else:
            mode = MODE_PRESETS.get(mode, mode)
        offsets_array = _normalize_sdk_offsets(offsets, docs.shape[0])
        _validate_corpus_inputs(doc_id_values, docs, offsets_array, mode)
        if mode == "pooled_binary":
            from maxsim.pooling import pool_doc_tokens

            effective_pool_factor = 3 if requested_mode == "max_compression" and pool_factor is None else pool_factor
            if effective_pool_factor is None:
                effective_pool_factor = 2
            if isinstance(effective_pool_factor, bool) or not isinstance(effective_pool_factor, (int, np.integer)):
                raise ValueError("pool_factor must be an integer >= 1")
            if int(effective_pool_factor) < 1:
                raise ValueError("pool_factor must be an integer >= 1")
            effective_pool_factor = int(effective_pool_factor)
            original_tokens = int(docs.shape[0])
            docs, offsets_array = pool_doc_tokens(docs, offsets_array, effective_pool_factor)
            pool_metadata = {
                "pool_factor": int(effective_pool_factor),
                "original_tokens": original_tokens,
                "pooled_tokens": int(docs.shape[0]),
            }
            merged = {} if metadata is None else dict(metadata)
            merged.update(pool_metadata)
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=pack_signs(docs, offsets_array),
                metadata=merged,
            )
        if mode == "binary":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=pack_signs(docs, offsets_array),
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "binary_token_scale":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=pack_signs(docs, offsets_array, token_scale="mean_abs_fp16"),
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "binary_token_scale_u8":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=pack_signs(docs, offsets_array, token_scale="mean_abs_u8"),
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "binary_token_scale_u4":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=pack_signs(docs, offsets_array, token_scale="mean_abs_u4"),
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "binary_q40":
            thresholds = np.percentile(docs, 40.0, axis=0).astype(np.float32)
            calibration = fit_dim_centroid_calibration(docs, thresholds=thresholds)
            packed, calibration = pack_dim_centroid_signs(docs, offsets_array, calibration=calibration)
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=packed,
                calibration=calibration,
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "int4":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                int4_packed=pack_int4_symmetric(docs, offsets_array),
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "int4_per_token":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                int4_packed=pack_int4_symmetric(docs, offsets_array, scale_granularity="token"),
                metadata={} if metadata is None else dict(metadata),
            )
        if mode == "int4_residual":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                int4_packed=pack_residual_int4(docs, offsets_array),
                metadata={} if metadata is None else dict(metadata),
            )
        raise ValueError(
            "mode must be 'auto', one of 'binary', 'binary_token_scale', 'binary_token_scale_u8', "
            "'binary_token_scale_u4', 'pooled_binary', 'binary_q40', 'int4', "
            "'int4_per_token', 'int4_residual', "
            f"or a preset in {sorted(MODE_PRESETS)}"
        )

    def save(self, path) -> None:
        _validate_corpus(self)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, object] = {
            "schema_version": np.array(2, dtype=np.int64),
            "format": np.array("maxsim_corpus"),
            "mode": np.array(self.mode),
            "doc_ids": np.asarray(self.doc_ids),
            "metadata_json": np.array(json.dumps({} if self.metadata is None else self.metadata, sort_keys=True)),
        }
        if self.packed is not None:
            if self.packed.device != "cpu":
                raise ValueError("Corpus.save requires CPU packed docs")
            scale_kind, scale_values = _encode_scale(self.packed.scale)
            arrays.update(
                {
                    "packed_data": np.ascontiguousarray(self.packed.data, dtype=np.uint8),
                    "doc_offsets": np.ascontiguousarray(self.packed.doc_offsets, dtype=np.int64),
                    "dim": np.array(self.packed.dim, dtype=np.int64),
                    "scale_kind": np.array(scale_kind),
                    "scale_values": scale_values,
                }
            )
            if self.packed.token_scale is not None:
                if self.mode == "binary_token_scale_u8":
                    codes, params = _encode_log_codes(self.packed.token_scale, 256)
                    arrays["token_scale_u8_codes"] = codes
                    arrays["token_scale_u8_params"] = params
                elif self.mode == "binary_token_scale_u4":
                    codes, params = _encode_log_codes(self.packed.token_scale, 16)
                    padded = np.concatenate([codes, np.zeros(len(codes) % 2, dtype=np.uint8)])
                    arrays["token_scale_u4_codes"] = (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)
                    arrays["token_scale_u4_params"] = params
                    arrays["token_scale_u4_count"] = np.array(len(codes), dtype=np.int64)
                else:
                    arrays["token_scale_fp16"] = np.ascontiguousarray(self.packed.token_scale, dtype=np.float16)
            if self.calibration is not None:
                arrays.update(
                    {
                        "centroid_thresholds": np.ascontiguousarray(self.calibration.thresholds, dtype=np.float32),
                        "centroid_negative": np.ascontiguousarray(self.calibration.negative_centroids, dtype=np.float32),
                        "centroid_positive": np.ascontiguousarray(self.calibration.positive_centroids, dtype=np.float32),
                    }
                )
        elif self.int4_packed is not None and self.mode == "int4_residual":
            if self.int4_packed.device != "cpu":
                raise ValueError("Corpus.save requires CPU residual int4 packed docs")
            arrays.update(
                {
                    "int4_data": np.ascontiguousarray(self.int4_packed.prefix_data, dtype=np.uint8),
                    "doc_offsets": np.ascontiguousarray(self.int4_packed.doc_offsets, dtype=np.int64),
                    "dim": np.array(self.int4_packed.dim, dtype=np.int64),
                    "int4_scale": np.array(self.int4_packed.prefix_scale, dtype=np.float32),
                    "int4_token_scale": np.ascontiguousarray(
                        self.int4_packed.prefix_token_scale, dtype=np.float32
                    ),
                    "residual_int4_data": np.ascontiguousarray(
                        self.int4_packed.residual_data, dtype=np.uint8
                    ),
                    "residual_int4_scale": np.array(
                        self.int4_packed.residual_scale, dtype=np.float32
                    ),
                    "residual_int4_token_scale": np.ascontiguousarray(
                        self.int4_packed.residual_token_scale, dtype=np.float32
                    ),
                }
            )
        elif self.int4_packed is not None:
            if self.int4_packed.device != "cpu":
                raise ValueError("Corpus.save requires CPU int4 packed docs")
            arrays.update(
                {
                    "int4_data": np.ascontiguousarray(self.int4_packed.data, dtype=np.uint8),
                    "doc_offsets": np.ascontiguousarray(self.int4_packed.doc_offsets, dtype=np.int64),
                    "dim": np.array(self.int4_packed.dim, dtype=np.int64),
                    "int4_scale": np.array(self.int4_packed.scale, dtype=np.float32),
                }
            )
            if self.int4_packed.token_scale is not None:
                arrays["int4_token_scale"] = np.ascontiguousarray(self.int4_packed.token_scale, dtype=np.float32)
        else:
            raise ValueError("corpus has no packed payload")
        with output.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
        object.__setattr__(self, "_source_path", output.resolve())

    @classmethod
    def load(cls, path) -> "Corpus":
        source_path = Path(path).resolve()
        with np.load(source_path, allow_pickle=False) as data:
            if int(np.asarray(data["schema_version"]).item()) not in {1, 2}:
                raise ValueError("unsupported corpus schema_version")
            if str(np.asarray(data["format"]).item()) != "maxsim_corpus":
                raise ValueError("unsupported corpus format")
            mode = str(np.asarray(data["mode"]).item())
            doc_ids = tuple(str(value) for value in np.asarray(data["doc_ids"]))
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            raw_offsets = np.asarray(data["doc_offsets"])
            if not np.issubdtype(raw_offsets.dtype, np.integer):
                raise ValueError("serialized doc_offsets must contain integers")
            offsets = np.ascontiguousarray(raw_offsets, dtype=np.int64)
            dim = int(np.asarray(data["dim"]).item())
            if mode in {"binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary", "binary_q40"}:
                calibration = None
                if mode == "binary_q40":
                    calibration = DimCentroidCalibration(
                        thresholds=np.ascontiguousarray(data["centroid_thresholds"], dtype=np.float32),
                        negative_centroids=np.ascontiguousarray(data["centroid_negative"], dtype=np.float32),
                        positive_centroids=np.ascontiguousarray(data["centroid_positive"], dtype=np.float32),
                    )
                token_scale = None
                if "token_scale_fp16" in data:
                    token_scale = np.ascontiguousarray(data["token_scale_fp16"]).astype(np.float32)
                elif "token_scale_u8_codes" in data:
                    token_scale = _decode_log_codes(np.asarray(data["token_scale_u8_codes"]), np.asarray(data["token_scale_u8_params"]), 256)
                elif "token_scale_u4_codes" in data:
                    packed_codes = np.asarray(data["token_scale_u4_codes"])
                    count = int(np.asarray(data["token_scale_u4_count"]).item())
                    codes = np.empty(packed_codes.shape[0] * 2, dtype=np.uint8)
                    codes[0::2] = packed_codes & 0x0F
                    codes[1::2] = packed_codes >> 4
                    token_scale = _decode_log_codes(codes[:count], np.asarray(data["token_scale_u4_params"]), 16)
                corpus = cls(
                    doc_ids=doc_ids,
                    mode=mode,
                    packed=PackedDocs(
                        data=_load_serialized_uint8(data, "packed_data"),
                        doc_offsets=offsets,
                        dim=dim,
                        num_docs=len(doc_ids),
                        scale=_decode_scale(str(np.asarray(data["scale_kind"]).item()), np.asarray(data["scale_values"])),
                        device="cpu",
                        token_scale=token_scale,
                    ),
                    calibration=calibration,
                    metadata=metadata,
                    _source_path=source_path,
                )
                _validate_corpus(corpus)
                return corpus
            if mode == "int4_residual":
                int4_data = _load_serialized_uint8(data, "int4_data")
                residual_data = _load_serialized_uint8(data, "residual_int4_data")
                _validate_serialized_offsets(
                    offsets, len(doc_ids), int4_data, dim, bytes_per_token=dim // 2
                )
                _validate_serialized_offsets(
                    offsets, len(doc_ids), residual_data, dim, bytes_per_token=dim // 2
                )
                packed = ResidualInt4PackedDocs(
                    prefix_data=int4_data,
                    prefix_values=_unpack_signed_int4_data(int4_data, dim),
                    prefix_scale=float(np.asarray(data["int4_scale"]).item()),
                    prefix_token_scale=_load_serialized_float32(
                        data,
                        "int4_token_scale",
                    ),
                    residual_data=residual_data,
                    residual_values=_unpack_signed_int4_data(residual_data, dim),
                    residual_scale=float(np.asarray(data["residual_int4_scale"]).item()),
                    residual_token_scale=_load_serialized_float32(
                        data,
                        "residual_int4_token_scale",
                    ),
                    doc_offsets=offsets,
                    dim=dim,
                    num_docs=len(doc_ids),
                    device="cpu",
                )
                corpus = cls(
                    doc_ids=doc_ids,
                    mode=mode,
                    int4_packed=packed,
                    metadata=metadata,
                    _source_path=source_path,
                )
                _validate_corpus(corpus)
                return corpus
            if mode in {"int4", "int4_per_token"}:
                int4_data = _load_serialized_uint8(data, "int4_data")
                token_scale = None
                if "int4_token_scale" in data:
                    token_scale = _load_serialized_float32(
                        data,
                        "int4_token_scale",
                    )
                _validate_serialized_offsets(offsets, len(doc_ids), int4_data, dim, bytes_per_token=dim // 2)
                values = _unpack_signed_int4_data(int4_data, dim)
                corpus = cls(
                    doc_ids=doc_ids,
                    mode=mode,
                    int4_packed=Int4PackedDocs(
                        data=int4_data,
                        values=values,
                        doc_offsets=offsets,
                        dim=dim,
                        num_docs=len(doc_ids),
                        scale=float(np.asarray(data["int4_scale"]).item()),
                        device="cpu",
                        token_scale=token_scale,
                    ),
                    metadata=metadata,
                    _source_path=source_path,
                )
                _validate_corpus(corpus)
                return corpus
            raise ValueError("unsupported corpus mode")

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def dim(self) -> int:
        if self.packed is not None:
            return int(self.packed.dim)
        if self.int4_packed is not None:
            return int(self.int4_packed.dim)
        raise ValueError("corpus has no packed payload")

    @property
    def encoded_bytes(self) -> int:
        """Logical persisted payload bytes used by compression-ratio reports."""
        if self.packed is not None:
            extra = 0 if self.calibration is None else int(self.calibration.metadata_bytes)
            if isinstance(self.packed.scale, np.ndarray):
                extra += int(self.packed.scale.astype(np.float32, copy=False).nbytes)
            elif self.packed.scale is not None:
                extra += 4
            if self.packed.token_scale is not None:
                count = int(self.packed.token_scale.shape[0])
                if self.mode == "binary_token_scale_u8":
                    extra += count + 16
                elif self.mode == "binary_token_scale_u4":
                    extra += (count + 1) // 2 + 16
                else:
                    extra += count * 2
            if isinstance(self.packed.data, np.ndarray):
                return int(self.packed.data.size + extra)
            return int(self.packed.doc_offsets[-1] * (self.packed.dim // 8) + extra)
        if self.int4_packed is not None:
            return int(self.int4_packed.storage_bytes)
        return 0

    @property
    def storage_bytes(self) -> int:
        """Backward-compatible alias for :attr:`encoded_bytes`."""
        return self.encoded_bytes

    def memory_report(self, serialized_path=None) -> CorpusMemoryReport:
        """Report logical size and current NumPy/CUDA residency separately.

        ``host_array_bytes`` intentionally measures owned array buffers rather
        than Python object overhead. CUDA workspace is ``None`` when the native
        handle does not expose an allocation counter.
        """
        _validate_corpus(self)
        source_path = self._source_path if serialized_path is None else Path(serialized_path)
        serialized_bytes = None if source_path is None else int(source_path.stat().st_size)
        return CorpusMemoryReport(
            encoded_bytes=self.encoded_bytes,
            host_array_bytes=_corpus_host_array_bytes(self),
            device_index_bytes=_corpus_device_index_bytes(self),
            device_workspace_bytes=_corpus_device_workspace_bytes(self),
            serialized_bytes=serialized_bytes,
        )


class Reranker:
    def __init__(self, corpus: Corpus, *, device: Literal["cpu", "cuda"] = "cpu", int4_query: Literal["fp32", "int8"] = "fp32"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if int4_query not in {"fp32", "int8"}:
            raise ValueError("int4_query must be 'fp32' or 'int8'")
        self.corpus = _move_corpus(corpus, device)
        self.device = device
        self.int4_query = int4_query
        self._doc_index = {doc_id: idx for idx, doc_id in enumerate(self.corpus.doc_ids)}

    @classmethod
    def from_corpus(cls, corpus: Corpus, *, device: Literal["cpu", "cuda"] = "cpu", int4_query: Literal["fp32", "int8"] = "fp32") -> "Reranker":
        return cls(corpus, device=device, int4_query=int4_query)

    @classmethod
    def load(
        cls,
        path,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        int4_query: Literal["fp32", "int8"] = "fp32",
    ) -> "Reranker":
        return cls.from_corpus(Corpus.load(path), device=device, int4_query=int4_query)

    def search(
        self,
        query_embeddings,
        *,
        k: int = 10,
        rescore_candidates: int | None = None,
        reducer: str = "maxsim",
        query_weights=None,
        temperature: float = 1.0,
    ):
        _validate_k(k, self.corpus.num_docs)
        actual_k = min(k, self.corpus.num_docs)
        if rescore_candidates is not None and self.corpus.mode != "int4_residual":
            raise ValueError("rescore_candidates requires corpus mode='int4_residual'")
        if self.corpus.mode in {"binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary"}:
            scores, indices = topk_maxsim(
                query_embeddings,
                self.corpus.packed,
                actual_k,
                device="auto",
                reducer=reducer,
                query_weights=query_weights,
                temperature=temperature,
            )
        elif self.corpus.mode == "binary_q40":
            _require_maxsim_reducer(self.corpus.mode, reducer, query_weights, temperature)
            scores, indices = topk_dim_centroid_maxsim(query_embeddings, self.corpus.packed, self.corpus.calibration, actual_k, device="auto")
        elif self.corpus.mode == "int4_residual":
            if rescore_candidates is None:
                fused_scores = residual_score(
                    query_embeddings,
                    self.corpus.int4_packed,
                    device="auto",
                    reducer=reducer,
                    query_weights=query_weights,
                    temperature=temperature,
                )
                scores, indices = _topk_score_arrays(fused_scores, actual_k)
            else:
                scores, indices = cascade_topk(
                    query_embeddings,
                    self.corpus.int4_packed,
                    actual_k,
                    candidates=rescore_candidates,
                    device="auto",
                    reducer=reducer,
                    query_weights=query_weights,
                    temperature=temperature,
                )
        elif self.corpus.mode in {"int4", "int4_per_token"}:
            scores, indices = topk_int4_maxsim(
                query_embeddings,
                self.corpus.int4_packed,
                actual_k,
                device="auto",
                prefer_int8_query=self.int4_query == "int8",
                reducer=reducer,
                query_weights=query_weights,
                temperature=temperature,
            )
        else:
            raise ValueError(f"unsupported corpus mode: {self.corpus.mode}")
        return _format_topk_results(scores, indices, self.corpus.doc_ids)

    def rerank(
        self,
        query_embeddings,
        candidate_ids,
        *,
        k: int = 10,
        reducer: str = "maxsim",
        query_weights=None,
        temperature: float = 1.0,
    ):
        candidates = _dedupe_candidate_ids(candidate_ids)
        candidate_indices = self._candidate_indices(candidates)
        if not candidates:
            return [] if _as_numpy(query_embeddings).ndim == 2 else []
        scores = self._score_candidates(
            query_embeddings,
            candidate_indices,
            reducer=reducer,
            query_weights=query_weights,
            temperature=temperature,
        )
        candidate_positions = np.arange(len(candidates), dtype=np.int64)
        return _format_candidate_results(scores, candidate_positions, candidates, min(k, len(candidates)))

    def _score_candidates(self, query_embeddings, candidate_indices, *, reducer, query_weights, temperature):
        if self.corpus.mode in {"binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary"}:
            return maxsim(
                query_embeddings,
                self.corpus.packed,
                device="auto",
                reducer=reducer,
                query_weights=query_weights,
                temperature=temperature,
                candidate_indices=candidate_indices,
            )
        if self.corpus.mode == "binary_q40":
            _require_maxsim_reducer(self.corpus.mode, reducer, query_weights, temperature)
            return dim_centroid_maxsim(
                query_embeddings,
                self.corpus.packed,
                self.corpus.calibration,
                device="auto",
                candidate_indices=candidate_indices,
            )
        if self.corpus.mode in {"int4", "int4_per_token"}:
            return int4_maxsim(
                query_embeddings,
                self.corpus.int4_packed,
                device="auto",
                reducer=reducer,
                query_weights=query_weights,
                temperature=temperature,
                candidate_indices=candidate_indices,
            )
        if self.corpus.mode == "int4_residual":
            return residual_score(
                query_embeddings,
                self.corpus.int4_packed,
                device="auto",
                reducer=reducer,
                query_weights=query_weights,
                temperature=temperature,
                candidate_indices=candidate_indices,
            )
        raise ValueError(f"unsupported corpus mode: {self.corpus.mode}")

    def _candidate_indices(self, candidate_ids: tuple[str, ...]) -> np.ndarray:
        indices = []
        for doc_id in candidate_ids:
            try:
                indices.append(self._doc_index[doc_id])
            except KeyError as exc:
                raise KeyError(doc_id) from exc
        return np.asarray(indices, dtype=np.int64)


def _normalize_doc_ids(doc_ids) -> tuple[str, ...]:
    values = tuple(str(doc_id) for doc_id in doc_ids)
    if not values:
        raise ValueError("doc_ids must contain at least one id")
    if len(set(values)) != len(values):
        raise ValueError("doc_ids must be unique")
    return values


def _normalize_sdk_offsets(offsets, num_tokens: int) -> np.ndarray:
    raw_values = _as_numpy(offsets)
    if not np.issubdtype(raw_values.dtype, np.integer):
        raise ValueError("offsets must contain integers")
    values = raw_values.astype(np.int64, copy=False)
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError("offsets must have shape [num_docs + 1]")
    if int(values[0]) != 0 or int(values[-1]) != num_tokens:
        raise ValueError("offsets must span embeddings")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("offsets must be monotonically nondecreasing")
    return values.copy()


def _validate_corpus_inputs(doc_ids: tuple[str, ...], docs: np.ndarray, offsets: np.ndarray, mode: str) -> None:
    if docs.ndim != 2:
        raise ValueError("embeddings must have shape [total_doc_tokens, dim]")
    if docs.shape[1] % 8 != 0:
        raise ValueError("embedding dim must be divisible by 8")
    if offsets.shape[0] != len(doc_ids) + 1:
        raise ValueError("offsets must have one more entry than doc_ids")
    if mode not in {
        "binary",
        "binary_token_scale",
        "binary_token_scale_u8",
        "binary_token_scale_u4",
        "pooled_binary",
        "binary_q40",
        "int4",
        "int4_per_token",
        "int4_residual",
    }:
        raise ValueError(
            "mode must be 'auto', one of 'binary', 'binary_token_scale', 'binary_token_scale_u8', "
            "'binary_token_scale_u4', 'pooled_binary', 'binary_q40', 'int4', "
            "'int4_per_token', 'int4_residual', "
            f"or a preset in {sorted(MODE_PRESETS)}"
        )


def _validate_corpus(corpus: Corpus) -> None:
    if not isinstance(corpus, Corpus):
        raise TypeError("corpus must be a Corpus instance")
    if not corpus.doc_ids:
        raise ValueError("corpus.doc_ids must contain at least one id")
    if any(not isinstance(doc_id, str) for doc_id in corpus.doc_ids):
        raise ValueError("corpus.doc_ids must contain strings")
    if len(set(corpus.doc_ids)) != len(corpus.doc_ids):
        raise ValueError("corpus.doc_ids must be unique")
    if corpus.metadata is not None and not isinstance(corpus.metadata, dict):
        raise ValueError("corpus.metadata must be a dictionary or None")

    binary_modes = {
        "binary",
        "binary_token_scale",
        "binary_token_scale_u8",
        "binary_token_scale_u4",
        "pooled_binary",
        "binary_q40",
    }
    int4_modes = {"int4", "int4_per_token"}
    if corpus.mode in binary_modes:
        if corpus.packed is None or corpus.int4_packed is not None:
            raise ValueError(f"mode {corpus.mode!r} requires exactly one binary packed payload")
        num_tokens = _validate_binary_corpus_payload(corpus.packed, len(corpus.doc_ids))
        requires_token_scale = corpus.mode in {"binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4"}
        if requires_token_scale != (corpus.packed.token_scale is not None):
            requirement = "requires" if requires_token_scale else "does not accept"
            raise ValueError(f"mode {corpus.mode!r} {requirement} binary token scales")
        _validate_token_scale(corpus.packed.token_scale, num_tokens, "packed.token_scale")
        if corpus.mode == "binary_q40":
            if corpus.calibration is None:
                raise ValueError("binary_q40 mode requires centroid calibration")
            _validate_calibration(corpus.calibration, corpus.packed.dim)
        elif corpus.calibration is not None:
            raise ValueError("centroid calibration is only valid for binary_q40 mode")
        return
    if corpus.mode in int4_modes:
        if corpus.int4_packed is None or corpus.packed is not None:
            raise ValueError(f"mode {corpus.mode!r} requires exactly one int4 packed payload")
        if corpus.calibration is not None:
            raise ValueError("centroid calibration is not valid for int4 modes")
        num_tokens = _validate_int4_corpus_payload(corpus.int4_packed, len(corpus.doc_ids))
        token_scale = getattr(corpus.int4_packed, "token_scale", None)
        requires_token_scale = corpus.mode == "int4_per_token"
        if requires_token_scale != (token_scale is not None):
            requirement = "requires" if requires_token_scale else "does not accept"
            raise ValueError(f"mode {corpus.mode!r} {requirement} int4 token scales")
        _validate_token_scale(
            token_scale,
            num_tokens,
            "int4_packed.token_scale",
            allow_zero=False,
        )
        return
    if corpus.mode == "int4_residual":
        if corpus.int4_packed is None or corpus.packed is not None:
            raise ValueError("mode 'int4_residual' requires exactly one residual int4 payload")
        if corpus.calibration is not None:
            raise ValueError("centroid calibration is not valid for int4_residual")
        _validate_residual_packed(
            corpus.int4_packed,
            verify_packed_values=True,
        )
        if corpus.int4_packed.num_docs != len(corpus.doc_ids):
            raise ValueError("residual int4 num_docs must match corpus.doc_ids")
        return
    raise ValueError(f"unsupported corpus mode: {corpus.mode}")


def _validate_binary_corpus_payload(packed: PackedDocs, num_docs: int) -> int:
    if not isinstance(packed, PackedDocs):
        raise ValueError("corpus.packed must be a PackedDocs instance")
    if packed.dim <= 0 or packed.dim % 8 != 0:
        raise ValueError("packed.dim must be positive and divisible by 8")
    if packed.num_docs != num_docs:
        raise ValueError("packed.num_docs must match corpus.doc_ids")
    if packed.device == "cpu":
        if not isinstance(packed.data, np.ndarray) or packed.data.dtype != np.uint8:
            raise ValueError("CPU packed.data must be a uint8 NumPy array")
        if packed.data.ndim != 2 or packed.data.shape[1] != packed.dim // 8:
            raise ValueError("packed.data must have shape [num_doc_tokens, dim / 8]")
        num_tokens = int(packed.data.shape[0])
    elif packed.device == "cuda":
        handle_dim = _read_size_accessor(packed.data, "dim")
        handle_docs = _read_size_accessor(packed.data, "num_docs")
        if handle_dim is not None and handle_dim != packed.dim:
            raise ValueError("CUDA packed handle dim does not match packed.dim")
        if handle_docs is not None and handle_docs != packed.num_docs:
            raise ValueError("CUDA packed handle num_docs does not match packed.num_docs")
        packed_size = _read_size_accessor(packed.data, "packed_size")
        if packed_size is None:
            num_tokens = _offset_terminal(packed.doc_offsets)
        else:
            byte_dim = packed.dim // 8
            if packed_size % byte_dim != 0:
                raise ValueError("CUDA packed handle size does not match packed.dim")
            num_tokens = packed_size // byte_dim
    else:
        raise ValueError("packed.device must be 'cpu' or 'cuda'")
    _validate_offset_values(packed.doc_offsets, num_docs, num_tokens)
    _validate_scale(packed.scale, num_docs, "packed.scale")
    return num_tokens


def _validate_int4_corpus_payload(packed, num_docs: int) -> int:
    if not isinstance(packed, Int4PackedDocs):
        raise ValueError("corpus.int4_packed must be an Int4PackedDocs instance")
    _validate_int4_packed(packed)
    dim = int(getattr(packed, "dim", 0))
    if dim <= 0 or dim % 8 != 0:
        raise ValueError("int4_packed.dim must be positive and divisible by 8")
    if int(getattr(packed, "num_docs", -1)) != num_docs:
        raise ValueError("int4_packed.num_docs must match corpus.doc_ids")
    device = getattr(packed, "device", None)
    if device == "cpu":
        data = getattr(packed, "data", None)
        values = getattr(packed, "values", None)
        if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
            raise ValueError("CPU int4_packed.data must be a uint8 NumPy array")
        if data.ndim != 2 or data.shape[1] != dim // 2:
            raise ValueError("int4_packed.data must have shape [num_doc_tokens, dim / 2]")
        num_tokens = int(data.shape[0])
        if not isinstance(values, np.ndarray) or values.dtype != np.int8:
            raise ValueError("CPU int4_packed.values must be an int8 NumPy array")
        if values.shape != (num_tokens, dim):
            raise ValueError("int4_packed.values must have shape [num_doc_tokens, dim]")
        if np.any(values < -7) or np.any(values > 7):
            raise ValueError("int4_packed.values must contain symmetric int4 values in [-7, 7]")
        _validate_int4_data_matches_values(packed)
    elif device == "cuda":
        handle = getattr(packed, "data", None)
        handle_dim = _read_size_accessor(handle, "dim")
        handle_docs = _read_size_accessor(handle, "num_docs")
        if handle_dim is not None and handle_dim != dim:
            raise ValueError("CUDA int4 handle dim does not match int4_packed.dim")
        if handle_docs is not None and handle_docs != num_docs:
            raise ValueError("CUDA int4 handle num_docs does not match int4_packed.num_docs")
        packed_size = _read_size_accessor(handle, "packed_size")
        if packed_size is None:
            num_tokens = _offset_terminal(getattr(packed, "doc_offsets", None))
        else:
            byte_dim = dim // 2
            if packed_size % byte_dim != 0:
                raise ValueError("CUDA int4 handle size does not match int4_packed.dim")
            num_tokens = packed_size // byte_dim
        values = getattr(packed, "values", None)
        if values is not None and (
            not isinstance(values, np.ndarray) or values.dtype != np.int8 or values.shape != (num_tokens, dim)
        ):
            raise ValueError("int4_packed.values must have shape [num_doc_tokens, dim] when present")
    else:
        raise ValueError("int4_packed.device must be 'cpu' or 'cuda'")
    _validate_offset_values(getattr(packed, "doc_offsets", None), num_docs, num_tokens)
    scale = float(getattr(packed, "scale", float("nan")))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("int4_packed.scale must be finite and > 0")
    return num_tokens


def _validate_offset_values(offsets, num_docs: int, num_tokens: int) -> None:
    if not isinstance(offsets, np.ndarray) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("doc_offsets must be an integer NumPy array")
    if offsets.ndim != 1 or offsets.shape[0] != num_docs + 1:
        raise ValueError("doc_offsets must have shape [num_docs + 1]")
    if int(offsets[0]) != 0:
        raise ValueError("doc_offsets must start at 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("doc_offsets must be monotonically nondecreasing")
    if int(offsets[-1]) != num_tokens:
        raise ValueError("doc_offsets must end at num_doc_tokens")


def _offset_terminal(offsets) -> int:
    if not isinstance(offsets, np.ndarray) or offsets.ndim != 1 or offsets.shape[0] < 2:
        raise ValueError("doc_offsets must have shape [num_docs + 1]")
    return int(offsets[-1])


def _validate_scale(scale, num_docs: int, name: str) -> None:
    if scale is None:
        return
    if isinstance(scale, np.ndarray):
        if scale.ndim != 1 or scale.shape[0] != num_docs:
            raise ValueError(f"{name} must have shape [num_docs]")
        if not np.all(np.isfinite(scale)):
            raise ValueError(f"{name} must contain finite values")
        return
    if not np.isfinite(float(scale)):
        raise ValueError(f"{name} must be finite")


def _validate_token_scale(
    scale,
    num_tokens: int,
    name: str,
    *,
    allow_zero: bool = True,
) -> None:
    if scale is None:
        return
    if not isinstance(scale, np.ndarray) or scale.ndim != 1 or scale.shape[0] != num_tokens:
        raise ValueError(f"{name} must have shape [num_doc_tokens]")
    if not np.issubdtype(scale.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    lower_bound_invalid = scale < 0.0 if allow_zero else scale <= 0.0
    if not np.all(np.isfinite(scale)) or np.any(lower_bound_invalid):
        constraint = "nonnegative" if allow_zero else "> 0"
        raise ValueError(f"{name} must contain finite values {constraint}")


def _validate_calibration(calibration, dim: int) -> None:
    for name in ("thresholds", "negative_centroids", "positive_centroids"):
        values = getattr(calibration, name, None)
        if not isinstance(values, np.ndarray) or values.ndim != 1 or values.shape[0] != dim:
            raise ValueError(f"calibration.{name} must have shape [dim]")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"calibration.{name} must contain finite values")


def _validate_serialized_offsets(offsets, num_docs: int, data: np.ndarray, dim: int, *, bytes_per_token: int) -> None:
    if dim <= 0 or dim % 8 != 0:
        raise ValueError("serialized dim must be positive and divisible by 8")
    if data.ndim != 2 or data.shape[1] != bytes_per_token:
        raise ValueError("serialized packed data shape does not match dim")
    _validate_offset_values(offsets, num_docs, int(data.shape[0]))


def _load_serialized_uint8(data, name: str) -> np.ndarray:
    values = np.asarray(data[name])
    if values.dtype != np.uint8:
        raise ValueError(f"serialized {name} must have dtype uint8")
    return np.ascontiguousarray(values, dtype=np.uint8)


def _load_serialized_float32(data, name: str) -> np.ndarray:
    values = np.asarray(data[name])
    if values.dtype != np.float32:
        raise ValueError(f"serialized {name} must have dtype float32")
    return np.ascontiguousarray(values, dtype=np.float32)


def _corpus_host_array_bytes(corpus: Corpus) -> int:
    arrays: list[np.ndarray] = []
    if corpus.packed is not None:
        if isinstance(corpus.packed.data, np.ndarray):
            arrays.append(corpus.packed.data)
        arrays.append(corpus.packed.doc_offsets)
        if isinstance(corpus.packed.scale, np.ndarray):
            arrays.append(corpus.packed.scale)
        if isinstance(corpus.packed.token_scale, np.ndarray):
            arrays.append(corpus.packed.token_scale)
    if corpus.int4_packed is not None:
        names = (
            "prefix_data",
            "prefix_values",
            "prefix_token_scale",
            "residual_data",
            "residual_values",
            "residual_token_scale",
            "doc_offsets",
        ) if corpus.mode == "int4_residual" else (
            "data",
            "values",
            "doc_offsets",
            "token_scale",
        )
        for name in names:
            values = getattr(corpus.int4_packed, name, None)
            if isinstance(values, np.ndarray):
                arrays.append(values)
    if corpus.calibration is not None:
        for name in ("thresholds", "negative_centroids", "positive_centroids"):
            values = getattr(corpus.calibration, name, None)
            if isinstance(values, np.ndarray):
                arrays.append(values)
    return int(sum(values.nbytes for values in arrays))


def _corpus_device_index_bytes(corpus: Corpus) -> int:
    payload = corpus.packed if corpus.packed is not None else corpus.int4_packed
    if payload is None or getattr(payload, "device", "cpu") != "cuda":
        return 0
    handle = payload.prefix_data if corpus.mode == "int4_residual" else payload.data
    explicit = _read_size_accessor(handle, "device_index_bytes", "index_bytes", "resident_bytes")
    if explicit is not None:
        return explicit
    packed_bytes = _read_size_accessor(handle, "packed_size")
    if packed_bytes is None:
        packed_bytes = int(payload.doc_offsets[-1]) * (int(payload.dim) // (8 if corpus.packed is not None else 2))
    if corpus.mode == "int4_residual":
        token_count = int(payload.doc_offsets[-1])
        return int(2 * packed_bytes + payload.doc_offsets.nbytes + 8 * token_count)
    total = packed_bytes + int(payload.doc_offsets.nbytes)
    scale_count = _read_size_accessor(handle, "scale_vector_size") or 0
    token_scale_count = _read_size_accessor(handle, "token_scale_vector_size")
    if token_scale_count is None:
        token_scale = getattr(payload, "token_scale", None)
        token_scale_count = 0 if token_scale is None else int(token_scale.shape[0])
    return int(total + 4 * scale_count + 4 * token_scale_count)


def _corpus_device_workspace_bytes(corpus: Corpus) -> int | None:
    payload = corpus.packed if corpus.packed is not None else corpus.int4_packed
    if payload is None or getattr(payload, "device", "cpu") != "cuda":
        return 0
    handle = payload.prefix_data if corpus.mode == "int4_residual" else payload.data
    return _read_size_accessor(handle, "device_workspace_bytes", "workspace_bytes")


def _read_size_accessor(value, *names: str) -> int | None:
    for name in names:
        try:
            result = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if callable(result):
            result = result()
        result = int(result)
        if result < 0:
            raise ValueError(f"{name} must be nonnegative")
        return result
    return None


def _move_corpus(corpus: Corpus, device: str) -> Corpus:
    _validate_corpus(corpus)
    if device == "cpu":
        return corpus
    if corpus.packed is not None:
        return Corpus(
            doc_ids=corpus.doc_ids,
            mode=corpus.mode,
            packed=to_device(corpus.packed, "cuda"),
            int4_packed=corpus.int4_packed,
            calibration=corpus.calibration,
            metadata=corpus.metadata,
            _source_path=corpus._source_path,
        )
    if corpus.mode == "int4_residual":
        return Corpus(
            doc_ids=corpus.doc_ids,
            mode=corpus.mode,
            packed=corpus.packed,
            int4_packed=residual_int4_to_device(corpus.int4_packed),
            calibration=corpus.calibration,
            metadata=corpus.metadata,
            _source_path=corpus._source_path,
        )
    if corpus.int4_packed is not None:
        return Corpus(
            doc_ids=corpus.doc_ids,
            mode=corpus.mode,
            packed=corpus.packed,
            int4_packed=int4_to_device(corpus.int4_packed),
            calibration=corpus.calibration,
            metadata=corpus.metadata,
            _source_path=corpus._source_path,
        )
    return corpus


def _encode_scale(scale) -> tuple[str, np.ndarray]:
    if scale is None:
        return "none", np.empty((0,), dtype=np.float32)
    if isinstance(scale, np.ndarray):
        return "vector", np.ascontiguousarray(scale, dtype=np.float32)
    return "scalar", np.array([float(scale)], dtype=np.float32)


def _decode_scale(kind: str, values: np.ndarray):
    if kind == "none":
        return None
    if kind == "scalar":
        return float(values.reshape(-1)[0])
    if kind == "vector":
        return np.ascontiguousarray(values, dtype=np.float32)
    raise ValueError(f"unknown scale kind: {kind}")


def _encode_log_codes(values: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    logs = np.log(np.maximum(values.astype(np.float64), 1e-12))
    lo = float(logs.min())
    hi = float(logs.max())
    if hi <= lo:
        return np.zeros(values.shape[0], dtype=np.uint8), np.array([lo, lo], dtype=np.float64)
    steps = float(levels - 1)
    codes = np.clip(np.rint((logs - lo) * (steps / (hi - lo))), 0, steps).astype(np.uint8)
    return codes, np.array([lo, hi], dtype=np.float64)


def _decode_log_codes(codes: np.ndarray, params: np.ndarray, levels: int) -> np.ndarray:
    lo = float(params[0])
    hi = float(params[1])
    if hi <= lo:
        return np.full(codes.shape[0], np.exp(lo), dtype=np.float32)
    return np.exp(lo + codes.astype(np.float64) * ((hi - lo) / float(levels - 1))).astype(np.float32)


def _unpack_signed_int4_data(data: np.ndarray, dim: int) -> np.ndarray:
    values = np.empty((data.shape[0], dim), dtype=np.int8)
    low = (data & 0x0F).astype(np.int8)
    high = ((data >> 4) & 0x0F).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    values[:, 0::2] = low
    values[:, 1::2] = high
    return np.ascontiguousarray(values, dtype=np.int8)


def _validate_k(k: int, max_count: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_count < 1:
        raise ValueError("corpus must contain at least one document")


def _require_maxsim_reducer(mode: str, reducer, query_weights, temperature) -> None:
    if reducer != "maxsim":
        raise ValueError(f"corpus mode '{mode}' currently supports only reducer='maxsim'")
    if query_weights is not None:
        raise ValueError("query_weights is only supported with reducer='weighted_maxsim'")
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a finite value greater than 0") from exc
    if not np.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be a finite value greater than 0")


def _dedupe_candidate_ids(candidate_ids) -> tuple[str, ...]:
    seen = set()
    result = []
    for candidate_id in candidate_ids:
        value = str(candidate_id)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _topk_score_arrays(scores, k: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim not in {1, 2}:
        raise ValueError("scores must have shape [num_docs] or [batch, num_docs]")
    return _stable_topk(values, k)


def _format_topk_results(scores, indices, doc_ids: tuple[str, ...]):
    score_values = np.asarray(scores, dtype=np.float32)
    index_values = np.asarray(indices, dtype=np.int64)
    if score_values.ndim == 1:
        return _format_result_row(score_values, index_values, doc_ids)
    return [_format_result_row(row_scores, row_indices, doc_ids) for row_scores, row_indices in zip(score_values, index_values)]


def _format_candidate_results(scores, candidate_indices: np.ndarray, candidate_ids: tuple[str, ...], k: int):
    score_values = np.asarray(scores, dtype=np.float32)
    if score_values.ndim == 1:
        return _format_candidate_row(score_values, candidate_indices, candidate_ids, k)
    return [_format_candidate_row(row, candidate_indices, candidate_ids, k) for row in score_values]


def _format_candidate_row(scores: np.ndarray, candidate_indices: np.ndarray, candidate_ids: tuple[str, ...], k: int):
    candidate_scores = scores[candidate_indices]
    order = np.lexsort((np.arange(candidate_scores.shape[0], dtype=np.int64), -candidate_scores))[:k]
    return [
        SearchResult(doc_id=candidate_ids[int(position)], score=float(candidate_scores[int(position)]), rank=rank)
        for rank, position in enumerate(order, start=1)
    ]


def _format_result_row(scores: np.ndarray, indices: np.ndarray, doc_ids: tuple[str, ...]):
    return [
        SearchResult(doc_id=doc_ids[int(index)], score=float(score), rank=rank)
        for rank, (score, index) in enumerate(zip(scores, indices), start=1)
    ]


# The primary product name for a packed corpus; Corpus remains as an alias
# used throughout the benchmark harness.
Index = Corpus
