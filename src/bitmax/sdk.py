from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bitmax._api import PackedDocs, _as_numpy, maxsim, pack_signs, to_device, topk_maxsim
from bitmax.experimental import (
    DimCentroidCalibration,
    dim_centroid_maxsim,
    fit_dim_centroid_calibration,
    int4_maxsim,
    int4_to_device,
    pack_dim_centroid_signs,
    pack_int4_symmetric,
    topk_dim_centroid_maxsim,
    topk_int4_maxsim,
)

CorpusMode = Literal["binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary", "binary_q40", "int4"]

# Named accuracy/size/latency tiers over the measured pareto frontier
# (docs/gpu_optimization.md, 2026-07-02/03). "balanced" is the default.
# max_compression trades NDCG@10 ~-0.018 vs dense for ~64x compression via
# token pooling (requires scipy); "compact" keeps near-dense quality at ~30x.
MODE_PRESETS = {
    "balanced": "binary_token_scale",
    "max_quality": "int4",
    "max_compression": "pooled_binary",
    "compact": "binary_token_scale_u4",
    "max_speed": "binary",
}


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

    @classmethod
    def from_embeddings(
        cls,
        doc_ids,
        embeddings,
        offsets,
        *,
        mode: str = "balanced",
        pool_factor: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> "Corpus":
        mode = MODE_PRESETS.get(mode, mode)
        doc_id_values = _normalize_doc_ids(doc_ids)
        docs = _as_numpy(embeddings).astype(np.float32, copy=False)
        offsets_array = _normalize_sdk_offsets(offsets, docs.shape[0])
        _validate_corpus_inputs(doc_id_values, docs, offsets_array, mode)
        if mode == "pooled_binary":
            from bitmax.pooling import pool_doc_tokens

            original_tokens = int(docs.shape[0])
            docs, offsets_array = pool_doc_tokens(docs, offsets_array, pool_factor)
            pool_metadata = {
                "pool_factor": int(pool_factor),
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
        raise ValueError(
            "mode must be one of 'binary', 'binary_token_scale', 'binary_token_scale_u8', 'pooled_binary', 'binary_q40', 'int4', "
            f"or a preset in {sorted(MODE_PRESETS)}"
        )

    def save(self, path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, object] = {
            "schema_version": np.array(1, dtype=np.int64),
            "format": np.array("bitmax_corpus"),
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
        else:
            raise ValueError("corpus has no packed payload")
        np.savez_compressed(output, **arrays)

    @classmethod
    def load(cls, path) -> "Corpus":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(np.asarray(data["schema_version"]).item()) != 1:
                raise ValueError("unsupported corpus schema_version")
            if str(np.asarray(data["format"]).item()) != "bitmax_corpus":
                raise ValueError("unsupported corpus format")
            mode = str(np.asarray(data["mode"]).item())
            doc_ids = tuple(str(value) for value in np.asarray(data["doc_ids"]))
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            offsets = np.ascontiguousarray(data["doc_offsets"], dtype=np.int64)
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
                return cls(
                    doc_ids=doc_ids,
                    mode=mode,
                    packed=PackedDocs(
                        data=np.ascontiguousarray(data["packed_data"], dtype=np.uint8),
                        doc_offsets=offsets,
                        dim=dim,
                        num_docs=len(doc_ids),
                        scale=_decode_scale(str(np.asarray(data["scale_kind"]).item()), np.asarray(data["scale_values"])),
                        device="cpu",
                        token_scale=token_scale,
                    ),
                    calibration=calibration,
                    metadata=metadata,
                )
            if mode == "int4":
                from bitmax.experimental import Int4PackedDocs

                int4_data = np.ascontiguousarray(data["int4_data"], dtype=np.uint8)
                values = _unpack_signed_int4_data(int4_data, dim)
                return cls(
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
                    ),
                    metadata=metadata,
                )
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
    def storage_bytes(self) -> int:
        if self.packed is not None:
            extra = 0 if self.calibration is None else int(self.calibration.metadata_bytes)
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
            return int(self.int4_packed.storage_bytes + 4)
        return 0


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
    def load(cls, path, *, device: Literal["cpu", "cuda"] = "cpu") -> "Reranker":
        return cls.from_corpus(Corpus.load(path), device=device)

    def search(self, query_embeddings, *, k: int = 10):
        _validate_k(k, self.corpus.num_docs)
        actual_k = min(k, self.corpus.num_docs)
        if self.corpus.mode in {"binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary"}:
            scores, indices = topk_maxsim(query_embeddings, self.corpus.packed, actual_k, device="auto")
        elif self.corpus.mode == "binary_q40":
            scores, indices = topk_dim_centroid_maxsim(query_embeddings, self.corpus.packed, self.corpus.calibration, actual_k, device="auto")
        elif self.corpus.mode == "int4":
            scores, indices = topk_int4_maxsim(
                query_embeddings, self.corpus.int4_packed, actual_k, device="auto", prefer_int8_query=self.int4_query == "int8"
            )
        else:
            raise ValueError(f"unsupported corpus mode: {self.corpus.mode}")
        return _format_topk_results(scores, indices, self.corpus.doc_ids)

    def rerank(self, query_embeddings, candidate_ids, *, k: int = 10):
        candidates = _dedupe_candidate_ids(candidate_ids)
        candidate_indices = self._candidate_indices(candidates)
        if not candidates:
            return [] if _as_numpy(query_embeddings).ndim == 2 else []
        scores = self._score_all(query_embeddings)
        return _format_candidate_results(scores, candidate_indices, candidates, min(k, len(candidates)))

    def _score_all(self, query_embeddings):
        if self.corpus.mode in {"binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary"}:
            return maxsim(query_embeddings, self.corpus.packed, device="auto")
        if self.corpus.mode == "binary_q40":
            return dim_centroid_maxsim(query_embeddings, self.corpus.packed, self.corpus.calibration, device="auto")
        if self.corpus.mode == "int4":
            return int4_maxsim(query_embeddings, self.corpus.int4_packed, device="auto")
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
    values = _as_numpy(offsets).astype(np.int64, copy=False)
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
    if mode not in {"binary", "binary_token_scale", "binary_token_scale_u8", "binary_token_scale_u4", "pooled_binary", "binary_q40", "int4"}:
        raise ValueError(
            "mode must be one of 'binary', 'binary_token_scale', 'binary_token_scale_u8', 'pooled_binary', 'binary_q40', 'int4', "
            f"or a preset in {sorted(MODE_PRESETS)}"
        )


def _move_corpus(corpus: Corpus, device: str) -> Corpus:
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
        )
    if corpus.int4_packed is not None:
        return Corpus(
            doc_ids=corpus.doc_ids,
            mode=corpus.mode,
            packed=corpus.packed,
            int4_packed=int4_to_device(corpus.int4_packed),
            calibration=corpus.calibration,
            metadata=corpus.metadata,
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


def _dedupe_candidate_ids(candidate_ids) -> tuple[str, ...]:
    seen = set()
    result = []
    for candidate_id in candidate_ids:
        value = str(candidate_id)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


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
