from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from bitmax._api import PackedDocs, _as_numpy, maxsim, pack_signs, to_device, topk_maxsim

CorpusMode = Literal["binary", "binary_q40", "int4"]


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
        mode: CorpusMode = "binary",
        metadata: dict[str, Any] | None = None,
    ) -> "Corpus":
        doc_id_values = _normalize_doc_ids(doc_ids)
        docs = _as_numpy(embeddings).astype(np.float32, copy=False)
        offsets_array = _normalize_sdk_offsets(offsets, docs.shape[0])
        _validate_corpus_inputs(doc_id_values, docs, offsets_array, mode)
        if mode == "binary":
            return cls(
                doc_ids=doc_id_values,
                mode=mode,
                packed=pack_signs(docs, offsets_array),
                metadata={} if metadata is None else dict(metadata),
            )
        raise NotImplementedError(f"mode={mode!r} is implemented in Task 2")

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
            if isinstance(self.packed.data, np.ndarray):
                return int(self.packed.data.size + extra)
            return int(self.packed.doc_offsets[-1] * (self.packed.dim // 8) + extra)
        if self.int4_packed is not None:
            return int(self.int4_packed.storage_bytes + 4)
        return 0


class Reranker:
    def __init__(self, corpus: Corpus, *, device: Literal["cpu", "cuda"] = "cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.corpus = _move_corpus(corpus, device)
        self.device = device
        self._doc_index = {doc_id: idx for idx, doc_id in enumerate(self.corpus.doc_ids)}

    @classmethod
    def from_corpus(cls, corpus: Corpus, *, device: Literal["cpu", "cuda"] = "cpu") -> "Reranker":
        return cls(corpus, device=device)

    def search(self, query_embeddings, *, k: int = 10):
        _validate_k(k, self.corpus.num_docs)
        if self.corpus.packed is None:
            raise ValueError("binary search requires packed docs")
        scores, indices = topk_maxsim(
            query_embeddings,
            self.corpus.packed,
            min(k, self.corpus.num_docs),
            device="auto",
        )
        return _format_topk_results(scores, indices, self.corpus.doc_ids)

    def rerank(self, query_embeddings, candidate_ids, *, k: int = 10):
        candidates = _dedupe_candidate_ids(candidate_ids)
        candidate_indices = self._candidate_indices(candidates)
        if not candidates:
            return [] if _as_numpy(query_embeddings).ndim == 2 else []
        if self.corpus.packed is None:
            raise ValueError("binary rerank requires packed docs")
        scores = maxsim(query_embeddings, self.corpus.packed, device="auto")
        return _format_candidate_results(scores, candidate_indices, candidates, min(k, len(candidates)))

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
    if mode not in {"binary", "binary_q40", "int4"}:
        raise ValueError("mode must be 'binary', 'binary_q40', or 'int4'")


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
    return corpus


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
