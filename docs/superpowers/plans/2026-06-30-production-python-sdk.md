# Production Python SDK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished Python SDK for CUDA-first compressed multi-vector search and reranking.

**Architecture:** Add high-level `Corpus`, `Reranker`, and `SearchResult` APIs above the existing binary, centroid-binary, and int4 kernels. Persist corpus files with doc IDs and mode metadata. Add a CUDA-focused demo that compares dense fp16 CUDA against bitmax binary, q40 centroid binary, and int4 on the same embedding slice.

**Tech Stack:** Python, NumPy, pybind11/CUDA extension, PyTorch CUDA for dense baseline demo, pytest, VAST RTX 4090 worker.

---

## File Structure

- Create `src/bitmax/sdk.py`: high-level SDK dataclasses, corpus packing/loading, runtime search/rerank.
- Modify `src/bitmax/__init__.py`: export `Corpus`, `Reranker`, and `SearchResult`.
- Create `tests/test_sdk.py`: local correctness tests for API shape, validation, persistence, search, rerank, modes.
- Modify `tests/test_cuda_extension.py`: CUDA SDK correctness tests for binary, q40, and int4 modes.
- Create `examples/local_multivector_search.py`: CUDA-first SDK demo over retrieval embedding `.npz` files.
- Create `tests/test_sdk_demo.py`: tiny fixture test for demo JSON/table behavior.
- Modify `pyproject.toml`: include `examples` in editable/wheel packages if using `python -m examples.local_multivector_search`.
- Modify `README.md`, `docs/api.md`, `docs/benchmarks.md`: document SDK-first usage and CUDA demo proof.

## Task 1: Public SDK API Correctness Tests

**Files:**
- Create: `tests/test_sdk.py`
- Create after red test: `src/bitmax/sdk.py`
- Modify after red test: `src/bitmax/__init__.py`

- [ ] **Step 1: Write failing SDK API tests**

Create `tests/test_sdk.py` with:

```python
import numpy as np
import pytest

import bitmax


def _tiny_multivector_docs():
    docs = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    return docs, offsets


def test_corpus_from_embeddings_exposes_metadata_and_storage():
    docs, offsets = _tiny_multivector_docs()

    corpus = bitmax.Corpus.from_embeddings(
        doc_ids=["positive", "negative", "mixed"],
        embeddings=docs,
        offsets=offsets,
        mode="binary",
        metadata={"model": "tiny"},
    )

    assert corpus.doc_ids == ("positive", "negative", "mixed")
    assert corpus.num_docs == 3
    assert corpus.dim == 8
    assert corpus.mode == "binary"
    assert corpus.metadata == {"model": "tiny"}
    assert corpus.storage_bytes == 3


def test_reranker_search_returns_ranked_results_for_single_query():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.search(query, k=2)

    assert [result.doc_id for result in results] == ["positive", "mixed"]
    assert [result.rank for result in results] == [1, 2]
    assert all(isinstance(result.score, float) for result in results)


def test_reranker_search_returns_batch_results_for_batched_queries():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    query = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )

    results = reranker.search(query, k=1)

    assert [[result.doc_id for result in row] for row in results] == [["positive"], ["negative"]]
    assert [[result.rank for result in row] for row in results] == [[1], [1]]


def test_reranker_rerank_returns_only_candidates_and_collapses_duplicates():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    query = np.ones((1, 8), dtype=np.float32)
    results = reranker.rerank(query, candidate_ids=["negative", "mixed", "mixed"], k=3)

    assert [result.doc_id for result in results] == ["mixed", "negative"]
    assert [result.rank for result in results] == [1, 2]


def test_reranker_rejects_unknown_candidate_id():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    reranker = bitmax.Reranker.from_corpus(corpus)

    with pytest.raises(KeyError, match="missing"):
        reranker.rerank(np.ones((1, 8), dtype=np.float32), candidate_ids=["missing"], k=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_sdk.py
```

Expected: tests fail because `bitmax.Corpus` is not exported.

- [ ] **Step 3: Implement minimal SDK API**

Create `src/bitmax/sdk.py` with:

```python
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
            return int(np.asarray(self.packed.data).size + extra)
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
        scores, indices = topk_maxsim(query_embeddings, self.corpus.packed, min(k, self.corpus.num_docs), device="auto")
        return _format_topk_results(scores, indices, self.corpus.doc_ids)

    def rerank(self, query_embeddings, candidate_ids, *, k: int = 10):
        candidates = _dedupe_candidate_ids(candidate_ids)
        candidate_indices = self._candidate_indices(candidates)
        if not candidates:
            return [] if _as_numpy(query_embeddings).ndim == 2 else []
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
```

Modify `src/bitmax/__init__.py`:

```python
from bitmax._api import PackedDocs, maxsim, pack_signs, to_device, topk_maxsim
from bitmax.io import PackedBundle, load_packed, save_packed
from bitmax.sdk import Corpus, Reranker, SearchResult

__all__ = [
    "PackedDocs",
    "PackedBundle",
    "Corpus",
    "Reranker",
    "SearchResult",
    "pack_signs",
    "to_device",
    "maxsim",
    "topk_maxsim",
    "save_packed",
    "load_packed",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_sdk.py
```

Expected: all `tests/test_sdk.py` tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/bitmax/sdk.py src/bitmax/__init__.py tests/test_sdk.py
git commit -m "feat: add sdk corpus and reranker"
```

## Task 2: SDK Validation, Modes, and Persistence

**Files:**
- Modify: `tests/test_sdk.py`
- Modify: `src/bitmax/sdk.py`

- [ ] **Step 1: Add failing tests for validation, q40, int4, and persistence**

Append to `tests/test_sdk.py`:

```python
def test_corpus_rejects_invalid_doc_ids_offsets_and_mode():
    docs, offsets = _tiny_multivector_docs()

    with pytest.raises(ValueError, match="unique"):
        bitmax.Corpus.from_embeddings(["dup", "dup", "mixed"], docs, offsets, mode="binary")
    with pytest.raises(ValueError, match="one more entry"):
        bitmax.Corpus.from_embeddings(["one", "two"], docs, offsets, mode="binary")
    with pytest.raises(ValueError, match="mode"):
        bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="bad")


def test_binary_q40_mode_searches_through_sdk():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary_q40")
    reranker = bitmax.Reranker.from_corpus(corpus)

    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=2)

    assert corpus.mode == "binary_q40"
    assert corpus.storage_bytes > 3
    assert len(results) == 2
    assert results[0].doc_id == "positive"


def test_int4_mode_searches_through_sdk():
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="int4")
    reranker = bitmax.Reranker.from_corpus(corpus)

    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=2)

    assert corpus.mode == "int4"
    assert corpus.storage_bytes == 3 * 8 // 2 + 4
    assert results[0].doc_id == "positive"


def test_corpus_save_load_roundtrips_binary_q40_scores_and_metadata(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(
        ["positive", "negative", "mixed"],
        docs,
        offsets,
        mode="binary_q40",
        metadata={"source": "tiny"},
    )
    query = np.ones((1, 8), dtype=np.float32)
    expected = bitmax.Reranker.from_corpus(corpus).search(query, k=3)

    corpus.save(tmp_path / "tiny.bitmax.npz")
    loaded = bitmax.Corpus.load(tmp_path / "tiny.bitmax.npz")
    actual = bitmax.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.doc_ids == corpus.doc_ids
    assert loaded.mode == "binary_q40"
    assert loaded.metadata == {"source": "tiny"}
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)


def test_reranker_load_constructs_runtime_from_saved_corpus(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="binary")
    corpus.save(tmp_path / "tiny.bitmax.npz")

    reranker = bitmax.Reranker.load(tmp_path / "tiny.bitmax.npz")
    results = reranker.search(np.ones((1, 8), dtype=np.float32), k=1)

    assert results[0].doc_id == "positive"


def test_corpus_save_load_roundtrips_int4_scores(tmp_path):
    docs, offsets = _tiny_multivector_docs()
    corpus = bitmax.Corpus.from_embeddings(["positive", "negative", "mixed"], docs, offsets, mode="int4")
    query = np.ones((1, 8), dtype=np.float32)
    expected = bitmax.Reranker.from_corpus(corpus).search(query, k=3)

    corpus.save(tmp_path / "tiny-int4.bitmax.npz")
    loaded = bitmax.Corpus.load(tmp_path / "tiny-int4.bitmax.npz")
    actual = bitmax.Reranker.from_corpus(loaded).search(query, k=3)

    assert loaded.mode == "int4"
    assert loaded.storage_bytes == corpus.storage_bytes
    assert [result.doc_id for result in actual] == [result.doc_id for result in expected]
    np.testing.assert_allclose([result.score for result in actual], [result.score for result in expected], rtol=0, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_sdk.py
```

Expected: q40/int4/persistence tests fail because those SDK modes and save/load are missing.

- [ ] **Step 3: Implement q40, int4, and persistence**

Update `src/bitmax/sdk.py`:

- Add imports:

```python
import json
from pathlib import Path

from bitmax.experimental import (
    DimCentroidCalibration,
    fit_dim_centroid_calibration,
    int4_maxsim,
    int4_to_device,
    pack_dim_centroid_signs,
    pack_int4_symmetric,
    topk_dim_centroid_maxsim,
    topk_int4_maxsim,
)
```

- In `Corpus.from_embeddings`, implement the two missing modes:

```python
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
```

- Add `Corpus.save` and `Corpus.load` methods:

```python
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
            if mode in {"binary", "binary_q40"}:
                calibration = None
                if mode == "binary_q40":
                    calibration = DimCentroidCalibration(
                        thresholds=np.ascontiguousarray(data["centroid_thresholds"], dtype=np.float32),
                        negative_centroids=np.ascontiguousarray(data["centroid_negative"], dtype=np.float32),
                        positive_centroids=np.ascontiguousarray(data["centroid_positive"], dtype=np.float32),
                    )
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
```

- Add `Reranker.load`:

```python
    @classmethod
    def load(cls, path, *, device: Literal["cpu", "cuda"] = "cpu") -> "Reranker":
        return cls.from_corpus(Corpus.load(path), device=device)
```

- Update `Reranker.search` and `Reranker.rerank` to dispatch by mode:

```python
    def search(self, query_embeddings, *, k: int = 10):
        _validate_k(k, self.corpus.num_docs)
        actual_k = min(k, self.corpus.num_docs)
        if self.corpus.mode == "binary":
            scores, indices = topk_maxsim(query_embeddings, self.corpus.packed, actual_k, device="auto")
        elif self.corpus.mode == "binary_q40":
            scores, indices = topk_dim_centroid_maxsim(query_embeddings, self.corpus.packed, self.corpus.calibration, actual_k, device="auto")
        elif self.corpus.mode == "int4":
            scores, indices = topk_int4_maxsim(query_embeddings, self.corpus.int4_packed, actual_k, device="auto")
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
        if self.corpus.mode == "binary":
            return maxsim(query_embeddings, self.corpus.packed, device="auto")
        if self.corpus.mode == "binary_q40":
            from bitmax.experimental import dim_centroid_maxsim

            return dim_centroid_maxsim(query_embeddings, self.corpus.packed, self.corpus.calibration, device="auto")
        if self.corpus.mode == "int4":
            return int4_maxsim(query_embeddings, self.corpus.int4_packed, device="auto")
        raise ValueError(f"unsupported corpus mode: {self.corpus.mode}")
```

- Update `_move_corpus` for int4:

```python
    if corpus.int4_packed is not None:
        return Corpus(
            doc_ids=corpus.doc_ids,
            mode=corpus.mode,
            packed=corpus.packed,
            int4_packed=int4_to_device(corpus.int4_packed),
            calibration=corpus.calibration,
            metadata=corpus.metadata,
        )
```

- Add helpers:

```python
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


def _unpack_signed_int4_data(data: np.ndarray, dim: int) -> np.ndarray:
    values = np.empty((data.shape[0], dim), dtype=np.int8)
    low = (data & 0x0F).astype(np.int8)
    high = ((data >> 4) & 0x0F).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    values[:, 0::2] = low
    values[:, 1::2] = high
    return np.ascontiguousarray(values, dtype=np.int8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_sdk.py tests/test_int4_reference.py tests/test_experimental_dim_centroids.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/bitmax/sdk.py tests/test_sdk.py
git commit -m "feat: add sdk corpus modes and persistence"
```

## Task 3: CUDA SDK Correctness Tests

**Files:**
- Modify: `tests/test_cuda_extension.py`

- [ ] **Step 1: Add failing CUDA SDK tests**

Append to `tests/test_cuda_extension.py`:

```python
@pytest.mark.cuda
def test_cuda_sdk_binary_and_q40_search_match_cpu():
    pytest.importorskip("bitmax._bitmax_cuda")
    rng = np.random.default_rng(20260707)
    docs = rng.normal(size=(64, 128)).astype(np.float32)
    offsets = np.array([0, 13, 29, 47, 64], dtype=np.int64)
    query = rng.normal(size=(3, 5, 128)).astype(np.float32)
    doc_ids = [f"doc-{idx}" for idx in range(4)]

    for mode in ("binary", "binary_q40"):
        corpus = bitmax.Corpus.from_embeddings(doc_ids, docs, offsets, mode=mode)
        cpu_results = bitmax.Reranker.from_corpus(corpus, device="cpu").search(query, k=3)
        cuda_results = bitmax.Reranker.from_corpus(corpus, device="cuda").search(query, k=3)

        assert [[result.doc_id for result in row] for row in cuda_results] == [[result.doc_id for result in row] for row in cpu_results]
        for cuda_row, cpu_row in zip(cuda_results, cpu_results):
            np.testing.assert_allclose([r.score for r in cuda_row], [r.score for r in cpu_row], rtol=0, atol=1e-4)


@pytest.mark.cuda
def test_cuda_sdk_int4_search_and_rerank_match_cpu():
    pytest.importorskip("bitmax._bitmax_cuda")
    rng = np.random.default_rng(20260708)
    docs = rng.normal(size=(64, 128)).astype(np.float32)
    offsets = np.array([0, 13, 29, 47, 64], dtype=np.int64)
    query = rng.normal(size=(2, 5, 128)).astype(np.float32)
    doc_ids = [f"doc-{idx}" for idx in range(4)]
    corpus = bitmax.Corpus.from_embeddings(doc_ids, docs, offsets, mode="int4")

    cpu = bitmax.Reranker.from_corpus(corpus, device="cpu")
    cuda = bitmax.Reranker.from_corpus(corpus, device="cuda")
    cpu_search = cpu.search(query, k=3)
    cuda_search = cuda.search(query, k=3)
    cpu_rerank = cpu.rerank(query, ["doc-3", "doc-1", "doc-2"], k=2)
    cuda_rerank = cuda.rerank(query, ["doc-3", "doc-1", "doc-2"], k=2)

    assert [[result.doc_id for result in row] for row in cuda_search] == [[result.doc_id for result in row] for row in cpu_search]
    assert [[result.doc_id for result in row] for row in cuda_rerank] == [[result.doc_id for result in row] for row in cpu_rerank]
    for cuda_row, cpu_row in zip(cuda_search, cpu_search):
        np.testing.assert_allclose([r.score for r in cuda_row], [r.score for r in cpu_row], rtol=0, atol=1e-4)
    for cuda_row, cpu_row in zip(cuda_rerank, cpu_rerank):
        np.testing.assert_allclose([r.score for r in cuda_row], [r.score for r in cpu_row], rtol=0, atol=1e-4)
```

- [ ] **Step 2: Run local tests to verify CUDA tests skip locally**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_cuda_extension.py
```

Expected: CUDA tests skip locally if no CUDA extension is installed.

- [ ] **Step 3: Sync to VAST, rebuild CUDA, and verify tests pass**

Run:

```bash
rsync -avR -e 'ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
  src/bitmax/sdk.py src/bitmax/__init__.py tests/test_sdk.py tests/test_cuda_extension.py \
  root@ssh2.vast.ai:/workspace/bitmax/
ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && BITMAX_BUILD_CUDA=1 python -m pip install -e ".[dev]" && python -m pytest -q tests/test_sdk.py tests/test_cuda_extension.py'
```

Expected: VAST reports all selected SDK and CUDA tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/test_cuda_extension.py
git commit -m "test: add cuda sdk correctness coverage"
```

## Task 4: CUDA-First Local Multivector Demo

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/local_multivector_search.py`
- Create: `tests/test_sdk_demo.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing demo smoke test**

Create `tests/test_sdk_demo.py`:

```python
import json

import numpy as np

from examples.local_multivector_search import main


def _write_demo_fixture(path):
    doc_embeddings = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ],
        dtype=np.float32,
    )
    query_embeddings = np.array(
        [
            [[1, 1, 1, 1, 1, 1, 1, 1]],
            [[-1, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        doc_embeddings=doc_embeddings,
        doc_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        query_embeddings=query_embeddings,
        qrels=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        doc_ids=np.array(["positive", "negative", "mixed"]),
        query_ids=np.array(["q0", "q1"]),
        dataset_name=np.array("demo-fixture"),
    )


def test_local_multivector_demo_emits_comparison_json(tmp_path, capsys):
    input_path = tmp_path / "fixture.npz"
    output_path = tmp_path / "demo.json"
    _write_demo_fixture(input_path)

    main([
        "--input",
        str(input_path),
        "--device",
        "cpu",
        "--modes",
        "binary,binary_q40,int4",
        "--output",
        str(output_path),
    ])

    captured = capsys.readouterr()
    assert "dense_fp16_baseline" in captured.out
    assert "bitmax_binary" in captured.out
    data = json.loads(output_path.read_text())
    rows = {row["implementation"]: row for row in data["results"]}
    assert data["benchmark"] == "sdk_local_multivector_search"
    assert rows["dense_fp16_baseline"]["ndcg_at_10"] == 1.0
    assert rows["bitmax_binary"]["doc_memory_compression_vs_fp32"] == 32.0
    assert rows["bitmax_binary_q40"]["doc_memory_compression_vs_fp32"] > 31.0
    assert rows["bitmax_int4"]["doc_memory_compression_vs_fp32"] == 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_sdk_demo.py
```

Expected: fails because `examples.local_multivector_search` is missing.

- [ ] **Step 3: Implement demo**

Create `examples/__init__.py` as an empty file.

Create `examples/local_multivector_search.py` with:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import bitmax

try:
    import torch
except ImportError:
    torch = None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the bitmax SDK local multi-vector CUDA demo.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--modes", default="binary,binary_q40,int4")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args(argv)

    dataset = _load_demo_dataset(Path(args.input), limit_queries=args.limit_queries)
    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    dense_scores, dense_latency = _time_call(lambda: _dense_fp16_scores(dataset, args.device), repeat=args.repeat)
    dense_row = _result_row(
        "dense_fp16_baseline",
        dataset,
        dense_scores,
        dense_latency,
        args.k,
        doc_storage_bytes=_dense_doc_bytes(dataset, 2),
        dense_scores=dense_scores,
    )
    rows = [dense_row]

    for mode in modes:
        corpus = bitmax.Corpus.from_embeddings(dataset["doc_ids"], dataset["doc_embeddings"], dataset["doc_offsets"], mode=mode)
        reranker = bitmax.Reranker.from_corpus(corpus, device=args.device)
        scores, latency = _time_call(lambda: _scores_from_search(reranker, dataset["query_embeddings"], dataset["doc_ids"], args.k), repeat=args.repeat)
        rows.append(
            _result_row(
                f"bitmax_{mode}",
                dataset,
                scores,
                latency,
                args.k,
                doc_storage_bytes=corpus.storage_bytes,
                dense_scores=dense_scores,
                baseline_latency=dense_row["latency_ms"],
                mode=mode,
                device=args.device,
            )
        )

    result = {
        "schema_version": 1,
        "benchmark": "sdk_local_multivector_search",
        "dataset": {
            "name": dataset["name"],
            "queries": int(len(dataset["query_embeddings"])),
            "docs": int(len(dataset["doc_ids"])),
            "dim": int(dataset["doc_embeddings"].shape[1]),
            "doc_tokens": int(dataset["doc_embeddings"].shape[0]),
        },
        "device": args.device,
        "top_k": int(args.k),
        "repeat": int(args.repeat),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print_table(rows, args.k)
    return result
```

Add helper functions in the same file:

```python
def _load_demo_dataset(path: Path, *, limit_queries: int | None):
    with np.load(path, allow_pickle=False) as data:
        query_embeddings = np.asarray(data["query_embeddings"], dtype=np.float32)
        if query_embeddings.ndim == 2:
            offsets = np.asarray(data["query_offsets"], dtype=np.int64)
            queries = tuple(query_embeddings[int(start) : int(end)] for start, end in zip(offsets[:-1], offsets[1:]))
        else:
            queries = tuple(query_embeddings[idx] for idx in range(query_embeddings.shape[0]))
        if limit_queries is not None:
            queries = queries[: int(limit_queries)]
        doc_offsets = np.asarray(data["doc_offsets"], dtype=np.int64)
        doc_count = int(doc_offsets.shape[0] - 1)
        qrels = np.asarray(data["qrels"], dtype=np.float32)[: len(queries)]
        doc_ids = tuple(str(value) for value in np.asarray(data["doc_ids"])) if "doc_ids" in data.files else tuple(f"doc-{idx}" for idx in range(doc_count))
        name = str(np.asarray(data["dataset_name"]).item()) if "dataset_name" in data.files else path.stem
        return {
            "name": name,
            "doc_embeddings": np.asarray(data["doc_embeddings"], dtype=np.float32),
            "doc_offsets": doc_offsets,
            "query_embeddings": tuple(np.ascontiguousarray(query, dtype=np.float32) for query in queries),
            "qrels": qrels,
            "doc_ids": doc_ids,
        }


def _time_call(fn, *, repeat: int):
    best_latency = float("inf")
    best_value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1_000.0
        if latency < best_latency:
            best_latency = latency
            best_value = value
    return best_value, best_latency


def _dense_fp16_scores(dataset, device: str) -> np.ndarray:
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        docs = torch.as_tensor(dataset["doc_embeddings"], dtype=torch.float16, device="cuda").to(torch.float32)
        rows = []
        for query in dataset["query_embeddings"]:
            query_tensor = torch.as_tensor(query, dtype=torch.float16, device="cuda").to(torch.float32)
            per_doc = []
            for doc_idx in range(len(dataset["doc_ids"])):
                start = int(dataset["doc_offsets"][doc_idx])
                end = int(dataset["doc_offsets"][doc_idx + 1])
                doc = docs[start:end]
                per_doc.append((query_tensor @ doc.T).max(dim=1).values.sum())
            rows.append(torch.stack(per_doc))
        torch.cuda.synchronize()
        return torch.stack(rows).detach().cpu().numpy().astype(np.float32)
    docs = dataset["doc_embeddings"].astype(np.float16).astype(np.float32)
    rows = []
    for query in dataset["query_embeddings"]:
        query_float = query.astype(np.float16).astype(np.float32)
        row = np.empty((len(dataset["doc_ids"]),), dtype=np.float32)
        for doc_idx in range(len(dataset["doc_ids"])):
            start = int(dataset["doc_offsets"][doc_idx])
            end = int(dataset["doc_offsets"][doc_idx + 1])
            row[doc_idx] = np.max(query_float @ docs[start:end].T, axis=1).sum(dtype=np.float32)
        rows.append(row)
    return np.stack(rows, axis=0)


def _scores_from_search(reranker, queries, doc_ids, k: int) -> np.ndarray:
    scores = np.full((len(queries), len(doc_ids)), -np.inf, dtype=np.float32)
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    for query_idx, query in enumerate(queries):
        results = reranker.search(query, k=min(k, len(doc_ids)))
        for result in results:
            scores[query_idx, doc_index[result.doc_id]] = result.score
    return scores
```

Add metric/table helpers in the same file:

```python
def _result_row(implementation, dataset, scores, latency_ms, k, *, doc_storage_bytes, dense_scores, baseline_latency=None, mode=None, device=None):
    metrics = _ranking_metrics(scores, dataset["qrels"], k=min(k, len(dataset["doc_ids"])))
    dense_metrics = _ranking_metrics(dense_scores, dataset["qrels"], k=min(k, len(dataset["doc_ids"])))
    row = {
        "implementation": implementation,
        "latency_ms": float(latency_ms),
        "query_count": int(len(dataset["query_embeddings"])),
        "docs": int(len(dataset["doc_ids"])),
        "doc_storage_bytes": int(doc_storage_bytes),
        "doc_memory_compression_vs_fp16": float(_dense_doc_bytes(dataset, 2) / max(doc_storage_bytes, 1)),
        "doc_memory_compression_vs_fp32": float(_dense_doc_bytes(dataset, 4) / max(doc_storage_bytes, 1)),
        "recall_at_1": float(_ranking_metrics(scores, dataset["qrels"], k=1)["recall_at_k"]),
        f"recall_at_{k}": float(metrics["recall_at_k"]),
        f"mrr_at_{k}": float(metrics["mrr_at_k"]),
        f"ndcg_at_{k}": float(metrics["ndcg_at_k"]),
        f"quality_delta_vs_dense_ndcg_at_{k}": float(metrics["ndcg_at_k"] - dense_metrics["ndcg_at_k"]),
    }
    if baseline_latency is not None:
        row["speedup_vs_dense_fp16"] = float(baseline_latency / max(latency_ms, 1e-12))
    if mode is not None:
        row["mode"] = mode
    if device is not None:
        row["device"] = device
    return row


def _dense_doc_bytes(dataset, bytes_per_value: int) -> int:
    return int(dataset["doc_embeddings"].shape[0] * dataset["doc_embeddings"].shape[1] * bytes_per_value)


def _ranking_metrics(scores: np.ndarray, qrels: np.ndarray, *, k: int):
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    for query_scores, query_relevance in zip(scores, qrels):
        relevant_total = float(np.sum(query_relevance > 0))
        order = np.lexsort((np.arange(query_scores.shape[0], dtype=np.int64), -query_scores))[:k]
        hits = query_relevance[order] > 0
        recalls.append(float(np.sum(hits)) / relevant_total if relevant_total else 0.0)
        hit_positions = np.flatnonzero(hits)
        reciprocal_ranks.append(0.0 if hit_positions.size == 0 else 1.0 / float(hit_positions[0] + 1))
        gains = query_relevance[order]
        discounts = 1.0 / np.log2(np.arange(2, gains.shape[0] + 2, dtype=np.float64))
        dcg = float(np.sum(gains * discounts))
        ideal = np.sort(query_relevance)[::-1][:k]
        ideal_dcg = float(np.sum(ideal * discounts[: ideal.shape[0]]))
        ndcgs.append(0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg)
    return {
        "recall_at_k": float(np.mean(recalls)),
        "mrr_at_k": float(np.mean(reciprocal_ranks)),
        "ndcg_at_k": float(np.mean(ndcgs)),
    }


def _print_table(rows, k: int) -> None:
    print("implementation, latency_ms, fp32_reduction, recall@1, recall@%d, mrr@%d, ndcg@%d" % (k, k, k))
    for row in rows:
        print(
            f"{row['implementation']}, {row['latency_ms']:.3f}, "
            f"{row['doc_memory_compression_vs_fp32']:.2f}x, "
            f"{row['recall_at_1']:.3f}, {row[f'recall_at_{k}']:.3f}, "
            f"{row[f'mrr_at_{k}']:.3f}, {row[f'ndcg_at_{k}']:.3f}"
        )


if __name__ == "__main__":
    main()
```

Modify `pyproject.toml`:

```toml
[tool.scikit-build]
cmake.source-dir = "."
wheel.packages = ["src/bitmax", "benchmarks", "ops", "examples"]
```

- [ ] **Step 4: Run local demo test**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_sdk_demo.py
```

Expected: demo test passes.

- [ ] **Step 5: Commit**

Run:

```bash
git add examples pyproject.toml tests/test_sdk_demo.py
git commit -m "feat: add sdk local multivector demo"
```

## Task 5: CUDA Demo Benchmark on VAST

**Files:**
- No local code files unless the CUDA demo exposes a bug.
- Fetch artifact: `benchmark-results/sdk-demo-local-search-limit256-cuda.json`

- [ ] **Step 1: Sync SDK/demo files to VAST**

Run:

```bash
rsync -avR -e 'ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
  src/bitmax/sdk.py src/bitmax/__init__.py examples pyproject.toml tests/test_sdk.py tests/test_sdk_demo.py tests/test_cuda_extension.py \
  root@ssh2.vast.ai:/workspace/bitmax/
```

Expected: files transfer to `/workspace/bitmax`.

- [ ] **Step 2: Reinstall on VAST**

Run:

```bash
ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && BITMAX_BUILD_CUDA=1 python -m pip install -e ".[dev]"'
```

Expected: editable wheel builds and installs successfully.

- [ ] **Step 3: Run VAST SDK and CUDA tests**

Run:

```bash
ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m pytest -q tests/test_sdk.py tests/test_sdk_demo.py tests/test_cuda_extension.py'
```

Expected: selected tests pass on CUDA worker.

- [ ] **Step 4: Run CUDA SDK demo benchmark**

Run:

```bash
ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m examples.local_multivector_search \
    --input benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
    --device cuda \
    --modes binary,binary_q40,int4 \
    --repeat 5 \
    --output benchmark-results/sdk-demo-local-search-limit256-cuda.json'
```

Expected: output table includes `dense_fp16_baseline`, `bitmax_binary`,
`bitmax_binary_q40`, and `bitmax_int4`.

- [ ] **Step 5: Fetch CUDA demo artifact**

Run:

```bash
scp -P 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@ssh2.vast.ai:/workspace/bitmax/benchmark-results/sdk-demo-local-search-limit256-cuda.json \
  benchmark-results/
```

Expected: artifact exists locally.

- [ ] **Step 6: Commit fetched artifact if benchmark artifacts are tracked**

Check:

```bash
git status --short benchmark-results/sdk-demo-local-search-limit256-cuda.json
```

If the artifact is not ignored, run:

```bash
git add benchmark-results/sdk-demo-local-search-limit256-cuda.json
git commit -m "bench: add sdk cuda demo artifact"
```

If the artifact is ignored, record the file path in docs during Task 6.

## Task 6: SDK Documentation and README

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/benchmarks.md`

- [ ] **Step 1: Update README with SDK-first usage**

Replace the minimal-use section in `README.md` with:

````markdown
## SDK Use

```python
import bitmax

corpus = bitmax.Corpus.from_embeddings(
    doc_ids=doc_ids,
    embeddings=doc_embeddings,
    offsets=doc_offsets,
    mode="binary_q40",
    metadata={"model": "vidore/colqwen2-v1.0-hf"},
)
corpus.save("docs.bitmax.npz")

reranker = bitmax.Reranker.load("docs.bitmax.npz", device="cuda")
results = reranker.search(query_embeddings, k=10)

reranked = reranker.rerank(
    query_embeddings,
    candidate_ids=["doc-17", "doc-03", "doc-91"],
    k=3,
)
```
````

Also document mode guidance:

```markdown
- `binary`: fastest, 32x fp32 document compression.
- `binary_q40`: experimental 32x-ish accuracy mode using q40 centroid calibration.
- `int4`: experimental accuracy-first mode, 8x fp32 compression.
```

- [ ] **Step 2: Update API docs**

Add sections to `docs/api.md`:

```markdown
## SDK Corpus

`Corpus.from_embeddings(doc_ids, embeddings, offsets, mode="binary")` builds a
portable packed corpus from multi-vector document embeddings.

## SDK Reranker

`Reranker.search(query_embeddings, k=10)` scores the whole corpus.
`Reranker.rerank(query_embeddings, candidate_ids, k=10)` returns only the
requested candidates sorted by compressed MaxSim score.

## SDK Results

Each result is `SearchResult(doc_id: str, score: float, rank: int)`.
```

- [ ] **Step 3: Update benchmark docs with CUDA demo command**

Add to `docs/benchmarks.md`:

````markdown
## SDK CUDA Demo

```bash
python -m examples.local_multivector_search \
  --input benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
  --device cuda \
  --modes binary,binary_q40,int4 \
  --output benchmark-results/sdk-demo-local-search-limit256-cuda.json
```

This demo is the SDK-level proof artifact. It compares dense fp16 CUDA against
bitmax SDK modes on the same multi-vector embedding slice and reports storage,
latency, speedup, recall@1, recall@10, MRR@10, and NDCG@10.
````

- [ ] **Step 4: Run docs sanity checks and local tests**

Run:

```bash
git diff --check
.venv/bin/python -m pytest -q tests/test_sdk.py tests/test_sdk_demo.py tests/test_api_reference.py
```

Expected: no whitespace errors; selected tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md docs/api.md docs/benchmarks.md
git commit -m "docs: document sdk cuda workflow"
```

## Task 7: Final CUDA-First Verification

**Files:**
- No code files expected.

- [ ] **Step 1: Run full local test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all local tests pass, CUDA tests skip if no local CUDA extension is installed.

- [ ] **Step 2: Run full VAST test suite**

Run:

```bash
ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m pytest -q'
```

Expected: full CUDA worker test suite passes.

- [ ] **Step 3: Re-run CUDA SDK demo if code changed after Task 5**

Run only if Tasks 6 or 7 changed SDK/demo behavior:

```bash
ssh -p 18164 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@ssh2.vast.ai \
  'cd /workspace/bitmax && . .venv_torch/bin/activate && python -m examples.local_multivector_search \
    --input benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
    --device cuda \
    --modes binary,binary_q40,int4 \
    --repeat 5 \
    --output benchmark-results/sdk-demo-local-search-limit256-cuda.json'
```

Expected: JSON artifact updates and printed table includes all four rows.

- [ ] **Step 4: Confirm git status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes unless benchmark artifact policy keeps generated files ignored.

## Self-Review

- Spec coverage: The plan covers `Corpus`, `Reranker`, `SearchResult`, binary/q40/int4 modes, persistence, candidate reranking, demo, docs, local correctness tests, CUDA tests, and VAST CUDA benchmark proof.
- Completeness scan: The plan avoids vague tasks; each code-producing step names concrete files, code, commands, and expected results.
- CUDA scope: CPU appears only for correctness tests and fallback execution; all performance proof steps target CUDA on the VAST worker.
- Type consistency: Public names are consistent: `Corpus`, `Reranker`, `SearchResult`, `from_embeddings`, `save`, `load`, `search`, `rerank`, `binary`, `binary_q40`, and `int4`.
