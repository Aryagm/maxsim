"""End-to-end retrieval benchmark over real ragged embedding caches.

The compressed cases deliberately use only the public ``maxsim.Index`` and
``maxsim.Reranker`` APIs.  Results are checkpointed one case at a time so a
long corpus run can resume without repeating completed formats.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import maxsim


DEFAULT_MODES = (
    "dense",
    "binary",
    "binary_token_scale_u4",
    "pooled_binary",
    "int4",
    "int4_per_token",
    "int4_residual",
)
_MODE_ALIASES = {
    "dense_fp16": "dense",
    "reference": "dense",
    "compact": "binary_token_scale_u4",
    "pool3": "pooled_binary",
    "max_compression": "pooled_binary",
    "tensor_int4": "int4",
    "per_token_int4": "int4_per_token",
    "residual": "int4_residual",
}
_COMPRESSED_MODES = {
    "binary",
    "binary_token_scale_u4",
    "pooled_binary",
    "int4",
    "int4_per_token",
    "int4_residual",
}


@dataclass(frozen=True)
class RetrievalCache:
    path: Path
    name: str
    doc_embeddings: np.ndarray
    doc_offsets: np.ndarray
    query_embeddings: np.ndarray
    query_offsets: np.ndarray
    qrels: np.ndarray
    doc_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def num_docs(self) -> int:
        return int(self.doc_offsets.shape[0] - 1)

    @property
    def num_queries(self) -> int:
        return int(self.query_offsets.shape[0] - 1)

    @property
    def dim(self) -> int:
        return int(self.doc_embeddings.shape[1])

    def query(self, index: int) -> np.ndarray:
        start = int(self.query_offsets[index])
        end = int(self.query_offsets[index + 1])
        return np.ascontiguousarray(self.query_embeddings[start:end], dtype=np.float32)


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    mode: str
    rescore_candidates: int | None = None


@dataclass(frozen=True)
class _RankedResult:
    doc_id: str
    score: float
    rank: int


class _SdkSearcher:
    def __init__(self, reranker: maxsim.Reranker, rescore_candidates: int | None):
        self.reranker = reranker
        self.rescore_candidates = rescore_candidates

    @property
    def implementation(self) -> str:
        return f"maxsim_sdk_{self.reranker.corpus.mode}"

    @property
    def uses_public_sdk(self) -> bool:
        return True

    def search(self, query: np.ndarray, k: int):
        return self.reranker.search(
            query,
            k=k,
            rescore_candidates=self.rescore_candidates,
        )

    def memory_report(self):
        return self.reranker.corpus.memory_report()


class _DenseReference:
    """Chunked dense MaxSim reference with bounded temporary memory."""

    def __init__(
        self,
        docs: np.ndarray,
        offsets: np.ndarray,
        doc_ids: tuple[str, ...],
        *,
        device: str,
        chunk_tokens: int,
    ) -> None:
        self.device = device
        self.offsets = offsets
        self.doc_ids = doc_ids
        self.chunk_tokens = chunk_tokens
        self._torch = None
        self.docs = docs.astype(np.float32, copy=False)
        if device == "cuda":
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - exercised on GPU hosts
                raise RuntimeError("dense CUDA reference requires PyTorch") from exc
            if not torch.cuda.is_available():  # pragma: no cover - exercised on GPU hosts
                raise RuntimeError("--device cuda requested but PyTorch cannot access CUDA")
            self._torch = torch
            self.docs = torch.as_tensor(docs, dtype=torch.float16, device="cuda")

    @property
    def encoded_bytes(self) -> int:
        itemsize = 2 if self.device == "cuda" else 4
        return int(self.offsets[-1]) * int(self.docs.shape[1]) * itemsize

    @property
    def implementation(self) -> str:
        return "torch_dense_fp16_reference" if self.device == "cuda" else "numpy_dense_fp32_reference"

    @property
    def uses_public_sdk(self) -> bool:
        return False

    @property
    def resident_gpu_bytes(self) -> int:
        return self.encoded_bytes if self.device == "cuda" else 0

    def search(self, query: np.ndarray, k: int):
        values = np.asarray(query, dtype=np.float32)
        squeeze = values.ndim == 2
        if squeeze:
            values = values[np.newaxis, :, :]
        if values.ndim != 3:
            raise ValueError("dense queries must have shape [tokens, dim] or [batch, tokens, dim]")
        scores = self._score_cuda(values) if self.device == "cuda" else self._score_cpu(values)
        rows = []
        actual_k = min(k, len(self.doc_ids))
        tie_ids = np.arange(len(self.doc_ids), dtype=np.int64)
        for score_row in scores:
            order = np.lexsort((tie_ids, -score_row))[:actual_k]
            rows.append(
                [
                    _RankedResult(
                        doc_id=self.doc_ids[int(index)],
                        score=float(score_row[int(index)]),
                        rank=rank,
                    )
                    for rank, index in enumerate(order, start=1)
                ]
            )
        return rows[0] if squeeze else rows

    def _score_cpu(self, queries: np.ndarray) -> np.ndarray:
        rows = np.empty((queries.shape[0], len(self.doc_ids)), dtype=np.float32)
        for batch_index, query in enumerate(queries):
            for doc_index, (start, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
                doc = self.docs[int(start) : int(end)]
                rows[batch_index, doc_index] = (
                    np.max(query @ doc.T, axis=1).sum(dtype=np.float32)
                    if doc.shape[0]
                    else 0.0
                )
        return rows

    def _score_cuda(self, queries: np.ndarray) -> np.ndarray:  # pragma: no cover - GPU only
        torch = self._torch
        query = torch.as_tensor(queries, dtype=torch.float16, device="cuda")
        score_chunks = []
        with torch.no_grad():
            for doc_start, doc_end in _document_chunks(self.offsets, self.chunk_tokens):
                token_start = int(self.offsets[doc_start])
                token_end = int(self.offsets[doc_end])
                dots = torch.matmul(query, self.docs[token_start:token_end].T)
                lengths = torch.as_tensor(
                    np.diff(self.offsets[doc_start : doc_end + 1]),
                    dtype=torch.int64,
                    device="cuda",
                )
                token_docs = torch.repeat_interleave(
                    torch.arange(doc_end - doc_start, device="cuda"), lengths
                )
                maxima = torch.full(
                    (query.shape[0], query.shape[1], doc_end - doc_start),
                    -torch.inf,
                    dtype=dots.dtype,
                    device="cuda",
                )
                maxima.scatter_reduce_(
                    2,
                    token_docs.reshape(1, 1, -1).expand(query.shape[0], query.shape[1], -1),
                    dots,
                    reduce="amax",
                    include_self=True,
                )
                maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
                score_chunks.append(maxima.float().sum(dim=1))
            scores = torch.cat(score_chunks, dim=1)
        torch.cuda.synchronize()
        return scores.cpu().numpy().astype(np.float32, copy=False)


def load_retrieval_cache(path: str | Path) -> RetrievalCache:
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {"doc_embeddings", "doc_offsets", "query_embeddings"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"embedding cache is missing required arrays: {sorted(missing)}")
        docs = np.asarray(data["doc_embeddings"]).astype(np.float32, copy=False)
        doc_offsets = _integer_array(data["doc_offsets"], "doc_offsets")
        queries, query_offsets = _flatten_queries(data)
        num_docs = int(doc_offsets.shape[0] - 1)
        num_queries = int(query_offsets.shape[0] - 1)
        qrels = _load_qrels(data, num_queries, num_docs)
        doc_ids = _load_ids(data, "doc_ids", num_docs, "doc")
        query_ids = _load_ids(data, "query_ids", num_queries, "query")
        name = _load_scalar(data, "dataset_name", source.stem)
        metadata = _load_cache_metadata(data)
    _validate_cache(docs, doc_offsets, queries, query_offsets, qrels)
    return RetrievalCache(
        path=source,
        name=name,
        doc_embeddings=docs,
        doc_offsets=doc_offsets,
        query_embeddings=queries,
        query_offsets=query_offsets,
        qrels=qrels,
        doc_ids=doc_ids,
        query_ids=query_ids,
        metadata=metadata,
    )


def run_production_benchmark(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cuda",
    modes: Sequence[str] | str = DEFAULT_MODES,
    top_k: int = 10,
    query_limit: int | None = None,
    seed: int = 20260715,
    warmup: int = 3,
    latency_repeats: int = 1,
    throughput_batch_size: int = 8,
    cascade_candidates: Sequence[int] = (),
    int4_query: str = "fp32",
    pooled_binary_pool_factor: int = 3,
    dense_chunk_tokens: int = 262_144,
    serialize_indexes: bool = True,
    checkpoint_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    cache = load_retrieval_cache(input_path)
    output = Path(output_path).resolve()
    normalized_modes = _normalize_modes(modes)
    candidates = tuple(sorted(set(int(value) for value in cascade_candidates)))
    _validate_run_config(
        cache,
        device=device,
        modes=normalized_modes,
        top_k=top_k,
        query_limit=query_limit,
        warmup=warmup,
        latency_repeats=latency_repeats,
        throughput_batch_size=throughput_batch_size,
        cascade_candidates=candidates,
        int4_query=int4_query,
        pooled_binary_pool_factor=pooled_binary_pool_factor,
        dense_chunk_tokens=dense_chunk_tokens,
    )
    selected = deterministic_query_indices(cache.num_queries, query_limit, seed)
    checkpoints = (
        Path(checkpoint_dir).resolve()
        if checkpoint_dir is not None
        else output.with_suffix("").with_name(output.stem + ".checkpoints")
    )
    indexes = (
        Path(index_dir).resolve()
        if index_dir is not None
        else output.with_suffix("").with_name(output.stem + ".indexes")
    )
    config = {
        "device": device,
        "modes": list(normalized_modes),
        "top_k": int(top_k),
        "query_limit": None if query_limit is None else int(query_limit),
        "seed": int(seed),
        "warmup": int(warmup),
        "latency_repeats": int(latency_repeats),
        "throughput_batch_size": int(throughput_batch_size),
        "cascade_candidates": list(candidates),
        "int4_query": int4_query,
        "pooled_binary_pool_factor": int(pooled_binary_pool_factor),
        "dense_chunk_tokens": int(dense_chunk_tokens),
        "serialize_indexes": bool(serialize_indexes),
    }
    signature = _run_signature(cache.path, config, selected)
    specs = _case_specs(normalized_modes, candidates)
    completed: dict[str, dict[str, Any]] = {}
    resumed_cases: list[str] = []
    for spec in specs:
        checkpoint = checkpoints / f"{spec.case_id}.json"
        row = None if force else _read_complete_checkpoint(checkpoint, signature, spec.case_id)
        if row is not None:
            completed[spec.case_id] = row
            resumed_cases.append(spec.case_id)

    if "dense" in normalized_modes and "dense" not in completed:
        _reset_torch_peak(device)
        before_gpu = _process_gpu_bytes()
        start = time.perf_counter()
        dense = _DenseReference(
            cache.doc_embeddings,
            cache.doc_offsets,
            cache.doc_ids,
            device=device,
            chunk_tokens=dense_chunk_tokens,
        )
        device_load_ms = _elapsed_ms(start)
        build = {
            "packing_ms": 0.0,
            "serialization_ms": None,
            "load_ms": None,
            "device_load_ms": device_load_ms,
            "index_reused": False,
            "encoded_bytes": dense.encoded_bytes,
            "serialized_bytes": None,
            "host_array_bytes_before_device": int(cache.doc_embeddings.nbytes),
        }
        memory_base = {
            "device_index_bytes": dense.resident_gpu_bytes,
            "device_workspace_bytes": 0,
            "resident_gpu_bytes": dense.resident_gpu_bytes,
            "process_gpu_bytes_before_load": before_gpu,
            "process_gpu_bytes_after_load": _process_gpu_bytes(),
        }
        row = _evaluate_case(
            cache,
            selected,
            _CaseSpec("dense", "dense"),
            dense,
            build=build,
            memory_base=memory_base,
            signature=signature,
            device=device,
            top_k=top_k,
            warmup=warmup,
            latency_repeats=latency_repeats,
            throughput_batch_size=throughput_batch_size,
        )
        _atomic_write_json(checkpoints / "dense.json", row)
        completed["dense"] = row
        del dense
        _release_runtime(device)

    for mode in normalized_modes:
        if mode == "dense":
            continue
        mode_specs = [spec for spec in specs if spec.mode == mode and spec.case_id not in completed]
        if not mode_specs:
            continue
        reranker, build, memory_base = _prepare_sdk_runtime(
            cache,
            mode,
            device=device,
            int4_query=int4_query,
            pooled_binary_pool_factor=pooled_binary_pool_factor,
            signature=signature,
            checkpoints=checkpoints,
            indexes=indexes,
            serialize_indexes=serialize_indexes,
            force=force,
        )
        for spec in mode_specs:
            row = _evaluate_case(
                cache,
                selected,
                spec,
                _SdkSearcher(reranker, spec.rescore_candidates),
                build=build,
                memory_base=memory_base,
                signature=signature,
                device=device,
                top_k=top_k,
                warmup=warmup,
                latency_repeats=latency_repeats,
                throughput_batch_size=throughput_batch_size,
            )
            _atomic_write_json(checkpoints / f"{spec.case_id}.json", row)
            completed[spec.case_id] = row
        del reranker
        _release_runtime(device)

    result = {
        "schema_version": 1,
        "benchmark": "production_retrieval",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_signature": signature,
        "input": {
            "path": str(cache.path),
            "size_bytes": cache.path.stat().st_size,
            "mtime_ns": cache.path.stat().st_mtime_ns,
            "dataset": cache.name,
            "num_docs": cache.num_docs,
            "num_queries": cache.num_queries,
            "num_doc_tokens": int(cache.doc_embeddings.shape[0]),
            "dim": cache.dim,
            "metadata": cache.metadata,
        },
        "config": config,
        "selected_queries": {
            "indices": selected.tolist(),
            "ids": [cache.query_ids[int(index)] for index in selected],
        },
        "checkpoint_dir": str(checkpoints),
        "index_dir": str(indexes) if serialize_indexes else None,
        "resumed_cases": resumed_cases,
        "cases": [completed[spec.case_id] for spec in specs],
    }
    _atomic_write_json(output, result)
    return result


def deterministic_query_indices(num_queries: int, limit: int | None, seed: int) -> np.ndarray:
    if num_queries < 1:
        raise ValueError("cache must contain at least one query")
    if limit is None or limit == 0 or limit >= num_queries:
        return np.arange(num_queries, dtype=np.int64)
    if limit < 1:
        raise ValueError("query_limit must be positive, zero, or None")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_queries, size=limit, replace=False)).astype(np.int64)


def _prepare_sdk_runtime(
    cache: RetrievalCache,
    mode: str,
    *,
    device: str,
    int4_query: str,
    pooled_binary_pool_factor: int,
    signature: str,
    checkpoints: Path,
    indexes: Path,
    serialize_indexes: bool,
    force: bool,
):
    build_checkpoint = checkpoints / "indexes" / f"{mode}.json"
    index_path = indexes / f"{mode}.maxsim.npz"
    saved = None if force else _read_build_checkpoint(build_checkpoint, signature, mode, index_path)
    before_gpu = _process_gpu_bytes()
    _reset_torch_peak(device)
    if saved is None:
        start = time.perf_counter()
        index_options = (
            {"pool_factor": int(pooled_binary_pool_factor)}
            if mode == "pooled_binary"
            else {}
        )
        corpus = maxsim.Index.from_embeddings(
            cache.doc_ids,
            cache.doc_embeddings,
            cache.doc_offsets,
            mode=mode,
            metadata={"dataset": cache.name, "source_cache": str(cache.path)},
            **index_options,
        )
        packing_ms = _elapsed_ms(start)
        cpu_report = corpus.memory_report()
        serialization_ms = None
        serialized_bytes = None
        if serialize_indexes:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            corpus.save(index_path)
            serialization_ms = _elapsed_ms(start)
            serialized_bytes = index_path.stat().st_size
        saved = {
            "schema_version": 1,
            "status": "index_ready",
            "run_signature": signature,
            "mode": mode,
            "index_path": str(index_path) if serialize_indexes else None,
            "packing_ms": packing_ms,
            "serialization_ms": serialization_ms,
            "encoded_bytes": cpu_report.encoded_bytes,
            "serialized_bytes": serialized_bytes,
            "host_array_bytes_before_device": cpu_report.host_array_bytes,
            "index_metadata": dict(corpus.metadata or {}),
        }
        _atomic_write_json(build_checkpoint, saved)
    else:
        corpus = None

    load_ms = None
    index_reused = corpus is None
    if serialize_indexes:
        del corpus
        gc.collect()
        start = time.perf_counter()
        corpus = maxsim.Index.load(index_path)
        load_ms = _elapsed_ms(start)
    elif corpus is None:
        raise RuntimeError("cannot resume an unserialized index build; rerun with --force")

    start = time.perf_counter()
    reranker = maxsim.Reranker.from_corpus(corpus, device=device, int4_query=int4_query)
    _synchronize(device)
    device_load_ms = _elapsed_ms(start)
    del corpus
    gc.collect()
    report = reranker.corpus.memory_report(index_path if serialize_indexes else None)
    workspace = report.device_workspace_bytes
    resident = report.device_index_bytes + (0 if workspace is None else workspace)
    build = {
        "packing_ms": saved["packing_ms"],
        "serialization_ms": saved["serialization_ms"],
        "load_ms": load_ms,
        "device_load_ms": device_load_ms,
        "index_reused": index_reused,
        "encoded_bytes": report.encoded_bytes,
        "serialized_bytes": report.serialized_bytes,
        "host_array_bytes_before_device": saved["host_array_bytes_before_device"],
        "index_metadata": dict(saved.get("index_metadata", {})),
    }
    memory = {
        "device_index_bytes": report.device_index_bytes,
        "device_workspace_bytes": workspace,
        "resident_gpu_bytes": resident,
        "process_gpu_bytes_before_load": before_gpu,
        "process_gpu_bytes_after_load": _process_gpu_bytes(),
    }
    return reranker, build, memory


def _evaluate_case(
    cache: RetrievalCache,
    selected: np.ndarray,
    spec: _CaseSpec,
    searcher,
    *,
    build: dict[str, Any],
    memory_base: dict[str, Any],
    signature: str,
    device: str,
    top_k: int,
    warmup: int,
    latency_repeats: int,
    throughput_batch_size: int,
) -> dict[str, Any]:
    memory = dict(memory_base)
    actual_k = min(top_k, cache.num_docs)
    first_query = cache.query(int(selected[0]))
    for _ in range(warmup):
        searcher.search(first_query, actual_k)
    _synchronize(device)
    _refresh_runtime_memory(memory, searcher)
    memory_samples = [
        memory.get("process_gpu_bytes_before_load"),
        memory.get("process_gpu_bytes_after_load"),
        _process_gpu_bytes(),
    ]

    doc_positions = {doc_id: index for index, doc_id in enumerate(cache.doc_ids)}
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    rankings: list[list[int]] = []
    ranking_scores: list[list[float]] = []
    latency_samples: list[list[float]] = []
    for query_index in selected:
        query = cache.query(int(query_index))
        result_for_metrics = None
        query_samples = []
        for _ in range(latency_repeats):
            _synchronize(device)
            start = time.perf_counter()
            results = searcher.search(query, actual_k)
            _synchronize(device)
            query_samples.append(_elapsed_ms(start))
            if result_for_metrics is None:
                result_for_metrics = results
        indices = [doc_positions[result.doc_id] for result in result_for_metrics]
        scores = [float(result.score) for result in result_for_metrics]
        ndcg, recall = _quality_at_k(cache.qrels[int(query_index)], indices, actual_k)
        ndcg_values.append(ndcg)
        recall_values.append(recall)
        rankings.append(indices)
        ranking_scores.append(scores)
        latency_samples.append(query_samples)
    memory_samples.append(_process_gpu_bytes())

    throughput_indices = selected[: min(len(selected), throughput_batch_size)]
    throughput_queries = [cache.query(int(index)) for index in throughput_indices]
    throughput_batch = _padded_query_batch(throughput_queries)
    searcher.search(throughput_batch, actual_k)
    _synchronize(device)
    start = time.perf_counter()
    searcher.search(throughput_batch, actual_k)
    _synchronize(device)
    throughput_ms = _elapsed_ms(start)
    _refresh_runtime_memory(memory, searcher)
    memory_samples.append(_process_gpu_bytes())

    flat_latency = [sample for row in latency_samples for sample in row]
    observed = [value for value in memory_samples if value is not None]
    torch_peak = _torch_peak_bytes(device)
    return {
        "schema_version": 1,
        "status": "complete",
        "run_signature": signature,
        "case_id": spec.case_id,
        "mode": spec.mode,
        "implementation": searcher.implementation,
        "uses_public_sdk": searcher.uses_public_sdk,
        "rescore_candidates": spec.rescore_candidates,
        "device": device,
        "query_count": len(selected),
        "top_k": actual_k,
        "query_indices": selected.tolist(),
        "quality": {
            "metric_k": actual_k,
            "ndcg_at_10_per_query": ndcg_values,
            "recall_at_10_per_query": recall_values,
            "ndcg_at_10_mean": float(np.mean(ndcg_values)),
            "recall_at_10_mean": float(np.mean(recall_values)),
        },
        "rankings": {
            "doc_indices": rankings,
            "scores": ranking_scores,
        },
        "latency": {
            "batch_size": 1,
            "warmup": warmup,
            "repeats_per_query": latency_repeats,
            "samples_ms_by_query": latency_samples,
            "samples_ms": flat_latency,
            "p50_ms": float(np.percentile(flat_latency, 50)),
            "p95_ms": float(np.percentile(flat_latency, 95)),
        },
        "throughput": {
            "batch_size": len(throughput_indices),
            "padded_query_tokens": int(throughput_batch.shape[1]),
            "latency_ms": throughput_ms,
            "queries_per_second": float(len(throughput_indices) * 1000.0 / throughput_ms),
        },
        "storage": dict(build),
        "memory": {
            **memory,
            "process_gpu_bytes_samples": memory_samples,
            "peak_observed_process_gpu_bytes": max(observed) if observed else None,
            "peak_observed_process_gpu_method": (
                "nvidia-smi point samples; transient allocations may be missed" if observed else "unavailable"
            ),
            "torch_peak_allocated_bytes": torch_peak,
        },
    }


def _refresh_runtime_memory(memory: dict[str, Any], searcher) -> None:
    reporter = getattr(searcher, "memory_report", None)
    if reporter is None:
        return
    report = reporter()
    previous_workspace = memory.get("device_workspace_bytes")
    current_workspace = report.device_workspace_bytes
    if previous_workspace is None:
        workspace = current_workspace
    elif current_workspace is None:
        workspace = previous_workspace
    else:
        workspace = max(int(previous_workspace), int(current_workspace))
    memory["device_index_bytes"] = report.device_index_bytes
    memory["device_workspace_bytes"] = workspace
    memory["resident_gpu_bytes"] = report.device_index_bytes + (0 if workspace is None else workspace)


def _quality_at_k(qrels: np.ndarray, ranking: Sequence[int], k: int) -> tuple[float, float]:
    gains = np.asarray([qrels[int(index)] for index in ranking[:k]], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, gains.shape[0] + 2, dtype=np.float64))
    dcg = float(np.sum(np.expm1(np.log(2.0) * gains) * discounts))
    relevance = np.asarray(qrels, dtype=np.float64)
    if k < relevance.shape[0]:
        ideal = np.partition(relevance, relevance.shape[0] - k)[-k:]
        ideal = np.sort(ideal)[::-1]
    else:
        ideal = np.sort(relevance)[::-1]
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal.shape[0] + 2, dtype=np.float64))
    idcg = float(np.sum(np.expm1(np.log(2.0) * ideal) * ideal_discounts))
    relevant = int(np.count_nonzero(qrels > 0))
    retrieved_relevant = int(np.count_nonzero(gains > 0))
    return (0.0 if idcg == 0.0 else dcg / idcg, 0.0 if relevant == 0 else retrieved_relevant / relevant)


def _flatten_queries(data) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(data["query_embeddings"]).astype(np.float32, copy=False)
    if queries.ndim == 3:
        batch, tokens, dim = queries.shape
        return (
            np.ascontiguousarray(queries.reshape(batch * tokens, dim), dtype=np.float32),
            np.arange(0, (batch + 1) * tokens, tokens, dtype=np.int64),
        )
    if queries.ndim != 2 or "query_offsets" not in data.files:
        raise ValueError("query_embeddings must be 3-D, or 2-D with query_offsets")
    return np.ascontiguousarray(queries, dtype=np.float32), _integer_array(data["query_offsets"], "query_offsets")


def _load_qrels(data, num_queries: int, num_docs: int) -> np.ndarray:
    if "qrels" in data.files:
        qrels = np.asarray(data["qrels"])
        if not (np.issubdtype(qrels.dtype, np.number) or np.issubdtype(qrels.dtype, np.bool_)):
            raise ValueError("qrels must be numeric")
        qrels = qrels.astype(np.float32, copy=False)
    elif "relevant_doc_ids" in data.files:
        relevant = _integer_array(data["relevant_doc_ids"], "relevant_doc_ids")
        if relevant.shape != (num_queries,):
            raise ValueError("relevant_doc_ids must have shape [num_queries]")
        if np.any(relevant < 0) or np.any(relevant >= num_docs):
            raise ValueError("relevant_doc_ids contains an index outside the corpus")
        qrels = np.zeros((num_queries, num_docs), dtype=np.float32)
        qrels[np.arange(num_queries), relevant] = 1.0
    else:
        raise ValueError("embedding cache must contain qrels or relevant_doc_ids")
    if qrels.shape != (num_queries, num_docs):
        raise ValueError(f"qrels must have shape [{num_queries}, {num_docs}]")
    return qrels


def _integer_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return np.ascontiguousarray(array, dtype=np.int64)


def _load_ids(data, key: str, count: int, prefix: str) -> tuple[str, ...]:
    if key not in data.files:
        return tuple(f"{prefix}-{index}" for index in range(count))
    values = np.asarray(data[key])
    if values.ndim != 1 or values.shape[0] != count:
        raise ValueError(f"{key} must have shape [{count}]")
    return tuple(str(value) for value in values)


def _load_scalar(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar")
    return str(value.reshape(-1)[0])


def _load_cache_metadata(data) -> dict[str, Any]:
    keys = (
        "builder_schema_version",
        "dataset_id",
        "model_name",
        "model_requested",
        "model_resolved",
        "model_profile",
        "model_options_json",
        "model_fallback_used",
        "source_dataset",
        "source_config",
        "source_split",
    )
    metadata = {}
    for key in keys:
        if key not in data.files:
            continue
        value = np.asarray(data[key])
        if value.size == 1:
            scalar = value.reshape(-1)[0]
            scalar = scalar.item() if isinstance(scalar, np.generic) else scalar
            metadata[key] = scalar.decode() if isinstance(scalar, bytes) else scalar
    return metadata


def _validate_cache(docs, doc_offsets, queries, query_offsets, qrels) -> None:
    if docs.ndim != 2 or docs.shape[1] < 1 or docs.shape[1] % 8:
        raise ValueError("doc_embeddings must have shape [tokens, dim] with dim divisible by 8")
    if doc_offsets.ndim != 1 or doc_offsets.shape[0] < 2:
        raise ValueError("doc_offsets must have shape [num_docs + 1]")
    if int(doc_offsets[0]) != 0 or int(doc_offsets[-1]) != docs.shape[0] or np.any(np.diff(doc_offsets) < 0):
        raise ValueError("doc_offsets must monotonically span doc_embeddings")
    if queries.ndim != 2 or queries.shape[1] != docs.shape[1]:
        raise ValueError("query embedding dim must match document embedding dim")
    if query_offsets.ndim != 1 or query_offsets.shape[0] < 2:
        raise ValueError("query_offsets must have shape [num_queries + 1]")
    if int(query_offsets[0]) != 0 or int(query_offsets[-1]) != queries.shape[0] or np.any(np.diff(query_offsets) < 0):
        raise ValueError("query_offsets must monotonically span query_embeddings")
    if qrels.shape != (query_offsets.shape[0] - 1, doc_offsets.shape[0] - 1):
        raise ValueError("qrels shape does not match query and document counts")
    if not np.all(np.isfinite(docs)) or not np.all(np.isfinite(queries)) or not np.all(np.isfinite(qrels)):
        raise ValueError("cache arrays must contain only finite values")
    if np.any(qrels < 0):
        raise ValueError("qrels must be nonnegative")


def _normalize_modes(modes: Sequence[str] | str) -> tuple[str, ...]:
    values = modes.split(",") if isinstance(modes, str) else modes
    normalized = []
    for raw in values:
        value = _MODE_ALIASES.get(str(raw).strip(), str(raw).strip())
        if value and value not in normalized:
            normalized.append(value)
    unknown = sorted(set(normalized) - ({"dense"} | _COMPRESSED_MODES))
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    if not normalized:
        raise ValueError("at least one mode is required")
    return tuple(normalized)


def _validate_run_config(
    cache: RetrievalCache,
    *,
    device: str,
    modes: Sequence[str],
    top_k: int,
    query_limit: int | None,
    warmup: int,
    latency_repeats: int,
    throughput_batch_size: int,
    cascade_candidates: Sequence[int],
    int4_query: str,
    pooled_binary_pool_factor: int,
    dense_chunk_tokens: int,
) -> None:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if top_k < 1 or warmup < 0 or latency_repeats < 1 or throughput_batch_size < 1:
        raise ValueError("top_k, latency_repeats, and throughput_batch_size must be positive; warmup may be zero")
    if query_limit is not None and query_limit < 0:
        raise ValueError("query_limit must be nonnegative or None")
    if int4_query not in {"fp32", "int8"}:
        raise ValueError("int4_query must be 'fp32' or 'int8'")
    if isinstance(pooled_binary_pool_factor, bool) or not isinstance(
        pooled_binary_pool_factor, (int, np.integer)
    ):
        raise ValueError("pooled_binary_pool_factor must be an integer >= 1")
    if int(pooled_binary_pool_factor) < 1:
        raise ValueError("pooled_binary_pool_factor must be an integer >= 1")
    if dense_chunk_tokens < 1:
        raise ValueError("dense_chunk_tokens must be positive")
    if cascade_candidates and "int4_residual" not in modes:
        raise ValueError("cascade candidates require mode 'int4_residual'")
    actual_k = min(top_k, cache.num_docs)
    if any(value < actual_k or value > cache.num_docs for value in cascade_candidates):
        raise ValueError(f"cascade candidates must satisfy {actual_k} <= M <= {cache.num_docs}")


def _case_specs(modes: Sequence[str], candidates: Sequence[int]) -> list[_CaseSpec]:
    specs = []
    for mode in modes:
        specs.append(_CaseSpec(mode, mode))
        if mode == "int4_residual":
            specs.extend(
                _CaseSpec(f"int4_residual_cascade_m{value}", mode, value)
                for value in candidates
            )
    return specs


def _padded_query_batch(queries: Sequence[np.ndarray]) -> np.ndarray:
    max_tokens = max(int(query.shape[0]) for query in queries)
    batch = np.zeros((len(queries), max_tokens, queries[0].shape[1]), dtype=np.float32)
    for index, query in enumerate(queries):
        batch[index, : query.shape[0]] = query
    return batch


def _document_chunks(offsets: np.ndarray, max_tokens: int) -> Iterable[tuple[int, int]]:
    num_docs = offsets.shape[0] - 1
    start = 0
    while start < num_docs:
        target = int(offsets[start]) + max_tokens
        end = int(np.searchsorted(offsets, target, side="right") - 1)
        end = min(num_docs, max(start + 1, end))
        yield start, end
        start = end


def _run_signature(cache_path: Path, config: dict[str, Any], selected: np.ndarray) -> str:
    stat = cache_path.stat()
    payload = {
        "input_path": str(cache_path),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "config": config,
        "selected_queries": selected.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_complete_checkpoint(path: Path, signature: str, case_id: str):
    data = _read_json(path)
    if data is None:
        return None
    if data.get("status") == "complete" and data.get("run_signature") == signature and data.get("case_id") == case_id:
        return data
    return None


def _read_build_checkpoint(path: Path, signature: str, mode: str, index_path: Path):
    data = _read_json(path)
    if data is None:
        return None
    if (
        data.get("status") == "index_ready"
        and data.get("run_signature") == signature
        and data.get("mode") == mode
        and data.get("index_path") == str(index_path)
        and index_path.is_file()
    ):
        return data
    return None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _process_gpu_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    total_mib = 0
    found = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            if int(fields[0]) == os.getpid():
                total_mib += int(fields[1])
                found = True
        except ValueError:
            continue
    return total_mib * 1024 * 1024 if found else None


def _synchronize(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _reset_torch_peak(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _torch_peak_bytes(device: str) -> int | None:
    if device != "cuda":
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except ImportError:
        return None


def _release_runtime(device: str) -> None:
    gc.collect()
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Ragged retrieval embedding .npz")
    parser.add_argument("--output", required=True, type=Path, help="Aggregate result JSON")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=0, help="0 evaluates every query")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--throughput-batch-size", type=int, default=8)
    parser.add_argument("--cascade-candidates", default="", help="Comma-separated residual candidate budgets")
    parser.add_argument("--int4-query", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--pooled-binary-pool-factor", type=int, default=3)
    parser.add_argument("--dense-chunk-tokens", type=int, default=262_144)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--no-serialize-indexes", action="store_false", dest="serialize_indexes")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(serialize_indexes=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidate_values = tuple(
        int(value.strip()) for value in args.cascade_candidates.split(",") if value.strip()
    )
    result = run_production_benchmark(
        args.input,
        args.output,
        device=args.device,
        modes=args.modes,
        top_k=args.top_k,
        query_limit=None if args.query_limit == 0 else args.query_limit,
        seed=args.seed,
        warmup=args.warmup,
        latency_repeats=args.latency_repeats,
        throughput_batch_size=args.throughput_batch_size,
        cascade_candidates=candidate_values,
        int4_query=args.int4_query,
        pooled_binary_pool_factor=args.pooled_binary_pool_factor,
        dense_chunk_tokens=args.dense_chunk_tokens,
        serialize_indexes=args.serialize_indexes,
        checkpoint_dir=args.checkpoint_dir,
        index_dir=args.index_dir,
        force=args.force,
    )
    print(json.dumps({"output": str(args.output), "cases": [row["case_id"] for row in result["cases"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
