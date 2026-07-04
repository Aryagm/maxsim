from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import maxsim

try:
    import faiss as _PRELOADED_FAISS

    _FAISS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional OSS benchmark install
    _PRELOADED_FAISS = None
    _FAISS_IMPORT_ERROR = exc

from examples.local_multivector_search import (
    _dense_doc_bytes,
    _dense_fp16_scores,
    _load_demo_dataset,
    _ranking_metrics,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional CUDA baseline dependency
    torch = None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Compare maxsim against popular open-source retrieval baselines.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--implementations",
        default="dense_fp16,faiss_pooled,faiss_token_dense_rerank,bitmax_binary,bitmax_binary_q40,bitmax_int4",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric-ks", default="1,5,10", help="Comma-separated ranking cutoffs to report, always including --k.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--faiss-token-topn", type=int, default=512)
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--fast-plaid-nbits", type=int, default=4)
    parser.add_argument("--fast-plaid-n-full-scores", type=int, default=4096)
    parser.add_argument("--fast-plaid-n-ivf-probe", type=int, default=4)
    args = parser.parse_args(argv)

    dataset = _load_demo_dataset(Path(args.input), limit_queries=args.limit_queries)
    implementations = tuple(value.strip() for value in args.implementations.split(",") if value.strip())
    metric_ks = _parse_metric_ks(args.metric_ks, default_k=args.k)
    score_k = min(max(metric_ks), len(dataset["doc_ids"]))
    rows = []
    dense_scores = None
    dense_latency = None

    for implementation in implementations:
        if implementation == "dense_fp16":
            dense_scores, dense_latency, dense_latency_stats = _time_call(lambda: _dense_fp16_scores(dataset, args.device), repeat=args.repeat)
            _release_cuda_cache()
            rows.append(
                _result_row(
                    "dense_fp16_baseline",
                    "dense_cuda" if args.device == "cuda" else "dense_cpu",
                    dataset,
                    dense_scores,
                    dense_latency,
                    args.k,
                    metric_ks=metric_ks,
                    doc_storage_bytes=_dense_doc_bytes(dataset, 2),
                    dense_scores=dense_scores,
                    latency_stats=dense_latency_stats,
                    metadata={"library": "torch", "storage_dtype": "fp16", "device": args.device},
                )
            )
            continue

        if implementation == "dense_fp16_vectorized":
            vec_scores, vec_latency, vec_latency_stats = _time_call(lambda: _dense_fp16_scores_vectorized(dataset, args.device), repeat=args.repeat)
            _release_cuda_cache()
            if dense_scores is None:
                dense_scores, dense_latency = vec_scores, vec_latency
            rows.append(
                _result_row(
                    "dense_fp16_vectorized",
                    "dense_cuda_vectorized" if args.device == "cuda" else "dense_cpu_vectorized",
                    dataset,
                    vec_scores,
                    vec_latency,
                    args.k,
                    metric_ks=metric_ks,
                    doc_storage_bytes=_dense_doc_bytes(dataset, 2),
                    dense_scores=dense_scores if dense_scores is not None else vec_scores,
                    latency_stats=vec_latency_stats,
                    metadata={"library": "torch", "storage_dtype": "fp16", "device": args.device,
                              "formula": "flat_matmul_plus_segment_amax"},
                )
            )
            continue

        if dense_scores is None:
            dense_scores, dense_latency, _dense_latency_stats = _time_call(lambda: _dense_fp16_scores(dataset, args.device), repeat=args.repeat)
            _release_cuda_cache()

        if implementation.startswith("bitmax_"):
            mode = implementation.removeprefix("bitmax_")
            int4_query = "fp32"
            pool_factor = 2
            if mode == "int4_dp4a":
                mode = "int4"
                int4_query = "int8"
            if mode == "pooled_binary3":
                mode = "pooled_binary"
                pool_factor = 3
            corpus = maxsim.Corpus.from_embeddings(
                dataset["doc_ids"], dataset["doc_embeddings"], dataset["doc_offsets"], mode=mode, pool_factor=pool_factor
            )
            reranker = maxsim.Reranker.from_corpus(corpus, device=args.device, int4_query=int4_query)
            scores, latency, latency_stats = _time_call(lambda: _scores_from_sdk(reranker, dataset["query_embeddings"], dataset["doc_ids"], score_k), repeat=args.repeat)
            rows.append(
                _result_row(
                    implementation,
                    "bitmax_sdk",
                    dataset,
                    scores,
                    latency,
                    args.k,
                    metric_ks=metric_ks,
                    doc_storage_bytes=corpus.storage_bytes,
                    dense_scores=dense_scores,
                    baseline_latency=dense_latency,
                    latency_stats=latency_stats,
                    metadata={"mode": mode, "device": args.device},
                )
            )
            continue

        if implementation == "faiss_pooled":
            scores, latency, latency_stats, storage_bytes, metadata = _faiss_pooled_scores(dataset, score_k, args.device, repeat=args.repeat)
            rows.append(
                _result_row(
                    "faiss_gpu_mean_pool_flat_ip" if args.device == "cuda" else "faiss_cpu_mean_pool_flat_ip",
                    "open_source_faiss",
                    dataset,
                    scores,
                    latency,
                    args.k,
                    metric_ks=metric_ks,
                    doc_storage_bytes=storage_bytes,
                    dense_scores=dense_scores,
                    baseline_latency=dense_latency,
                    latency_stats=latency_stats,
                    metadata=metadata,
                )
            )
            continue

        if implementation == "faiss_token_dense_rerank":
            scores, latency, latency_stats, storage_bytes, metadata = _faiss_token_dense_rerank_scores(
                dataset,
                score_k,
                args.device,
                token_topn=args.faiss_token_topn,
                repeat=args.repeat,
            )
            rows.append(
                _result_row(
                    "faiss_gpu_token_candidates_dense_rerank" if args.device == "cuda" else "faiss_cpu_token_candidates_dense_rerank",
                    "open_source_faiss",
                    dataset,
                    scores,
                    latency,
                    args.k,
                    metric_ks=metric_ks,
                    doc_storage_bytes=storage_bytes,
                    dense_scores=dense_scores,
                    baseline_latency=dense_latency,
                    latency_stats=latency_stats,
                    metadata=metadata,
                )
            )
            continue

        if implementation == "qdrant_multivector":
            row = _optional_measured_row(
                implementation,
                "open_source_qdrant",
                dataset,
                dense_scores,
                dense_latency,
                args.k,
                metric_ks,
                args.allow_unavailable,
                lambda: _qdrant_multivector_scores(dataset, score_k, repeat=args.repeat),
            )
            rows.append(row)
            continue

        if implementation == "cuvs_pooled":
            row = _optional_measured_row(
                "cuvs_gpu_mean_pool_flat_ip" if args.device == "cuda" else "cuvs_cpu_mean_pool_flat_ip",
                "open_source_cuvs",
                dataset,
                dense_scores,
                dense_latency,
                args.k,
                metric_ks,
                args.allow_unavailable,
                lambda: _cuvs_pooled_scores(dataset, score_k, args.device, repeat=args.repeat),
            )
            rows.append(row)
            continue

        if implementation == "fast_plaid":
            row = _optional_measured_row(
                "fast_plaid",
                "open_source_fast_plaid",
                dataset,
                dense_scores,
                dense_latency,
                args.k,
                metric_ks,
                args.allow_unavailable,
                lambda: _fast_plaid_scores(
                    dataset,
                    score_k,
                    args.device,
                    repeat=args.repeat,
                    nbits=args.fast_plaid_nbits,
                    n_full_scores=args.fast_plaid_n_full_scores,
                    n_ivf_probe=args.fast_plaid_n_ivf_probe,
                ),
            )
            rows.append(row)
            continue

        if implementation == "colbert_plaid":
            rows.append(
                _status_row(
                    "colbert_plaid",
                    "open_source_colbert",
                    dataset,
                    "not_applicable_to_embedding_slice",
                    "ColBERT/PLAID indexes ColBERT model outputs through its own text/index pipeline; this harness uses precomputed ColQwen2 token embeddings.",
                    metadata={"library": "colbert-ai", "comparison_note": "use fast_plaid for the embedding-array PLAID-style competitor in this benchmark"},
                )
            )
            continue

        if implementation == "vespa_multivector":
            rows.append(
                _status_row(
                    "vespa_multivector",
                    "open_source_vespa",
                    dataset,
                    "requires_service_benchmark",
                    "Vespa multivector comparison requires a running Vespa service/schema and ingestion path; this local CUDA harness is embedding-array only.",
                    metadata={"library": "vespa", "comparison_note": "service benchmark should be run separately with the same embeddings and qrels"},
                )
            )
            continue

        raise ValueError(f"unknown implementation: {implementation}")

    result = {
        "schema_version": 1,
        "benchmark": "open_source_comparison",
        "dataset": {
            "name": dataset["name"],
            "queries": int(len(dataset["query_embeddings"])),
            "docs": int(len(dataset["doc_ids"])),
            "dim": int(dataset["doc_embeddings"].shape[1]),
            "doc_tokens": int(dataset["doc_embeddings"].shape[0]),
            "doc_token_count_summary": _doc_token_count_summary(dataset),
        },
        "device": args.device,
        "top_k": int(args.k),
        "metric_ks": [int(value) for value in metric_ks],
        "repeat": int(args.repeat),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print_comparison_table(rows, args.k)
    return result


def _scores_from_sdk(reranker, queries, doc_ids, k: int) -> np.ndarray:
    scores = np.full((len(queries), len(doc_ids)), -np.inf, dtype=np.float32)
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    for query_idx, query in enumerate(queries):
        results = reranker.search(query, k=min(k, len(doc_ids)))
        for result in results:
            scores[query_idx, doc_index[result.doc_id]] = result.score
    return scores


def _faiss_pooled_scores(dataset, k: int, device: str, *, repeat: int):
    faiss = _import_faiss()
    doc_vectors = _mean_pool_docs(dataset).astype(np.float32, copy=False)
    query_vectors = np.stack([query.mean(axis=0, dtype=np.float64).astype(np.float32) for query in dataset["query_embeddings"]], axis=0)
    index, resources = _build_faiss_index(faiss, doc_vectors, device)
    actual_k = min(k, len(dataset["doc_ids"]))

    def run():
        distances, indices = index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), actual_k)
        _sync_cuda(resources)
        return _scores_from_faiss(indices, distances, len(dataset["doc_ids"]))

    scores, latency, latency_stats = _time_call(run, repeat=repeat)
    metadata = {
        "library": "faiss",
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "formula": "mean_pool_doc_query_index_flat_ip",
        "device": device,
    }
    return scores, latency, latency_stats, int(doc_vectors.nbytes), metadata


def _faiss_token_dense_rerank_scores(dataset, k: int, device: str, *, token_topn: int, repeat: int):
    if device != "cuda":
        raise RuntimeError("faiss_token_dense_rerank requires --device cuda for this benchmark")
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("faiss_token_dense_rerank requires torch with CUDA")
    faiss = _import_faiss()
    doc_tokens = np.ascontiguousarray(dataset["doc_embeddings"], dtype=np.float32)
    token_doc_ids = _token_doc_ids(dataset["doc_offsets"])
    index, resources = _build_faiss_index(faiss, doc_tokens, device)
    docs_tensor = torch.as_tensor(dataset["doc_embeddings"], dtype=torch.float16, device="cuda").to(torch.float32)
    actual_k = min(k, len(dataset["doc_ids"]))
    actual_token_topn = min(int(token_topn), doc_tokens.shape[0])

    def run():
        rows = []
        for query in dataset["query_embeddings"]:
            distances, token_indices = index.search(np.ascontiguousarray(query, dtype=np.float32), actual_token_topn)
            candidate_doc_ids = np.unique(token_doc_ids[token_indices.reshape(-1)])
            row = _dense_candidate_scores_torch(query, docs_tensor, dataset["doc_offsets"], candidate_doc_ids, len(dataset["doc_ids"]))
            rows.append(row)
        _sync_cuda(resources)
        return np.stack(rows, axis=0)

    scores, latency, latency_stats = _time_call(run, repeat=repeat)
    metadata = {
        "library": "faiss+torch",
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "formula": "faiss_token_flat_ip_candidates_then_dense_fp16_maxsim_rerank",
        "device": device,
        "token_topn": int(actual_token_topn),
        "mean_candidates_per_query": float(np.mean([np.isfinite(row).sum() for row in scores])),
    }
    storage_bytes = int(doc_tokens.nbytes + _dense_doc_bytes(dataset, 2))
    return scores, latency, latency_stats, storage_bytes, metadata


def _qdrant_multivector_scores(dataset, k: int, *, repeat: int):
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - depends on optional benchmark install
        raise RuntimeError("qdrant_multivector requires qdrant-client") from exc

    collection_name = "bitmax_compare"
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(
            size=int(dataset["doc_embeddings"].shape[1]),
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
        ),
    )
    points = []
    for doc_idx, (start, end) in enumerate(zip(dataset["doc_offsets"][:-1], dataset["doc_offsets"][1:])):
        points.append(
            models.PointStruct(
                id=int(doc_idx),
                vector=np.ascontiguousarray(dataset["doc_embeddings"][int(start) : int(end)], dtype=np.float32).tolist(),
            )
        )
    client.upsert(collection_name, points=points)
    actual_k = min(k, len(dataset["doc_ids"]))

    def run():
        rows = []
        for query in dataset["query_embeddings"]:
            response = client.query_points(
                collection_name,
                query=np.ascontiguousarray(query, dtype=np.float32).tolist(),
                limit=actual_k,
                with_payload=False,
                with_vectors=False,
            )
            row = np.full((len(dataset["doc_ids"]),), -np.inf, dtype=np.float32)
            for point in response.points:
                row[int(point.id)] = float(point.score)
            rows.append(row)
        return np.stack(rows, axis=0)

    scores, latency, latency_stats = _time_call(run, repeat=repeat)
    metadata = {
        "library": "qdrant-client",
        "formula": "in_memory_multivector_dot_maxsim",
        "device": "cpu",
        "status": "ok",
    }
    return scores, latency, latency_stats, _dense_doc_bytes(dataset, 4), metadata


def _cuvs_pooled_scores(dataset, k: int, device: str, *, repeat: int):
    if device != "cuda":
        raise RuntimeError("cuvs_pooled requires --device cuda")
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("cuvs_pooled requires torch with CUDA")
    try:
        from cuvs.neighbors import brute_force
    except ImportError as exc:  # pragma: no cover - depends on optional benchmark install
        raise RuntimeError("cuvs_pooled requires cuvs-cu12") from exc

    doc_vectors = _mean_pool_docs(dataset).astype(np.float32, copy=False)
    query_vectors = np.stack([query.mean(axis=0, dtype=np.float64).astype(np.float32) for query in dataset["query_embeddings"]], axis=0)
    doc_tensor = torch.as_tensor(np.ascontiguousarray(doc_vectors, dtype=np.float32), device="cuda")
    query_tensor = torch.as_tensor(np.ascontiguousarray(query_vectors, dtype=np.float32), device="cuda")
    index = brute_force.build(doc_tensor, metric="inner_product")
    actual_k = min(k, len(dataset["doc_ids"]))

    def run():
        distances, indices = brute_force.search(index, query_tensor, actual_k)
        _sync_cuda(None)
        return _scores_from_faiss(indices.copy_to_host(), distances.copy_to_host(), len(dataset["doc_ids"]))

    scores, latency, latency_stats = _time_call(run, repeat=repeat)
    metadata = {
        "library": "cuvs",
        "formula": "mean_pool_doc_query_bruteforce_inner_product",
        "device": "cuda",
        "status": "ok",
    }
    return scores, latency, latency_stats, int(doc_vectors.nbytes), metadata


def _fast_plaid_scores(
    dataset,
    k: int,
    device: str,
    *,
    repeat: int,
    nbits: int,
    n_full_scores: int,
    n_ivf_probe: int,
):
    if torch is None:
        raise RuntimeError("fast_plaid requires torch")
    try:
        from fast_plaid.search import FastPlaid
    except ImportError as exc:  # pragma: no cover - depends on optional benchmark install
        raise RuntimeError("fast_plaid requires fast-plaid") from exc

    docs = [
        torch.as_tensor(
            np.ascontiguousarray(dataset["doc_embeddings"][int(start) : int(end)], dtype=np.float32),
            device=device if device == "cuda" and torch.cuda.is_available() else "cpu",
        )
        for start, end in zip(dataset["doc_offsets"][:-1], dataset["doc_offsets"][1:])
    ]
    queries = [
        torch.as_tensor(
            np.ascontiguousarray(query, dtype=np.float32),
            device=device if device == "cuda" and torch.cuda.is_available() else "cpu",
        )
        for query in dataset["query_embeddings"]
    ]
    index_dir = tempfile.mkdtemp(prefix="maxsim-fast-plaid-")
    actual_k = min(k, len(dataset["doc_ids"]))
    actual_n_full_scores = min(int(n_full_scores), len(dataset["doc_ids"]))
    search_device = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"
    try:
        searcher = FastPlaid(index_dir, device=search_device)
        build_start = time.perf_counter()
        searcher.create(
            docs,
            nbits=int(nbits),
            metadata=[{"doc_id": doc_id} for doc_id in dataset["doc_ids"]],
            start_from_scratch=0,
            use_triton_kmeans=None,
        )
        build_ms = (time.perf_counter() - build_start) * 1_000.0

        def run():
            raw_rows = searcher.search(
                queries,
                top_k=actual_k,
                n_full_scores=actual_n_full_scores,
                n_ivf_probe=int(n_ivf_probe),
                show_progress=False,
            )
            _sync_cuda(None)
            rows = []
            for raw in raw_rows:
                row = np.full((len(dataset["doc_ids"]),), -np.inf, dtype=np.float32)
                for doc_idx, score in raw:
                    row[int(doc_idx)] = float(score)
                rows.append(row)
            return np.stack(rows, axis=0)

        scores, latency, latency_stats = _time_call(run, repeat=repeat)
        metadata = {
            "library": "fast-plaid",
            "formula": "plaid_index_search",
            "device": search_device,
            "status": "ok",
            "nbits": int(nbits),
            "n_full_scores": actual_n_full_scores,
            "n_ivf_probe": int(n_ivf_probe),
            "build_ms": float(build_ms),
        }
        return scores, latency, latency_stats, _directory_size(index_dir), metadata
    finally:
        shutil.rmtree(index_dir, ignore_errors=True)


def _dense_candidate_scores_torch(query: np.ndarray, docs_tensor, doc_offsets: np.ndarray, candidate_doc_ids: np.ndarray, num_docs: int) -> np.ndarray:
    query_tensor = torch.as_tensor(query, dtype=torch.float16, device="cuda").to(torch.float32)
    row = np.full((num_docs,), -np.inf, dtype=np.float32)
    values = []
    for doc_idx in candidate_doc_ids:
        start = int(doc_offsets[int(doc_idx)])
        end = int(doc_offsets[int(doc_idx) + 1])
        doc = docs_tensor[start:end]
        values.append((int(doc_idx), (query_tensor @ doc.T).max(dim=1).values.sum()))
    if values:
        torch.cuda.synchronize()
        for doc_idx, score in values:
            row[doc_idx] = float(score.detach().cpu().item())
    return row


def _mean_pool_docs(dataset) -> np.ndarray:
    rows = []
    for start, end in zip(dataset["doc_offsets"][:-1], dataset["doc_offsets"][1:]):
        rows.append(dataset["doc_embeddings"][int(start) : int(end)].mean(axis=0, dtype=np.float64).astype(np.float32))
    return np.stack(rows, axis=0)


def _token_doc_ids(doc_offsets: np.ndarray) -> np.ndarray:
    values = np.empty((int(doc_offsets[-1]),), dtype=np.int64)
    for doc_idx, (start, end) in enumerate(zip(doc_offsets[:-1], doc_offsets[1:])):
        values[int(start) : int(end)] = int(doc_idx)
    return values


def _build_faiss_index(faiss, vectors: np.ndarray, device: str):
    cpu_index = faiss.IndexFlatIP(vectors.shape[1])
    resources = None
    if device == "cuda":
        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
            raise RuntimeError("FAISS GPU benchmark requested but no FAISS GPU is available")
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    else:
        index = cpu_index
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    _sync_cuda(resources)
    return index, resources


def _scores_from_faiss(indices: np.ndarray, distances: np.ndarray, num_docs: int) -> np.ndarray:
    scores = np.full((indices.shape[0], num_docs), -np.inf, dtype=np.float32)
    for row_idx in range(indices.shape[0]):
        for score, doc_idx in zip(distances[row_idx], indices[row_idx]):
            if int(doc_idx) >= 0:
                scores[row_idx, int(doc_idx)] = float(score)
    return scores


def _optional_measured_row(
    implementation,
    implementation_kind,
    dataset,
    dense_scores,
    dense_latency,
    k,
    metric_ks,
    allow_unavailable: bool,
    fn,
):
    try:
        scores, latency, latency_stats, storage_bytes, metadata = fn()
    except Exception as exc:
        if not allow_unavailable:
            raise
        return _status_row(
            implementation,
            implementation_kind,
            dataset,
            "unavailable",
            f"{type(exc).__name__}: {exc}",
        )
    return _result_row(
        implementation,
        implementation_kind,
        dataset,
        scores,
        latency,
        k,
        metric_ks=metric_ks,
        doc_storage_bytes=storage_bytes,
        dense_scores=dense_scores,
        baseline_latency=dense_latency,
        latency_stats=latency_stats,
        metadata=metadata,
    )


def _status_row(implementation, implementation_kind, dataset, status, reason, *, metadata=None):
    row = {
        "implementation": implementation,
        "implementation_kind": implementation_kind,
        "status": status,
        "reason": reason,
        "query_count": int(len(dataset["query_embeddings"])),
        "docs": int(len(dataset["doc_ids"])),
    }
    if metadata is not None:
        row.update(metadata)
    return row


def _directory_size(path: str) -> int:
    total = 0
    for file_path in Path(path).rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return int(total)


def _parse_metric_ks(value: str, *, default_k: int) -> tuple[int, ...]:
    values = {1, int(default_k)}
    for raw in value.split(","):
        item = raw.strip()
        if item:
            parsed = int(item)
            if parsed <= 0:
                raise ValueError("--metric-ks values must be positive")
            values.add(parsed)
    return tuple(sorted(values))


def _time_call(fn, *, repeat: int):
    best_latency = float("inf")
    best_value = None
    samples = []
    for _ in range(max(int(repeat), 1)):
        start = time.perf_counter()
        value = fn()
        _sync_cuda(None)
        latency = (time.perf_counter() - start) * 1_000.0
        samples.append(float(latency))
        if latency < best_latency:
            best_latency = latency
            best_value = value
    return best_value, best_latency, _latency_stats(samples)


def _latency_stats(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "latency_samples_ms": [float(value) for value in samples],
        "latency_mean_ms": float(np.mean(values)),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
    }


def _sync_cuda(resources) -> None:
    if resources is not None and hasattr(resources, "syncDefaultStreamCurrentDevice"):
        resources.syncDefaultStreamCurrentDevice()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _dense_fp16_scores_vectorized(dataset, device: str) -> "np.ndarray":
    """Honest dense fp16 MaxSim: one flat matmul per query + segment-amax.

    Replaces the per-document Python loop with [q_tokens, total_tokens] fp16
    matmuls and a scatter amax over document segments; numerically the same
    scores (fp16 dot precision) at vectorized speed. Empty documents score 0.
    """
    import torch

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    docs = torch.as_tensor(dataset["doc_embeddings"], dtype=torch.float16, device=torch_device)
    offsets = np.asarray(dataset["doc_offsets"], dtype=np.int64)
    num_docs = offsets.shape[0] - 1
    lengths = np.diff(offsets)
    doc_index = torch.as_tensor(np.repeat(np.arange(num_docs, dtype=np.int64), lengths), device=torch_device)
    scores = np.empty((len(dataset["query_embeddings"]), num_docs), dtype=np.float32)
    neg_inf = torch.finfo(torch.float32).min
    for query_idx, query in enumerate(dataset["query_embeddings"]):
        q = torch.as_tensor(np.ascontiguousarray(query), dtype=torch.float16, device=torch_device)
        dots = (q @ docs.T).to(torch.float32)
        seg = torch.full((q.shape[0], num_docs), neg_inf, dtype=torch.float32, device=torch_device)
        seg.scatter_reduce_(1, doc_index.expand(q.shape[0], -1), dots, reduce="amax")
        seg[seg == neg_inf] = 0.0
        scores[query_idx] = seg.sum(dim=0).cpu().numpy()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    return scores




def _release_cuda_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _import_faiss():
    if _PRELOADED_FAISS is not None:
        return _PRELOADED_FAISS
    raise RuntimeError("FAISS benchmark requires a working faiss-gpu or faiss-cpu install") from _FAISS_IMPORT_ERROR


def _doc_token_count_summary(dataset) -> dict[str, float | int]:
    counts = np.diff(np.asarray(dataset["doc_offsets"], dtype=np.int64)).astype(np.float64)
    if counts.size == 0:
        return {"min": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "min": int(np.min(counts)),
        "mean": float(np.mean(counts)),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(np.max(counts)),
    }


def _token_bucket_quality(scores: np.ndarray, qrels: np.ndarray, doc_offsets: np.ndarray, *, k: int, dense_scores: np.ndarray | None) -> dict[str, dict[str, Any]]:
    doc_token_counts = np.diff(np.asarray(doc_offsets, dtype=np.int64)).astype(np.float64)
    if doc_token_counts.size == 0:
        return {}
    lower, upper = np.percentile(doc_token_counts, [33.3333333333, 66.6666666667])
    bucket_indices: dict[str, list[int]] = {"short": [], "medium": [], "long": []}
    bucket_lengths: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
    for query_idx, relevance in enumerate(qrels):
        relevant_lengths = doc_token_counts[np.asarray(relevance) > 0]
        if relevant_lengths.size == 0:
            continue
        mean_relevant_length = float(np.mean(relevant_lengths))
        if mean_relevant_length <= lower:
            bucket = "short"
        elif mean_relevant_length <= upper:
            bucket = "medium"
        else:
            bucket = "long"
        bucket_indices[bucket].append(int(query_idx))
        bucket_lengths[bucket].append(mean_relevant_length)

    result: dict[str, dict[str, Any]] = {}
    effective_k = min(int(k), scores.shape[1])
    for bucket, indices in bucket_indices.items():
        lengths = bucket_lengths[bucket]
        if not indices:
            result[bucket] = {
                "queries": 0,
                "mean_relevant_doc_tokens": None,
                f"recall_at_{k}": None,
                f"mrr_at_{k}": None,
                f"ndcg_at_{k}": None,
                f"quality_delta_vs_dense_ndcg_at_{k}": None,
            }
            continue
        subset_scores = scores[indices]
        subset_qrels = qrels[indices]
        metrics = _ranking_metrics(subset_scores, subset_qrels, k=effective_k)
        item = {
            "queries": int(len(indices)),
            "mean_relevant_doc_tokens": float(np.mean(lengths)),
            "min_relevant_doc_tokens": float(np.min(lengths)),
            "max_relevant_doc_tokens": float(np.max(lengths)),
            f"recall_at_{k}": float(metrics["recall_at_k"]),
            f"mrr_at_{k}": float(metrics["mrr_at_k"]),
            f"ndcg_at_{k}": float(metrics["ndcg_at_k"]),
        }
        if dense_scores is not None:
            dense_metrics = _ranking_metrics(dense_scores[indices], subset_qrels, k=effective_k)
            item[f"quality_delta_vs_dense_ndcg_at_{k}"] = float(metrics["ndcg_at_k"] - dense_metrics["ndcg_at_k"])
        result[bucket] = item
    return result


def _result_row(
    implementation,
    implementation_kind,
    dataset,
    scores,
    latency_ms,
    k,
    *,
    metric_ks,
    doc_storage_bytes,
    dense_scores,
    baseline_latency=None,
    latency_stats=None,
    metadata=None,
):
    row = {
        "implementation": implementation,
        "implementation_kind": implementation_kind,
        "latency_ms": float(latency_ms),
        "query_count": int(len(dataset["query_embeddings"])),
        "docs": int(len(dataset["doc_ids"])),
        "doc_storage_bytes": int(doc_storage_bytes),
        "doc_memory_compression_vs_fp16": float(_dense_doc_bytes(dataset, 2) / max(doc_storage_bytes, 1)),
        "doc_memory_compression_vs_fp32": float(_dense_doc_bytes(dataset, 4) / max(doc_storage_bytes, 1)),
    }
    row.update(latency_stats or _latency_stats([float(latency_ms)]))
    for metric_k in metric_ks:
        effective_k = min(int(metric_k), len(dataset["doc_ids"]))
        metrics = _ranking_metrics(scores, dataset["qrels"], k=effective_k)
        dense_metrics = _ranking_metrics(dense_scores, dataset["qrels"], k=effective_k)
        row[f"recall_at_{metric_k}"] = float(metrics["recall_at_k"])
        row[f"mrr_at_{metric_k}"] = float(metrics["mrr_at_k"])
        row[f"ndcg_at_{metric_k}"] = float(metrics["ndcg_at_k"])
        row[f"quality_delta_vs_dense_ndcg_at_{metric_k}"] = float(metrics["ndcg_at_k"] - dense_metrics["ndcg_at_k"])
        row[f"token_bucket_quality_at_{metric_k}"] = _token_bucket_quality(
            scores,
            dataset["qrels"],
            dataset["doc_offsets"],
            k=int(metric_k),
            dense_scores=dense_scores,
        )
    if baseline_latency is not None:
        row["speedup_vs_dense_fp16"] = float(baseline_latency / max(latency_ms, 1e-12))
    if metadata is not None:
        row.update(metadata)
    row.setdefault("status", "ok")
    return row


def _print_comparison_table(rows, k: int) -> None:
    print("implementation, status, latency_ms, fp32_reduction, recall@1, recall@%d, mrr@%d, ndcg@%d" % (k, k, k))
    for row in rows:
        if row.get("status", "ok") != "ok" or "latency_ms" not in row:
            print(f"{row['implementation']}, {row.get('status', 'unknown')}, n/a, n/a, n/a, n/a, n/a, n/a")
            continue
        print(
            f"{row['implementation']}, {row.get('status', 'ok')}, {row['latency_ms']:.3f}, "
            f"{row['doc_memory_compression_vs_fp32']:.2f}x, "
            f"{row['recall_at_1']:.3f}, {row[f'recall_at_{k}']:.3f}, "
            f"{row[f'mrr_at_{k}']:.3f}, {row[f'ndcg_at_{k}']:.3f}"
        )


if __name__ == "__main__":
    main()
