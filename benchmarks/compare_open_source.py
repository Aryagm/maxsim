from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import bitmax

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
    parser = argparse.ArgumentParser(description="Compare bitmax against popular open-source retrieval baselines.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--implementations",
        default="dense_fp16,faiss_pooled,faiss_token_dense_rerank,bitmax_binary,bitmax_binary_q40,bitmax_int4",
    )
    parser.add_argument("--k", type=int, default=10)
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
    rows = []
    dense_scores = None
    dense_latency = None

    for implementation in implementations:
        if implementation == "dense_fp16":
            dense_scores, dense_latency = _time_call(lambda: _dense_fp16_scores(dataset, args.device), repeat=args.repeat)
            rows.append(
                _result_row(
                    "dense_fp16_baseline",
                    "dense_cuda" if args.device == "cuda" else "dense_cpu",
                    dataset,
                    dense_scores,
                    dense_latency,
                    args.k,
                    doc_storage_bytes=_dense_doc_bytes(dataset, 2),
                    dense_scores=dense_scores,
                    metadata={"library": "torch", "storage_dtype": "fp16", "device": args.device},
                )
            )
            continue

        if dense_scores is None:
            dense_scores, dense_latency = _time_call(lambda: _dense_fp16_scores(dataset, args.device), repeat=args.repeat)

        if implementation.startswith("bitmax_"):
            mode = implementation.removeprefix("bitmax_")
            corpus = bitmax.Corpus.from_embeddings(dataset["doc_ids"], dataset["doc_embeddings"], dataset["doc_offsets"], mode=mode)
            reranker = bitmax.Reranker.from_corpus(corpus, device=args.device)
            scores, latency = _time_call(lambda: _scores_from_sdk(reranker, dataset["query_embeddings"], dataset["doc_ids"], args.k), repeat=args.repeat)
            rows.append(
                _result_row(
                    implementation,
                    "bitmax_sdk",
                    dataset,
                    scores,
                    latency,
                    args.k,
                    doc_storage_bytes=corpus.storage_bytes,
                    dense_scores=dense_scores,
                    baseline_latency=dense_latency,
                    metadata={"mode": mode, "device": args.device},
                )
            )
            continue

        if implementation == "faiss_pooled":
            scores, latency, storage_bytes, metadata = _faiss_pooled_scores(dataset, args.k, args.device, repeat=args.repeat)
            rows.append(
                _result_row(
                    "faiss_gpu_mean_pool_flat_ip" if args.device == "cuda" else "faiss_cpu_mean_pool_flat_ip",
                    "open_source_faiss",
                    dataset,
                    scores,
                    latency,
                    args.k,
                    doc_storage_bytes=storage_bytes,
                    dense_scores=dense_scores,
                    baseline_latency=dense_latency,
                    metadata=metadata,
                )
            )
            continue

        if implementation == "faiss_token_dense_rerank":
            scores, latency, storage_bytes, metadata = _faiss_token_dense_rerank_scores(
                dataset,
                args.k,
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
                    doc_storage_bytes=storage_bytes,
                    dense_scores=dense_scores,
                    baseline_latency=dense_latency,
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
                args.allow_unavailable,
                lambda: _qdrant_multivector_scores(dataset, args.k, repeat=args.repeat),
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
                args.allow_unavailable,
                lambda: _cuvs_pooled_scores(dataset, args.k, args.device, repeat=args.repeat),
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
                args.allow_unavailable,
                lambda: _fast_plaid_scores(
                    dataset,
                    args.k,
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
        },
        "device": args.device,
        "top_k": int(args.k),
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

    scores, latency = _time_call(run, repeat=repeat)
    metadata = {
        "library": "faiss",
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "formula": "mean_pool_doc_query_index_flat_ip",
        "device": device,
    }
    return scores, latency, int(doc_vectors.nbytes), metadata


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

    scores, latency = _time_call(run, repeat=repeat)
    metadata = {
        "library": "faiss+torch",
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "formula": "faiss_token_flat_ip_candidates_then_dense_fp16_maxsim_rerank",
        "device": device,
        "token_topn": int(actual_token_topn),
        "mean_candidates_per_query": float(np.mean([np.isfinite(row).sum() for row in scores])),
    }
    storage_bytes = int(doc_tokens.nbytes + _dense_doc_bytes(dataset, 2))
    return scores, latency, storage_bytes, metadata


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

    scores, latency = _time_call(run, repeat=repeat)
    metadata = {
        "library": "qdrant-client",
        "formula": "in_memory_multivector_dot_maxsim",
        "device": "cpu",
        "status": "ok",
    }
    return scores, latency, _dense_doc_bytes(dataset, 4), metadata


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

    scores, latency = _time_call(run, repeat=repeat)
    metadata = {
        "library": "cuvs",
        "formula": "mean_pool_doc_query_bruteforce_inner_product",
        "device": "cuda",
        "status": "ok",
    }
    return scores, latency, int(doc_vectors.nbytes), metadata


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
    index_dir = tempfile.mkdtemp(prefix="bitmax-fast-plaid-")
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

        scores, latency = _time_call(run, repeat=repeat)
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
        return scores, latency, _directory_size(index_dir), metadata
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
    allow_unavailable: bool,
    fn,
):
    try:
        scores, latency, storage_bytes, metadata = fn()
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
        doc_storage_bytes=storage_bytes,
        dense_scores=dense_scores,
        baseline_latency=dense_latency,
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


def _time_call(fn, *, repeat: int):
    best_latency = float("inf")
    best_value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        _sync_cuda(None)
        latency = (time.perf_counter() - start) * 1_000.0
        if latency < best_latency:
            best_latency = latency
            best_value = value
    return best_value, best_latency


def _sync_cuda(resources) -> None:
    if resources is not None and hasattr(resources, "syncDefaultStreamCurrentDevice"):
        resources.syncDefaultStreamCurrentDevice()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _import_faiss():
    if _PRELOADED_FAISS is not None:
        return _PRELOADED_FAISS
    raise RuntimeError("FAISS benchmark requires a working faiss-gpu or faiss-cpu install") from _FAISS_IMPORT_ERROR


def _result_row(
    implementation,
    implementation_kind,
    dataset,
    scores,
    latency_ms,
    k,
    *,
    doc_storage_bytes,
    dense_scores,
    baseline_latency=None,
    metadata=None,
):
    effective_k = min(k, len(dataset["doc_ids"]))
    metrics = _ranking_metrics(scores, dataset["qrels"], k=effective_k)
    dense_metrics = _ranking_metrics(dense_scores, dataset["qrels"], k=effective_k)
    row = {
        "implementation": implementation,
        "implementation_kind": implementation_kind,
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
