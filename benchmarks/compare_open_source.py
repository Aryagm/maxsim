from __future__ import annotations

import argparse
import json
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
    _print_table,
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
    _print_table(rows, args.k)
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
    return row


if __name__ == "__main__":
    main()
