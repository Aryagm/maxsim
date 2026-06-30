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
except ImportError:  # pragma: no cover - optional CUDA baseline dependency
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
        scores, latency = _time_call(
            lambda: _scores_from_search(reranker, dataset["query_embeddings"], dataset["doc_ids"], args.k),
            repeat=args.repeat,
        )
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


def _result_row(
    implementation,
    dataset,
    scores,
    latency_ms,
    k,
    *,
    doc_storage_bytes,
    dense_scores,
    baseline_latency=None,
    mode=None,
    device=None,
):
    effective_k = min(k, len(dataset["doc_ids"]))
    metrics = _ranking_metrics(scores, dataset["qrels"], k=effective_k)
    dense_metrics = _ranking_metrics(dense_scores, dataset["qrels"], k=effective_k)
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
