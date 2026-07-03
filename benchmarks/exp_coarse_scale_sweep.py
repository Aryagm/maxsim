"""CPU sweep: how few magnitude levels differentiate enough?

Part A (unpooled binary): quantize per-token mean-abs scales to k log-spaced
levels for k in {2, 4, 16, 256}; storage log2(k)/8 bytes per token.
Part B (pooled binary): scale definitions that account for cluster mass —
mean-abs of the pooled vector, cluster size, mean-abs x sqrt(size) — plus
4-level coarse versions of each.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.run_retrieval import (
    RetrievalEmbeddings,
    _dense_scores_with_docs,
    _load_embedding_file,
    _numpy_dense_fp16_scores,
    _per_query_ndcg,
    _ranking_metrics,
)


def _log_levels(scales: np.ndarray, k: int) -> np.ndarray:
    positive = np.maximum(scales.astype(np.float64), 1e-12)
    logs = np.log(positive)
    lo, hi = float(logs.min()), float(logs.max())
    if hi <= lo or k < 2:
        return scales.astype(np.float32)
    codes = np.clip(np.rint((logs - lo) * ((k - 1) / (hi - lo))), 0, k - 1)
    return np.exp(lo + codes * ((hi - lo) / (k - 1))).astype(np.float32)


def _scored(dataset: RetrievalEmbeddings, scales: np.ndarray | None) -> np.ndarray:
    signs = np.where(dataset.doc_embeddings >= 0, 1.0, -1.0).astype(np.float32)
    if scales is not None:
        signs = signs * scales[:, np.newaxis].astype(np.float32)
    return _dense_scores_with_docs(dataset, signs)


def _pool2_with_sizes(dataset: RetrievalEmbeddings):
    from scipy.cluster.hierarchy import fcluster, linkage

    docs = dataset.doc_embeddings
    offsets = dataset.doc_offsets
    chunks, sizes_chunks = [], []
    pooled_offsets = [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        tokens = docs[int(start) : int(end)]
        if tokens.shape[0] == 0:
            pooled_offsets.append(pooled_offsets[-1])
            continue
        target = max(1, math.ceil(tokens.shape[0] / 2))
        if tokens.shape[0] <= target:
            labels = np.arange(1, tokens.shape[0] + 1)
        else:
            norms = np.linalg.norm(tokens, axis=1, keepdims=True)
            normalized = tokens / np.maximum(norms, 1e-12)
            labels = fcluster(linkage(normalized, method="ward"), t=target, criterion="maxclust")
        count = int(labels.max())
        pooled = np.empty((count, tokens.shape[1]), dtype=np.float32)
        sizes = np.empty(count, dtype=np.float32)
        for cluster_id in range(1, count + 1):
            members = tokens[labels == cluster_id]
            pooled[cluster_id - 1] = members.mean(axis=0, dtype=np.float64)
            sizes[cluster_id - 1] = members.shape[0]
        chunks.append(pooled)
        sizes_chunks.append(sizes)
        pooled_offsets.append(pooled_offsets[-1] + count)
    pooled_docs = np.concatenate(chunks, axis=0)
    pooled_sizes = np.concatenate(sizes_chunks, axis=0)
    pooled_dataset = RetrievalEmbeddings(
        name=dataset.name,
        query_embeddings=dataset.query_embeddings,
        doc_embeddings=pooled_docs,
        doc_offsets=np.asarray(pooled_offsets, dtype=np.int64),
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )
    return pooled_dataset, pooled_sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dataset = _load_embedding_file(args.input)
    k = min(args.top_k, dataset.num_docs)
    dense = _numpy_dense_fp16_scores(dataset)
    dense_ndcg = _ranking_metrics(dense, dataset.qrels, k=k)["ndcg_at_k"]
    tokens = dataset.doc_embeddings.shape[0]
    fp32_bytes = tokens * dataset.dim * 4
    base_bytes = tokens * (dataset.dim // 8)

    scales = np.mean(np.abs(dataset.doc_embeddings), axis=1, dtype=np.float64).astype(np.float32)

    rows = []

    def record(name, scores, storage):
        metrics = _ranking_metrics(scores, dataset.qrels, k=k)
        rows.append(
            {
                "arm": name,
                "ndcg_at_k": metrics["ndcg_at_k"],
                "delta_vs_dense": metrics["ndcg_at_k"] - dense_ndcg,
                "doc_storage_bytes": int(storage),
                "compression_vs_fp32": fp32_bytes / storage,
                "per_query_ndcg": _per_query_ndcg(scores, dataset.qrels, k=k),
            }
        )
        print(
            f"{name:32s} ndcg@{k}={metrics['ndcg_at_k']:.4f} d_dense={rows[-1]['delta_vs_dense']:+.4f} "
            f"comp={rows[-1]['compression_vs_fp32']:5.1f}x"
        )

    record("binary", _scored(dataset, None), base_bytes)
    for levels in (2, 4, 16, 256):
        bits = max(1, int(math.ceil(math.log2(levels))))
        record(
            f"ts_levels_{levels}",
            _scored(dataset, _log_levels(scales, levels)),
            base_bytes + tokens * bits / 8 + 16,
        )

    pooled, sizes = _pool2_with_sizes(dataset)
    pooled_tokens = int(pooled.doc_offsets[-1])
    pooled_base = pooled_tokens * (dataset.dim // 8)
    pooled_scales = np.mean(np.abs(pooled.doc_embeddings), axis=1, dtype=np.float64).astype(np.float32)

    def pooled_record(name, pooled_scale_values, bits):
        storage = pooled_base + (pooled_tokens * bits / 8 + 16 if bits else 0)
        record(name, _scored(pooled, pooled_scale_values), storage)

    pooled_record("pool2_plain", None, 0)
    pooled_record("pool2_meanabs_fp16", pooled_scales.astype(np.float16).astype(np.float32), 16)
    pooled_record("pool2_size", sizes, 4)
    pooled_record("pool2_sqrt_size", np.sqrt(sizes), 4)
    pooled_record("pool2_meanabs_x_sqrt_size", pooled_scales * np.sqrt(sizes), 16)
    pooled_record("pool2_meanabs_x_size", pooled_scales * sizes, 16)
    pooled_record("pool2_sqrt_size_lv4", _log_levels(np.sqrt(sizes), 4), 2)
    pooled_record("pool2_meanabs_x_sqrt_size_lv4", _log_levels(pooled_scales * np.sqrt(sizes), 4), 2)

    payload = {"input": str(args.input), "dense_ndcg_at_k": dense_ndcg, "top_k": k, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
