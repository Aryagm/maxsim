"""CPU sweep: can pooled-corpus accuracy be bought back?

Arms: ward mean pooling (current pooled_binary), norm-weighted cluster means,
adaptive threshold pooling (merge only tokens with average cosine similarity
above a threshold), each optionally packed as binary or int4.
"""

from __future__ import annotations

import argparse
import json
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


def _cluster_labels(tokens: np.ndarray, *, factor: int | None, sim_threshold: float | None) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage

    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    normalized = tokens / np.maximum(norms, 1e-12)
    if factor is not None:
        import math

        target = max(1, math.ceil(tokens.shape[0] / factor))
        if tokens.shape[0] <= target:
            return np.arange(1, tokens.shape[0] + 1)
        tree = linkage(normalized, method="ward")
        return fcluster(tree, t=target, criterion="maxclust")
    tree = linkage(normalized, method="average", metric="cosine")
    return fcluster(tree, t=1.0 - sim_threshold, criterion="distance")


def _pool(dataset: RetrievalEmbeddings, *, factor=None, sim_threshold=None, norm_weighted=False):
    docs = dataset.doc_embeddings
    offsets = dataset.doc_offsets
    chunks = []
    pooled_offsets = [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        tokens = docs[int(start) : int(end)]
        if tokens.shape[0] == 0:
            pooled_offsets.append(pooled_offsets[-1])
            continue
        if tokens.shape[0] == 1:
            labels = np.array([1])
        else:
            labels = _cluster_labels(tokens, factor=factor, sim_threshold=sim_threshold)
        count = int(labels.max())
        pooled = np.empty((count, tokens.shape[1]), dtype=np.float32)
        for cluster_id in range(1, count + 1):
            members = tokens[labels == cluster_id]
            if norm_weighted and members.shape[0] > 1:
                weights = np.linalg.norm(members, axis=1)
                weights = weights / max(weights.sum(), 1e-12)
                pooled[cluster_id - 1] = (members * weights[:, np.newaxis]).sum(axis=0, dtype=np.float64)
            else:
                pooled[cluster_id - 1] = members.mean(axis=0, dtype=np.float64)
        chunks.append(pooled)
        pooled_offsets.append(pooled_offsets[-1] + count)
    pooled_docs = np.concatenate(chunks, axis=0) if chunks else np.empty((0, docs.shape[1]), dtype=np.float32)
    return RetrievalEmbeddings(
        name=dataset.name,
        query_embeddings=dataset.query_embeddings,
        doc_embeddings=pooled_docs,
        doc_offsets=np.asarray(pooled_offsets, dtype=np.int64),
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )


def _binary_scores(pooled: RetrievalEmbeddings) -> np.ndarray:
    signs = np.where(pooled.doc_embeddings >= 0, 1.0, -1.0).astype(np.float32)
    return _dense_scores_with_docs(pooled, signs)


def _int4_scores(pooled: RetrievalEmbeddings) -> np.ndarray:
    docs = pooled.doc_embeddings
    max_abs = float(np.max(np.abs(docs))) if docs.size else 0.0
    scale = 1.0 if max_abs == 0.0 else max_abs / 7.0
    values = np.clip(np.rint(docs / scale), -7, 7).astype(np.float32) * scale
    return _dense_scores_with_docs(pooled, values)


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
    original_tokens = dataset.doc_embeddings.shape[0]
    fp32_bytes = original_tokens * dataset.dim * 4

    arms = [
        ("pool2_ward_mean", {"factor": 2}, "binary"),
        ("pool2_norm_weighted", {"factor": 2, "norm_weighted": True}, "binary"),
        ("pool3_norm_weighted", {"factor": 3, "norm_weighted": True}, "binary"),
        ("thresh_sim090", {"sim_threshold": 0.90}, "binary"),
        ("thresh_sim085", {"sim_threshold": 0.85}, "binary"),
        ("thresh_sim080", {"sim_threshold": 0.80}, "binary"),
        ("thresh_sim085_norm_weighted", {"sim_threshold": 0.85, "norm_weighted": True}, "binary"),
        ("pool2_int4", {"factor": 2}, "int4"),
        ("pool2_norm_weighted_int4", {"factor": 2, "norm_weighted": True}, "int4"),
    ]

    rows = []
    for name, pool_kwargs, packing in arms:
        pooled = _pool(dataset, **pool_kwargs)
        pooled_tokens = int(pooled.doc_offsets[-1])
        scores = _binary_scores(pooled) if packing == "binary" else _int4_scores(pooled)
        metrics = _ranking_metrics(scores, dataset.qrels, k=k)
        per_token_bytes = dataset.dim // 8 if packing == "binary" else dataset.dim // 2
        storage = pooled_tokens * per_token_bytes
        rows.append(
            {
                "arm": name,
                "packing": packing,
                "ndcg_at_k": metrics["ndcg_at_k"],
                "delta_vs_dense": metrics["ndcg_at_k"] - dense_ndcg,
                "doc_storage_bytes": storage,
                "compression_vs_fp32": fp32_bytes / storage,
                "realized_token_ratio": original_tokens / max(pooled_tokens, 1),
                "per_query_ndcg": _per_query_ndcg(scores, dataset.qrels, k=k),
            }
        )
        print(
            f"{name:30s} ndcg@{k}={metrics['ndcg_at_k']:.4f} d_dense={rows[-1]['delta_vs_dense']:+.4f} "
            f"comp={rows[-1]['compression_vs_fp32']:5.1f}x ratio={rows[-1]['realized_token_ratio']:.2f}x"
        )

    payload = {
        "input": str(args.input),
        "dense_ndcg_at_k": dense_ndcg,
        "top_k": k,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
