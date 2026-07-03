"""Token pooling (ColPali-style hierarchical pooling).

Pools each document's token embeddings down to ceil(tokens / factor) clusters
by hierarchical clustering on cosine similarity, mean-pooling each cluster.
Deterministic given the input. Requires scipy (install bitmax[pooling]).
Measured tradeoff on ViDoRe docvqa limit256 (docs/gpu_optimization.md):
factor 2 + binary packing = 63.9x fp32 compression at NDCG@10 -0.018 vs dense.
"""

from __future__ import annotations

import math

import numpy as np


def pool_doc_tokens(
    doc_embeddings: np.ndarray,
    doc_offsets: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pooled_embeddings, pooled_offsets) with per-doc token clustering."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return doc_embeddings.copy(), doc_offsets.copy()

    from scipy.cluster.hierarchy import fcluster, linkage

    pooled_chunks: list[np.ndarray] = []
    pooled_offsets = [0]
    for start, end in zip(doc_offsets[:-1], doc_offsets[1:]):
        tokens = doc_embeddings[int(start) : int(end)]
        count = tokens.shape[0]
        if count == 0:
            pooled_offsets.append(pooled_offsets[-1])
            continue
        target = max(1, math.ceil(count / factor))
        if count <= target:
            pooled_chunks.append(tokens.astype(np.float32, copy=True))
            pooled_offsets.append(pooled_offsets[-1] + count)
            continue
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        normalized = tokens / np.maximum(norms, 1e-12)
        tree = linkage(normalized, method="ward")
        labels = fcluster(tree, t=target, criterion="maxclust")
        pooled = np.empty((int(labels.max()), tokens.shape[1]), dtype=np.float32)
        for cluster_id in range(1, int(labels.max()) + 1):
            pooled[cluster_id - 1] = tokens[labels == cluster_id].mean(axis=0, dtype=np.float64)
        pooled_chunks.append(pooled)
        pooled_offsets.append(pooled_offsets[-1] + pooled.shape[0])

    if pooled_chunks:
        pooled_embeddings = np.concatenate(pooled_chunks, axis=0).astype(np.float32)
    else:
        pooled_embeddings = np.empty((0, doc_embeddings.shape[1]), dtype=np.float32)
    return pooled_embeddings, np.asarray(pooled_offsets, dtype=np.int64)
