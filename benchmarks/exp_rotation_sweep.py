"""CPU sweep: ITQ-style learned rotation before sign binarization.

Orthogonal rotations preserve dot products, so MaxSim semantics are unchanged,
but sign() quantization error depends on axis orientation. ITQ alternates
B = sign(X R) with the orthogonal Procrustes solution for R to minimize
||B - X R||^2. Optional mean-centering is ranking-neutral for MaxSim (the q.c
term shifts all docs of a query token equally). Storage cost: one 128x128
fp32 matrix (+ centering vector) per corpus; packed bits unchanged.

Arms: {binary, token_scale_fp16, pool2_binary} x {identity, itq, centered itq}.
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


def fit_itq_rotation(docs: np.ndarray, *, iterations: int = 40, sample: int = 65536, seed: int = 163) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if docs.shape[0] > sample:
        idx = rng.choice(docs.shape[0], size=sample, replace=False)
        x = docs[idx].astype(np.float64)
    else:
        x = docs.astype(np.float64)
    dim = x.shape[1]
    r = np.linalg.qr(rng.standard_normal((dim, dim)))[0]
    for _ in range(iterations):
        b = np.sign(x @ r)
        b[b == 0] = 1.0
        u, _, vt = np.linalg.svd(b.T @ x, full_matrices=False)
        r = (u @ vt).T
    return np.ascontiguousarray(r, dtype=np.float32)


def _apply(dataset: RetrievalEmbeddings, rotation: np.ndarray | None, center: np.ndarray | None):
    docs = dataset.doc_embeddings
    queries = dataset.query_embeddings
    if center is not None:
        docs = docs - center[np.newaxis, :]
    if rotation is not None:
        docs = docs @ rotation
        queries = tuple(np.ascontiguousarray(q @ rotation, dtype=np.float32) for q in queries)
    return RetrievalEmbeddings(
        name=dataset.name,
        query_embeddings=queries,
        doc_embeddings=np.ascontiguousarray(docs, dtype=np.float32),
        doc_offsets=dataset.doc_offsets,
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )


def _pool2(dataset: RetrievalEmbeddings) -> RetrievalEmbeddings:
    from bitmax.pooling import pool_doc_tokens

    pooled_docs, pooled_offsets = pool_doc_tokens(dataset.doc_embeddings, dataset.doc_offsets, 2)
    return RetrievalEmbeddings(
        name=dataset.name,
        query_embeddings=dataset.query_embeddings,
        doc_embeddings=pooled_docs,
        doc_offsets=pooled_offsets,
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )


def _binary_scores(prepared: RetrievalEmbeddings, *, token_scale: bool) -> np.ndarray:
    signs = np.where(prepared.doc_embeddings >= 0, 1.0, -1.0).astype(np.float32)
    if token_scale:
        scales = np.mean(np.abs(prepared.doc_embeddings), axis=1, dtype=np.float64).astype(np.float32)
        scales = scales.astype(np.float16).astype(np.float32)
        signs = signs * scales[:, np.newaxis]
    return _dense_scores_with_docs(prepared, signs)


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

    rotation = fit_itq_rotation(dataset.doc_embeddings)
    center = np.mean(dataset.doc_embeddings, axis=0, dtype=np.float64).astype(np.float32)
    centered_rotation = fit_itq_rotation(dataset.doc_embeddings - center[np.newaxis, :])

    transforms = {
        "identity": (None, None),
        "itq": (rotation, None),
        "itq_centered": (centered_rotation, center),
    }
    original_tokens = dataset.doc_embeddings.shape[0]
    fp32_bytes = original_tokens * dataset.dim * 4
    rotation_overhead = dataset.dim * dataset.dim * 4

    rows = []
    for tname, (rot, cen) in transforms.items():
        prepared = _apply(dataset, rot, cen)
        overhead = 0 if rot is None else rotation_overhead + (0 if cen is None else dataset.dim * 4)
        for family, token_scale, pooled in (
            ("binary", False, False),
            ("token_scale_fp16", True, False),
            ("pool2_binary", False, True),
        ):
            target = _pool2(prepared) if pooled else prepared
            scores = _binary_scores(target, token_scale=token_scale)
            metrics = _ranking_metrics(scores, dataset.qrels, k=k)
            tokens = int(target.doc_offsets[-1])
            storage = tokens * (dataset.dim // 8) + (tokens * 2 if token_scale else 0) + overhead
            name = f"{family}__{tname}"
            rows.append(
                {
                    "arm": name,
                    "ndcg_at_k": metrics["ndcg_at_k"],
                    "delta_vs_dense": metrics["ndcg_at_k"] - dense_ndcg,
                    "doc_storage_bytes": storage,
                    "compression_vs_fp32": fp32_bytes / storage,
                    "per_query_ndcg": _per_query_ndcg(scores, dataset.qrels, k=k),
                }
            )
            print(
                f"{name:34s} ndcg@{k}={metrics['ndcg_at_k']:.4f} d_dense={rows[-1]['delta_vs_dense']:+.4f} "
                f"comp={rows[-1]['compression_vs_fp32']:5.1f}x"
            )

    payload = {"input": str(args.input), "dense_ndcg_at_k": dense_ndcg, "top_k": k, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
