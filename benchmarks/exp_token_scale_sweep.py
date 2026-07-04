"""CPU quality sweep for token-scale refinements and salience-aware pooling.

Pack-time-only questions answered on cached embeddings without a GPU:
- alpha-tempered token scales (sigma^alpha) with optional quantile clipping
- token scales combined with zero-threshold dim-centroid calibration
- salience-protected pooling (keep top-norm tokens unpooled, pool the rest)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.run_retrieval import (
    RetrievalEmbeddings,
    _load_embedding_file,
    _numpy_dense_fp16_scores,
    _per_query_ndcg,
    _ranking_metrics,
    _dense_scores_with_docs,
)
from maxsim.experimental import fit_dim_centroid_calibration


def _token_scales(docs: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)


def _tempered(scales: np.ndarray, alpha: float, clip: tuple[float, float] | None) -> np.ndarray:
    values = scales.astype(np.float64)
    if clip is not None:
        lo, hi = np.percentile(values, clip)
        values = np.clip(values, lo, hi)
    return np.power(values, alpha).astype(np.float32)


def _scaled_sign_scores(dataset, token_scales, *, query_weights=None) -> np.ndarray:
    signs = np.where(dataset.doc_embeddings >= 0, 1.0, -1.0).astype(np.float32)
    docs = signs if token_scales is None else signs * token_scales[:, np.newaxis]
    if query_weights is None:
        return _dense_scores_with_docs(dataset, docs)
    weighted = RetrievalEmbeddings(
        name=dataset.name,
        query_embeddings=tuple(np.ascontiguousarray(q * query_weights, dtype=np.float32) for q in dataset.query_embeddings),
        doc_embeddings=dataset.doc_embeddings,
        doc_offsets=dataset.doc_offsets,
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )
    return _dense_scores_with_docs(weighted, docs)


def _salient_pooled(dataset, protect_fraction: float, factor: int):
    """Pool low-norm tokens per doc at `factor`, keep top-norm tokens intact."""
    from benchmarks.pooling import pool_doc_tokens

    docs = dataset.doc_embeddings
    offsets = dataset.doc_offsets
    pooled_chunks = []
    pooled_offsets = [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        tokens = docs[int(start) : int(end)]
        if tokens.shape[0] == 0:
            pooled_offsets.append(pooled_offsets[-1])
            continue
        norms = np.linalg.norm(tokens, axis=1)
        keep = max(1, int(round(tokens.shape[0] * protect_fraction)))
        order = np.argsort(-norms, kind="stable")
        salient = tokens[np.sort(order[:keep])]
        rest = tokens[np.sort(order[keep:])]
        if rest.shape[0] > 1:
            rest_pooled, _ = pool_doc_tokens(rest, np.array([0, rest.shape[0]], dtype=np.int64), factor)
        else:
            rest_pooled = rest
        merged = np.concatenate([salient, rest_pooled], axis=0)
        pooled_chunks.append(merged.astype(np.float32))
        pooled_offsets.append(pooled_offsets[-1] + merged.shape[0])
    pooled_docs = np.concatenate(pooled_chunks, axis=0) if pooled_chunks else np.empty((0, docs.shape[1]), dtype=np.float32)
    pooled = RetrievalEmbeddings(
        name=dataset.name,
        query_embeddings=dataset.query_embeddings,
        doc_embeddings=pooled_docs,
        doc_offsets=np.asarray(pooled_offsets, dtype=np.int64),
        qrels=dataset.qrels,
        query_ids=dataset.query_ids,
        doc_ids=dataset.doc_ids,
    )
    return pooled


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
    scales = _token_scales(dataset.doc_embeddings)
    calibration = fit_dim_centroid_calibration(dataset.doc_embeddings)
    weights = calibration.positive_centroids - calibration.negative_centroids

    packed_bytes = dataset.doc_embeddings.shape[0] * (dataset.dim // 8)

    arms: dict[str, dict] = {
        "binary": {"scales": None},
        "token_scale": {"scales": scales},
        "ts_alpha_075": {"scales": _tempered(scales, 0.75, None)},
        "ts_alpha_050": {"scales": _tempered(scales, 0.5, None)},
        "ts_alpha_025": {"scales": _tempered(scales, 0.25, None)},
        "ts_clip_q05_q95": {"scales": _tempered(scales, 1.0, (5, 95))},
        "ts_clip_q10_q90": {"scales": _tempered(scales, 1.0, (10, 90))},
        "ts_alpha_050_clip_q05_q95": {"scales": _tempered(scales, 0.5, (5, 95))},
        "centroid_zero": {"scales": None, "weights": weights},
        "centroid_zero_token_scale": {"scales": scales, "weights": weights},
    }

    rows = []
    per_query_cache: dict[str, list] = {}
    for name, spec in arms.items():
        scores = _scaled_sign_scores(dataset, spec.get("scales"), query_weights=spec.get("weights"))
        metrics = _ranking_metrics(scores, dataset.qrels, k=k)
        per_query_cache[name] = _per_query_ndcg(scores, dataset.qrels, k=k)
        rows.append(
            {
                "arm": name,
                "ndcg_at_k": metrics["ndcg_at_k"],
                "delta_vs_dense": metrics["ndcg_at_k"] - dense_ndcg,
                "doc_storage_bytes": packed_bytes + (0 if spec.get("scales") is None else dataset.doc_embeddings.shape[0] * 2),
                "per_query_ndcg": per_query_cache[name],
            }
        )

    for protect, factor in ((0.2, 3), (0.3, 3), (0.2, 4), (0.3, 4)):
        pooled = _salient_pooled(dataset, protect, factor)
        pooled_scales = _token_scales(pooled.doc_embeddings)
        for scale_arm, use_scales in (("", None), ("_ts", pooled_scales)):
            scores = _scaled_sign_scores(pooled, use_scales)
            metrics = _ranking_metrics(scores, dataset.qrels, k=k)
            name = f"salient_p{int(protect*100)}_f{factor}{scale_arm}"
            pooled_tokens = int(pooled.doc_offsets[-1])
            rows.append(
                {
                    "arm": name,
                    "ndcg_at_k": metrics["ndcg_at_k"],
                    "delta_vs_dense": metrics["ndcg_at_k"] - dense_ndcg,
                    "doc_storage_bytes": pooled_tokens * (dataset.dim // 8) + (0 if use_scales is None else pooled_tokens * 2),
                    "realized_token_ratio": dataset.doc_embeddings.shape[0] / max(pooled_tokens, 1),
                    "per_query_ndcg": _per_query_ndcg(scores, dataset.qrels, k=k),
                }
            )

    payload = {
        "input": str(args.input),
        "dense_ndcg_at_k": dense_ndcg,
        "top_k": k,
        "query_count": dataset.num_queries,
        "docs": dataset.num_docs,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    fp32_bytes = dataset.doc_embeddings.shape[0] * dataset.dim * 4
    for row in rows:
        comp = fp32_bytes / row["doc_storage_bytes"]
        extra = f" ratio={row['realized_token_ratio']:.2f}x" if "realized_token_ratio" in row else ""
        print(f"{row['arm']:28s} ndcg@{k}={row['ndcg_at_k']:.4f} d_dense={row['delta_vs_dense']:+.4f} comp={comp:5.1f}x{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
