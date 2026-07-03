"""CPU simulation of the planned int8-query x int4-doc dp4a kernel semantics.

Decides whether the dp4a kernel is worth writing: quantizes queries to int8
per token, documents to int4 (per-tensor and per-token scale arms), accumulates
integer dots, takes the per-query-token max in the same domain the kernel
would, and reports NDCG deltas plus paired per-query changes against the
fp32-query int4 reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.run_retrieval import (
    RetrievalEmbeddings,
    _load_embedding_file,
    _per_query_ndcg,
    _ranking_metrics,
    _topk_agreement,
)


def _int4_doc_values(docs: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(docs))) if docs.size else 0.0
    scale = 1.0 if max_abs == 0.0 else max_abs / 7.0
    return np.clip(np.rint(docs / scale), -7, 7).astype(np.int8), scale


def _int4_doc_values_per_token(docs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(docs), axis=1)
    scales = np.where(max_abs == 0.0, 1.0, max_abs / 7.0).astype(np.float32)
    return np.clip(np.rint(docs / scales[:, np.newaxis]), -7, 7).astype(np.int8), scales


def _int8_query(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(query), axis=1)
    scales = np.where(max_abs == 0.0, 1.0, max_abs / 127.0).astype(np.float32)
    values = np.clip(np.rint(query / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    return values, scales


def _scores(
    dataset: RetrievalEmbeddings,
    doc_values: np.ndarray,
    *,
    doc_scale,
    quantize_query: bool,
) -> np.ndarray:
    per_token_doc_scale = isinstance(doc_scale, np.ndarray)
    scores = np.empty((dataset.num_queries, dataset.num_docs), dtype=np.float32)
    # Integer dots are computed in fp32 BLAS: |dot| <= 128*127*7 < 2**24, so
    # every intermediate integer is exactly representable and this matches
    # int32 accumulation bit-for-bit while being ~100x faster than numpy int32.
    docs_work = doc_values.astype(np.float32)
    for query_idx, query in enumerate(dataset.query_embeddings):
        query_float = query.astype(np.float32, copy=False)
        if quantize_query:
            q_values, q_scales = _int8_query(query_float)
            q_work = q_values.astype(np.float32)
        else:
            q_work = query_float
            q_scales = None
        for doc_idx in range(dataset.num_docs):
            start = int(dataset.doc_offsets[doc_idx])
            end = int(dataset.doc_offsets[doc_idx + 1])
            if start == end:
                scores[query_idx, doc_idx] = 0.0
                continue
            doc = docs_work[start:end]
            dots = q_work @ doc.T
            if per_token_doc_scale:
                dots = dots.astype(np.float64) * doc_scale[start:end][np.newaxis, :]
                best = dots.max(axis=1)
                per_q = best if q_scales is None else best * q_scales
                scores[query_idx, doc_idx] = np.float32(per_q.sum())
            else:
                best = dots.max(axis=1).astype(np.float64)
                per_q = best if q_scales is None else best * q_scales
                scores[query_idx, doc_idx] = np.float32(per_q.sum() * float(doc_scale))
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dataset = _load_embedding_file(args.input)
    k = min(args.top_k, dataset.num_docs)

    docs = dataset.doc_embeddings
    int4_values, tensor_scale = _int4_doc_values(docs)
    int4_token_values, token_scales = _int4_doc_values_per_token(docs)

    arms = {
        "int4_fp32q_per_tensor": _scores(dataset, int4_values, doc_scale=tensor_scale, quantize_query=False),
        "int4_int8q_per_tensor": _scores(dataset, int4_values, doc_scale=tensor_scale, quantize_query=True),
        "int4_fp32q_per_token": _scores(dataset, int4_token_values, doc_scale=token_scales, quantize_query=False),
        "int4_int8q_per_token": _scores(dataset, int4_token_values, doc_scale=token_scales, quantize_query=True),
    }

    reference = arms["int4_fp32q_per_tensor"]
    rows = []
    for name, scores in arms.items():
        metrics = _ranking_metrics(scores, dataset.qrels, k=k)
        per_query = _per_query_ndcg(scores, dataset.qrels, k=k)
        reference_per_query = _per_query_ndcg(reference, dataset.qrels, k=k)
        paired = [
            (a or 0.0) - (b or 0.0)
            for a, b in zip(per_query, reference_per_query)
            if a is not None and b is not None
        ]
        rows.append(
            {
                "arm": name,
                "ndcg_at_k": metrics["ndcg_at_k"],
                "recall_at_k": metrics["recall_at_k"],
                "mrr_at_k": metrics["mrr_at_k"],
                "ndcg_delta_vs_fp32q_per_tensor": metrics["ndcg_at_k"]
                - _ranking_metrics(reference, dataset.qrels, k=k)["ndcg_at_k"],
                "topk_agreement_vs_fp32q_per_tensor": _topk_agreement(scores, reference, k),
                "paired_queries_changed": int(sum(1 for d in paired if abs(d) > 1e-9)),
                "paired_mean_delta": float(np.mean(paired)) if paired else 0.0,
                "per_query_ndcg": per_query,
            }
        )

    payload = {
        "input": str(args.input),
        "dataset": dataset.name,
        "top_k": k,
        "query_count": dataset.num_queries,
        "docs": dataset.num_docs,
        "int4_per_tensor_scale": tensor_scale,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    for row in rows:
        print(
            f"{row['arm']:28s} ndcg@{k}={row['ndcg_at_k']:.4f} "
            f"delta={row['ndcg_delta_vs_fp32q_per_tensor']:+.4f} "
            f"agree={row['topk_agreement_vs_fp32q_per_tensor']:.4f} "
            f"changed={row['paired_queries_changed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
