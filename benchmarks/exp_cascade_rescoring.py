"""Cascade rescoring over an embedding cache: coarse compressed scan -> exact rescore.

Scans the whole corpus with a cheap per-token format, keeps the top-m
candidates per query, and rescores only those with a higher-fidelity tier.
The cascade is evaluated with a masking identity: computing the rescore
tier's per-doc scores for all documents and masking everything outside the
coarse top-m to -inf ranks identically to rescoring only the m survivors
(ties at -inf never enter the top-10 for m >= 10).

The headline configuration is the single-index EMBEDDED RESIDUAL code:
store per-token int4 (66 B/token at dim 128) plus an int4-quantized residual
(66 B more). The scan reads only the int4 prefix; the rescore reconstructs
prefix + residual (int8-class fidelity, 132 B/token total, ~3.9x vs fp32) —
dense-quality retrieval with no dense vectors stored anywhere. Measured on
BEIR (GTE-ModernColBERT): SciFact 0.7587 vs dense 0.7608 at m=100,
FiQA-57k 0.4529 vs 0.4536 embedded ceiling / 0.4556 dense at m=200
(0.35% of the corpus; the required fraction SHRINKS with corpus size).

    python -m benchmarks.exp_cascade_rescoring caches-full/beir/beir-scifact-gte-moderncolbert.npz
    python -m benchmarks.exp_cascade_rescoring caches-full/beir/beir-fiqa-gte-moderncolbert.npz \
        --coarse int4_pertoken --m 50 --m 100 --m 200 --m 400 --m 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.exp_text_scale_inversion import ndcg10

# nominal stored bytes per 128-dim token (fp16 scales included where per-token)
BYTES_PER_TOKEN = {
    "dense": 512.0,  # fp32 reference frame for compression factors
    "binary": 16.0,
    "int4_pertoken": 66.0,
    "int8_pertoken": 130.0,
    "embedded": 132.0,  # int4 prefix + int4 residual + two fp16 scales
    "pool3_binary": 16.0 / 3.0,
}


def _sign(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0, -1.0).astype(np.float32)


def _int4_pertoken(x: np.ndarray) -> np.ndarray:
    s = np.abs(x).max(axis=1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    return (np.clip(np.round(x / s), -8, 7) * s).astype(np.float32)


def _int8_pertoken(x: np.ndarray) -> np.ndarray:
    s = np.abs(x).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return (np.clip(np.round(x / s), -128, 127) * s).astype(np.float32)


def _embedded(x: np.ndarray) -> np.ndarray:
    """int4 prefix + int4-quantized residual: the rescore-side reconstruction."""
    prefix = _int4_pertoken(x)
    r = x - prefix
    return (prefix + _int4_pertoken(r)).astype(np.float32)


TRANSFORMS = {
    "dense": lambda x: x,
    "binary": _sign,
    "int4_pertoken": _int4_pertoken,
    "int8_pertoken": _int8_pertoken,
    "embedded": _embedded,
}


def score_all(
    docs: np.ndarray,
    offs: np.ndarray,
    queries: list[np.ndarray],
    formats: list[str],
    chunk_docs: int,
) -> dict[str, np.ndarray]:
    """Per-doc MaxSim scores [n_queries, n_docs] per format, chunked at doc boundaries."""
    num_docs = offs.shape[0] - 1
    scores = {f: np.empty((len(queries), num_docs), dtype=np.float32) for f in formats}
    for d0 in range(0, num_docs, chunk_docs):
        d1 = min(d0 + chunk_docs, num_docs)
        t0, t1 = int(offs[d0]), int(offs[d1])
        raw = docs[t0:t1].astype(np.float32)
        local = offs[d0 : d1 + 1] - t0
        starts = local[:-1]
        empty = local[:-1] == local[1:]
        mats = {f: TRANSFORMS[f](raw) for f in formats}
        for qi, q in enumerate(queries):
            qT = q.T
            for f in formats:
                dots = mats[f] @ qT
                per_doc_max = np.maximum.reduceat(dots, starts, axis=0)
                if empty.any():
                    per_doc_max[empty] = 0.0
                scores[f][qi, d0:d1] = per_doc_max.sum(axis=1)
        print(f"  scored docs {d0}..{d1} ({', '.join(formats)})", flush=True)
    return scores


def pool3_binary_scores(
    docs: np.ndarray, offs: np.ndarray, queries: list[np.ndarray]
) -> np.ndarray:
    from maxsim.pooling import pool_doc_tokens

    pooled, pooled_offs = pool_doc_tokens(docs.astype(np.float32), offs, 3)
    coarse = _sign(pooled)
    pooled_offs = pooled_offs.astype(np.int64)
    starts = pooled_offs[:-1]
    empty = pooled_offs[:-1] == pooled_offs[1:]
    out = np.empty((len(queries), pooled_offs.shape[0] - 1), dtype=np.float32)
    for qi, q in enumerate(queries):
        dots = coarse @ q.T
        per_doc_max = np.maximum.reduceat(dots, starts, axis=0)
        if empty.any():
            per_doc_max[empty] = 0.0
        out[qi] = per_doc_max.sum(axis=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--coarse",
        action="append",
        choices=["int4_pertoken", "binary", "pool3_binary"],
        default=[],
        help="coarse scan tier(s); default int4_pertoken",
    )
    parser.add_argument(
        "--rescore",
        action="append",
        choices=["dense", "int8_pertoken", "embedded"],
        default=[],
        help="rescore tier(s); default dense + int8_pertoken + embedded",
    )
    parser.add_argument("--m", action="append", type=int, default=[])
    parser.add_argument("--chunk-docs", type=int, default=4000)
    args = parser.parse_args()
    coarse_tiers = args.coarse or ["int4_pertoken"]
    rescore_tiers = args.rescore or ["dense", "int8_pertoken", "embedded"]
    ms = args.m or [25, 50, 100, 200]

    d = np.load(args.cache)
    docs = d["doc_embeddings"]
    offs = d["doc_offsets"].astype(np.int64)
    qrels = d["qrels"]
    q_flat, q_offs = d["query_embeddings"].astype(np.float32), d["query_offsets"].astype(np.int64)
    queries = [q_flat[a:b] for a, b in zip(q_offs[:-1], q_offs[1:])]
    num_docs = offs.shape[0] - 1
    print(f"{args.cache.name}: {num_docs} docs, {docs.shape[0]} tokens, {len(queries)} queries", flush=True)

    flat_formats = sorted(
        {"dense", "binary"} | {c for c in coarse_tiers if c != "pool3_binary"} | set(rescore_tiers)
    )
    scores = score_all(docs, offs, queries, flat_formats, args.chunk_docs)
    if "pool3_binary" in coarse_tiers:
        scores["pool3_binary"] = pool3_binary_scores(docs, offs, queries)

    def mean_ndcg(mat: np.ndarray) -> float:
        return float(np.mean([ndcg10(mat[qi], qrels[qi]) for qi in range(len(queries))]))

    anchors = {f: mean_ndcg(scores[f]) for f in scores}
    for f, v in sorted(anchors.items(), key=lambda kv: -kv[1]):
        print(f"  fullscan {f:14s} NDCG@10 = {v:.4f}", flush=True)

    cascades = []
    for coarse in coarse_tiers:
        order = np.argsort(-scores[coarse], axis=1)
        for rescore in rescore_tiers:
            r = scores[rescore]
            r_top10 = np.argsort(-r, axis=1)[:, :10]
            for m in ms:
                topm = order[:, :m]
                vals, recalls = [], []
                for qi in range(len(queries)):
                    keep = np.zeros(num_docs, dtype=bool)
                    keep[topm[qi]] = True
                    masked = np.where(keep, r[qi], -np.inf)
                    vals.append(ndcg10(masked, qrels[qi]))
                    recalls.append(np.isin(r_top10[qi], topm[qi]).mean())
                row = {
                    "coarse": coarse,
                    "rescore": rescore,
                    "m": m,
                    "ndcg_at_10": float(np.mean(vals)),
                    "recall_of_rescore_top10": float(np.mean(recalls)),
                    "delta_vs_rescore_fullscan": float(np.mean(vals) - anchors[rescore]),
                    "hot_bytes_per_token": BYTES_PER_TOKEN[coarse],
                    "total_bytes_per_token": (
                        BYTES_PER_TOKEN["embedded"]
                        if coarse == "int4_pertoken" and rescore == "embedded"
                        else BYTES_PER_TOKEN[coarse] + BYTES_PER_TOKEN[rescore]
                    ),
                }
                cascades.append(row)
                print(
                    f"  cascade {coarse}->{rescore:14s} m={m:4d}  NDCG@10 = {row['ndcg_at_10']:.4f}"
                    f"  recall = {row['recall_of_rescore_top10']:.4f}",
                    flush=True,
                )

    payload = {
        "cache": args.cache.name,
        "num_docs": num_docs,
        "num_queries": len(queries),
        "fullscan_ndcg_at_10": anchors,
        "cascades": cascades,
        "recipe": {
            "script": "benchmarks/exp_cascade_rescoring.py",
            "args": sys.argv[1:],
            "cache": args.cache.name,
        },
    }
    name = f"exp-cascade-{args.cache.stem}.json"
    for out in (Path("benchmark-results") / name, Path("docs/benchmark_results/raw") / name):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
