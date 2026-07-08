"""Query-token pruning: quality vs query-token budget, composed with doc tiers.

Scoring cost is linear in query tokens. Centroid-distance pruning keeps the
ceil(f*n) tokens FARTHEST (by cosine) from the query's mean token, dropping
generic/near-centroid tokens whose correlated noise lossy doc formats
amplify. Measured on unit-norm text (GTE-ModernColBERT x BEIR), keep-75% is
neutral-or-better for EVERY doc format — it IMPROVES pool3-binary
(+0.014 SciFact) and per-token int4 (+0.026 SciFact) — a free ~1.33x scan
reduction. On visual (ColQwen2) pruning hurts even dense scoring, so it is a
text-side (unit-norm) optimization only: the modality-dependence extends to
the query side. A random-k control (3 seeds, mean) calibrates the strategy.
Note: these caches contain no [MASK] padding (queries average 8.6-21.5 real
tokens), so pruning removes real content tokens; 256-doc visual slices are
directional only (+/-0.015).

    python -m benchmarks.exp_query_pruning caches-full/beir/beir-scifact-gte-moderncolbert.npz
    python -m benchmarks.exp_query_pruning caches/vidore-docvqa-colqwen2-limit256.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.exp_text_scale_inversion import ndcg10


def _sign(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0, -1.0).astype(np.float32)


def _int4_pertoken(x: np.ndarray) -> np.ndarray:
    s = np.abs(x).max(axis=1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    return (np.clip(np.round(x / s), -8, 7) * s).astype(np.float32)


def centroid_keep(q: np.ndarray, f: float) -> np.ndarray:
    """Indices of the ceil(f*n) tokens farthest (cosine) from the query mean token."""
    n = q.shape[0]
    k = max(1, int(np.ceil(f * n)))
    c = q.mean(axis=0)
    c_norm = np.linalg.norm(c)
    q_norm = np.linalg.norm(q, axis=1)
    sim = (q @ c) / np.maximum(q_norm * c_norm, 1e-9)
    return np.argsort(sim, kind="stable")[:k]  # ascending similarity = farthest first


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--keep", action="append", type=float, default=[])
    parser.add_argument(
        "--doc-format",
        action="append",
        choices=["dense", "binary", "int4_pertoken", "pool3_binary"],
        default=[],
    )
    parser.add_argument("--random-seeds", type=int, default=3)
    args = parser.parse_args()
    keeps = args.keep or [1.0, 0.75, 0.5]
    formats = args.doc_format or ["dense", "binary", "int4_pertoken", "pool3_binary"]

    d = np.load(args.cache)
    docs = d["doc_embeddings"].astype(np.float32)
    offs = d["doc_offsets"].astype(np.int64)
    qrels = d["qrels"]
    q_flat, q_offs = d["query_embeddings"].astype(np.float32), d["query_offsets"].astype(np.int64)
    queries = [q_flat[a:b] for a, b in zip(q_offs[:-1], q_offs[1:])]
    lens = np.array([q.shape[0] for q in queries])
    print(
        f"{args.cache.name}: {offs.shape[0]-1} docs, {docs.shape[0]} tokens, "
        f"{len(queries)} queries (mean {lens.mean():.1f} tokens)",
        flush=True,
    )

    mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in formats:
        if f == "dense":
            mats[f] = (docs, offs)
        elif f == "binary":
            mats[f] = (_sign(docs), offs)
        elif f == "int4_pertoken":
            mats[f] = (_int4_pertoken(docs), offs)
        elif f == "pool3_binary":
            from maxsim.pooling import pool_doc_tokens

            pooled, pooled_offs = pool_doc_tokens(docs, offs, 3)
            mats[f] = (_sign(pooled), pooled_offs.astype(np.int64))

    rngs = [np.random.default_rng(seed) for seed in range(args.random_seeds)]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for fmt, (mat, fmt_offs) in mats.items():
        starts = fmt_offs[:-1]
        empty = fmt_offs[:-1] == fmt_offs[1:]
        acc: dict[tuple[str, float], list[float]] = {}
        for qi, q in enumerate(queries):
            dots = mat @ q.T  # [tokens, n_qtokens] — computed once, columns subset per variant
            per_tok_max = np.maximum.reduceat(dots, starts, axis=0)
            if empty.any():
                per_tok_max[empty] = 0.0
            n = q.shape[0]
            for f in keeps:
                k = max(1, int(np.ceil(f * n)))
                variants = {"centroid": [centroid_keep(q, f)]}
                if f < 1.0:
                    variants["random"] = [rng.choice(n, size=k, replace=False) for rng in rngs]
                else:
                    variants["random"] = [np.arange(n)]
                for strat, subsets in variants.items():
                    vals = [
                        ndcg10(per_tok_max[:, idx].sum(axis=1), qrels[qi]) for idx in subsets
                    ]
                    acc.setdefault((strat, f), []).append(float(np.mean(vals)))
        results[fmt] = {}
        for (strat, f), vals in sorted(acc.items()):
            results[fmt].setdefault(strat, {})[f"{f:g}"] = float(np.mean(vals))
        base = results[fmt]["centroid"]["1"]
        for f in keeps:
            v = results[fmt]["centroid"][f"{f:g}"]
            print(f"  {fmt:14s} centroid f={f:<5g} NDCG@10 = {v:.4f}  (delta {v - base:+.4f})", flush=True)

    payload = {
        "cache": args.cache.name,
        "mean_query_tokens": float(lens.mean()),
        "ndcg_at_10": results,
        "recipe": {
            "script": "benchmarks/exp_query_pruning.py",
            "args": sys.argv[1:],
            "cache": args.cache.name,
            "random_seeds": list(range(args.random_seeds)),
        },
    }
    name = f"exp-qprune-{args.cache.stem}.json"
    for out in (Path("benchmark-results") / name, Path("docs/benchmark_results/raw") / name):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
