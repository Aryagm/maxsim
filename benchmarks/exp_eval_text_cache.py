"""Exact-simulation tier evaluation of a text cache on CPU (no GPU needed).

Computes NDCG@10 for dense fp32, binary signs, per-tensor int4 (exact grid
simulation, matches the dp4a kernel to 4 decimals), and optionally pooled
binary. Vectorized: one [tokens x q_tokens] matmul per query per doc-matrix.
Used to finish FiQA after the harness loop-dense proved too slow at 57k docs.

    python -m benchmarks.exp_eval_text_cache caches-full/beir/beir-fiqa-gte-moderncolbert.npz --pool 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.exp_text_scale_inversion import ndcg10


def _score_all(doc_matrix: np.ndarray, offs: np.ndarray, queries, qrels) -> float:
    starts = offs[:-1]
    empty = offs[:-1] == offs[1:]
    vals = []
    for qi, q in enumerate(queries):
        dots = doc_matrix @ q.T.astype(np.float32)
        per_doc_max = np.maximum.reduceat(dots, starts, axis=0)
        if empty.any():
            per_doc_max[empty] = 0.0
        vals.append(ndcg10(per_doc_max.sum(axis=1), qrels[qi]))
    return float(np.mean(vals))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--pool", type=int, action="append", default=[])
    args = parser.parse_args()

    d = np.load(args.cache)
    docs = d["doc_embeddings"].astype(np.float32)
    offs = d["doc_offsets"].astype(np.int64)
    qrels = d["qrels"]
    q_flat, q_offs = d["query_embeddings"], d["query_offsets"].astype(np.int64)
    queries = [q_flat[a:b].astype(np.float32) for a, b in zip(q_offs[:-1], q_offs[1:])]
    print(f"{args.cache.name}: {offs.shape[0]-1} docs, {docs.shape[0]} tokens, {len(queries)} queries", flush=True)

    results = {}

    def run(name, matrix, offsets=None):
        ndcg = _score_all(matrix, offs if offsets is None else offsets, queries, qrels)
        results[name] = ndcg
        print(f"  {name:22s} NDCG@10 = {ndcg:.4f}", flush=True)

    run("dense_fp32", docs)
    run("binary", np.where(docs >= 0, np.float32(1), np.float32(-1)))
    scale = np.float32(np.max(np.abs(docs)) / 7.0)
    run("int4_sim", np.clip(np.round(docs / scale), -8, 7).astype(np.float32) * scale)

    for f in args.pool:
        from maxsim.pooling import pool_doc_tokens

        pooled, pooled_offs = pool_doc_tokens(docs, offs, f)
        run(
            f"pool{f}_binary",
            np.where(pooled >= 0, np.float32(1), np.float32(-1)),
            pooled_offs.astype(np.int64),
        )

    out = Path("benchmark-results") / f"exp-eval-{args.cache.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cache": args.cache.name, "ndcg_at_10": results}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
