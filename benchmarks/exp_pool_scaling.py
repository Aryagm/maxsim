"""Does the optimal pooling factor grow with corpus size?

Measured so far (10k unique corpus): pool3 (95.9x) beats pool2 (63.9x) beats
plain binary (32x). If pooling-as-denoising strengthens with scale, pool4+
may continue the trend. Ward-pools the corpus at factors {4,5,6}, scores all
queries with exact binary MaxSim (vectorized numpy), reports NDCG@10 against
the committed pool2/pool3/binary/dense numbers.

    python -m benchmarks.exp_pool_scaling caches-full/vidore-mixed-public-unique-colqwen2-limit10000.npz 4 5 6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from maxsim.pooling import pool_doc_tokens
from benchmarks.exp_text_scale_inversion import ndcg10


def score_binary(docs: np.ndarray, offs: np.ndarray, queries, qrels) -> float:
    signs = np.where(docs >= 0, np.float32(1), np.float32(-1))
    starts = offs[:-1]
    empty = offs[:-1] == offs[1:]
    vals = []
    for qi, q in enumerate(queries):
        dots = signs @ q.T.astype(np.float32)
        per_doc_max = np.maximum.reduceat(dots, starts, axis=0)
        if empty.any():
            per_doc_max[empty] = 0.0
        vals.append(ndcg10(per_doc_max.sum(axis=1), qrels[qi]))
    return float(np.mean(vals))


def main() -> int:
    path = Path(sys.argv[1])
    factors = [int(a) for a in sys.argv[2:]] or [4, 5, 6]
    d = np.load(path)
    print("cache keys:", d.files, flush=True)
    docs = d["doc_embeddings"].astype(np.float32)
    offs = d["doc_offsets"].astype(np.int64)
    qrels = d["qrels"]
    q = d["query_embeddings"]
    if q.ndim == 3:
        queries = [np.ascontiguousarray(x, dtype=np.float32) for x in q]
    else:
        q_offs = d["query_offsets"].astype(np.int64)
        queries = [q[a:b].astype(np.float32) for a, b in zip(q_offs[:-1], q_offs[1:])]
    num_docs = offs.shape[0] - 1
    print(f"{path.name}: {num_docs} docs, {docs.shape[0]} tokens, {len(queries)} queries", flush=True)

    results = {}
    baseline = score_binary(docs, offs, queries, qrels)
    results["pool1_binary"] = {"ndcg_at_10": baseline, "tokens": int(docs.shape[0])}
    print(f"binary (no pooling)   NDCG@10 = {baseline:.4f}", flush=True)

    for f in factors:
        pooled, pooled_offs = pool_doc_tokens(docs, offs, f)
        ndcg = score_binary(pooled, pooled_offs, queries, qrels)
        compression = 32.0 * docs.shape[0] / pooled.shape[0]
        results[f"pool{f}_binary"] = {
            "ndcg_at_10": ndcg,
            "tokens": int(pooled.shape[0]),
            "approx_compression_vs_fp32": round(compression, 1),
        }
        print(f"pool{f} binary ({compression:.0f}x)  NDCG@10 = {ndcg:.4f}", flush=True)

    out = Path("benchmark-results") / f"exp-pool-scaling-{path.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cache": path.name, "results": results}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
