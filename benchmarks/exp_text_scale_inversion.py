"""Test scale-inversion hypotheses on unit-normalized text ColBERT caches.

Mechanism found on BEIR: with unit-norm tokens, mean|x| anti-correlates with
token informativeness (spiky rare-term tokens score LOW). If true, inverse or
spikiness-based per-token scales should beat mean-abs scales and may beat
plain binary. Signs are fixed across variants, so per query we compute the
sign dot-product matrix once and sweep every scale vector over it.

    python -m benchmarks.exp_text_scale_inversion caches-full/beir/beir-scifact-gte-moderncolbert.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def ndcg10(scores: np.ndarray, qrels_row: np.ndarray) -> float:
    order = np.argsort(-scores)[:10]
    gains = qrels_row[order]
    dcg = float(np.sum(gains / np.log2(np.arange(2, 12))))
    ideal_n = int(min(10, (qrels_row > 0).sum()))
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_n + 2)))) if ideal_n else 1.0
    return dcg / idcg if idcg else 0.0


def main() -> int:
    path = Path(sys.argv[1])
    d = np.load(path)
    docs, offs = d["doc_embeddings"], d["doc_offsets"].astype(np.int64)
    q_flat, q_offs = d["query_embeddings"], d["query_offsets"].astype(np.int64)
    qrels = d["qrels"]
    queries = [q_flat[a:b] for a, b in zip(q_offs[:-1], q_offs[1:])]
    num_docs = offs.shape[0] - 1
    starts = offs[:-1]

    signs = np.where(docs >= 0, np.float32(1), np.float32(-1))
    mean_abs = np.mean(np.abs(docs), axis=1, dtype=np.float64).astype(np.float32)
    max_abs = np.max(np.abs(docs), axis=1).astype(np.float32)
    spike = max_abs / np.maximum(mean_abs, 1e-9)

    def norm_mean1(v):
        return (v / v.mean()).astype(np.float32)

    variants = {
        "binary (s=1)": None,
        "mean_abs": norm_mean1(mean_abs),
        "inverse (1/mean_abs)": norm_mean1(1.0 / np.maximum(mean_abs, 1e-9)),
        "inverse^0.5": norm_mean1(np.maximum(mean_abs, 1e-9) ** -0.5),
        "max_abs": norm_mean1(max_abs),
        "spikiness (max/mean)": norm_mean1(spike),
        "spikiness^0.5": norm_mean1(spike**0.5),
    }

    sums = {k: [] for k in variants}
    for qi, q in enumerate(queries):
        dots = signs @ q.T.astype(np.float32)  # [tokens, q_tokens]
        for name, scale in variants.items():
            scaled = dots if scale is None else dots * scale[:, None]
            per_doc_max = np.maximum.reduceat(scaled, starts, axis=0)
            empty = offs[:-1] == offs[1:]
            if empty.any():
                per_doc_max[empty] = 0.0
            sums[name].append(ndcg10(per_doc_max.sum(axis=1), qrels[qi]))
        if (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(queries)} queries", flush=True)

    print(f"\n== {path.name} ({num_docs} docs, {len(queries)} queries)")
    results = {}
    for name, vals in sums.items():
        results[name] = float(np.mean(vals))
        print(f"  {name:24s} NDCG@10 = {np.mean(vals):.4f}")
    out = Path("benchmark-results") / f"exp-scale-inversion-{path.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cache": path.name, "ndcg_at_10": results}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
