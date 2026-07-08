"""int4 scale-granularity sweep: per-tensor vs per-token vs per-channel.

Produced the 2026-07-05 discovery that PER-TOKEN int4 scales close the text
quality gap: per-tensor int4 under-quantizes unit-norm text (~2 effective
levels; SciFact 0.505), while one scale per token recovers 0.698 vs binary
0.678 — and wins or ties on every modality tested (text, visual, audio).
Regenerates the committed exp-int4-variants-*.json artifacts.

    python -m benchmarks.exp_int4_variants caches-full/beir/beir-scifact-gte-moderncolbert.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.exp_text_scale_inversion import ndcg10


def _quant(x: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    return (np.clip(np.round(x / scale), -8, 7) * scale).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()

    d = np.load(args.cache)
    docs = d["doc_embeddings"].astype(np.float32)
    offs = d["doc_offsets"].astype(np.int64)
    qrels = d["qrels"]
    q_flat, q_offs = d["query_embeddings"].astype(np.float32), d["query_offsets"].astype(np.int64)
    queries = [q_flat[a:b] for a, b in zip(q_offs[:-1], q_offs[1:])]
    print(f"{args.cache.name}: {offs.shape[0]-1} docs, {docs.shape[0]} tokens, {len(queries)} queries", flush=True)

    absd = np.abs(docs)
    s_tensor = float(absd.max()) / 7.0
    s_token = absd.max(axis=1, keepdims=True) / 7.0
    s_token[s_token == 0] = 1.0
    s_channel = absd.max(axis=0, keepdims=True) / 7.0
    s_channel[s_channel == 0] = 1.0
    variants = {
        "int4_per_tensor": _quant(docs, s_tensor),
        "int4_per_token": _quant(docs, s_token),
        "int4_per_channel": _quant(docs, s_channel),
    }
    del absd

    starts = offs[:-1]
    empty = offs[:-1] == offs[1:]
    results = {}
    for name, mat in variants.items():
        vals = []
        for qi, q in enumerate(queries):
            per_doc_max = np.maximum.reduceat(mat @ q.T, starts, axis=0)
            if empty.any():
                per_doc_max[empty] = 0.0
            vals.append(ndcg10(per_doc_max.sum(axis=1), qrels[qi]))
        results[name] = float(np.mean(vals))
        print(f"  {name:18s} NDCG@10 = {results[name]:.4f}", flush=True)

    payload = {
        "cache": args.cache.name,
        "ndcg_at_10": results,
        "recipe": {
            "script": "benchmarks/exp_int4_variants.py",
            "args": sys.argv[1:],
            "cache": args.cache.name,
        },
    }
    name = f"exp-int4-variants-{args.cache.stem}.json"
    for out in (Path("benchmark-results") / name, Path("docs/benchmark_results/raw") / name):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
