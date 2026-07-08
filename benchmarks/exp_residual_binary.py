"""Residual (multi-stage sign) quantization of a text cache — a NEGATIVE result.

Two-stage sign quantization (b1 = sign(x), r = x - a1*b1, b2 = sign(r)) cuts
reconstruction MSE ~3x versus plain binary, yet loses badly on NDCG@10 for
unit-norm text token embeddings: SciFact drops from 0.678 (binary) to 0.322
(res2_global). Quality is non-monotone in reconstruction fidelity, so every
variant reports both NDCG@10 and per-element reconstruction MSE to make that
visible in the artifact itself. Variants sweep the scale granularity of the
stage weights (global scalar, per-token, ratio-only) plus a three-stage
extension; dense/binary/int4 controls included. See the negative-result
catalogue in docs/gpu_optimization.md and paper Section 6.

    python -m benchmarks.exp_residual_binary caches-full/beir/beir-scifact-gte-moderncolbert.npz
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
    b1 = _sign(docs)
    # control: per-token int4, the best int4 granularity (see exp-int4-variants artifacts)
    s4 = absd.max(axis=1, keepdims=True) / 7.0
    s4[s4 == 0] = 1.0
    int4 = (np.clip(np.round(docs / s4), -8, 7) * s4).astype(np.float32)

    # global-scalar stages: a_k = mean |residual| entering stage k (L1-optimal for sign codes)
    a1g = float(absd.mean())
    r1g = docs - a1g * b1
    b2g = _sign(r1g)
    a2g = float(np.abs(r1g).mean())
    r2g = r1g - a2g * b2g
    b3g = _sign(r2g)
    a3g = float(np.abs(r2g).mean())
    # per-token stage weights
    a1t = absd.mean(axis=1, keepdims=True).astype(np.float32)
    a1t_safe = np.where(a1t == 0, 1.0, a1t).astype(np.float32)
    r1t = docs - a1t * b1
    b2t = _sign(r1t)
    a2t = np.abs(r1t).mean(axis=1, keepdims=True).astype(np.float32)
    ratio_t = (a2t / a1t_safe).astype(np.float32)
    rg = np.float32(a2g / a1g)
    r3g = np.float32(a3g / a1g)

    recon_mse = {
        "dense": 0.0,
        "binary": float(np.mean(r1g**2)),  # vs a1g*b1, the L1-optimal binary reconstruction
        "int4": float(np.mean((docs - int4) ** 2)),
        "res2_global": float(np.mean(r2g**2)),
        "res2_pertoken": float(np.mean((r1t - a2t * b2t) ** 2)),
        "res3_global": float(np.mean((r2g - a3g * b3g) ** 2)),
    }
    # res2_ratio scores b1 + (a2t/a1t)*b2t: same reconstruction as res2_pertoken
    # up to the per-token scale a1t, so it inherits that fidelity.
    recon_mse["res2_ratio"] = recon_mse["res2_pertoken"]
    del absd, r1g, r2g, r1t
    print(f"  scales: a1g={a1g:.4f} a2g={a2g:.4f} a3g={a3g:.4f}", flush=True)

    starts = offs[:-1]
    empty = offs[:-1] == offs[1:]
    keys = ["dense", "binary", "int4", "res2_global", "res2_pertoken", "res2_ratio", "res3_global"]
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for qi, q in enumerate(queries):
        qT = q.T
        d1 = b1 @ qT
        d2g = b2g @ qT
        d2t = b2t @ qT
        d3 = b3g @ qT
        dots = {
            "dense": docs @ qT,
            "binary": d1,
            "int4": int4 @ qT,
            "res2_global": d1 + rg * d2g,
            "res2_pertoken": a1t * d1 + a2t * d2t,
            "res2_ratio": d1 + ratio_t * d2t,
            "res3_global": d1 + rg * d2g + r3g * d3,
        }
        for k, v in dots.items():
            per_doc_max = np.maximum.reduceat(v, starts, axis=0)
            if empty.any():
                per_doc_max[empty] = 0.0
            vals[k].append(ndcg10(per_doc_max.sum(axis=1), qrels[qi]))

    results = {k: float(np.mean(v)) for k, v in vals.items()}
    for k in keys:
        print(f"  {k:16s} NDCG@10 = {results[k]:.4f}   recon MSE = {recon_mse[k]:.5f}", flush=True)

    payload = {
        "cache": args.cache.name,
        "ndcg_at_10": results,
        "recon_mse": recon_mse,
        "recipe": {
            "script": "benchmarks/exp_residual_binary.py",
            "args": sys.argv[1:],
            "cache": args.cache.name,
        },
    }
    name = f"exp-residual-binary-{args.cache.stem}.json"
    for out in (Path("benchmark-results") / name, Path("docs/benchmark_results/raw") / name):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
