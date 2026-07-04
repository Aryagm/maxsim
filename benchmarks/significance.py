"""Paired significance analysis over stored per-query NDCG vectors.

Pools per-query NDCG@10 across the full ViDoRe suite (every query weighted
equally), then reports paired mean deltas with 95% bootstrap confidence
intervals and two-sided sign tests for the tier comparisons the paper cites.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

PAIRS = [
    ("int4_int8q_dp4a", "dense_fp16_baseline"),
    ("bitmax_binary", "dense_fp16_baseline"),
    ("pool3_binary", "dense_fp16_baseline"),
    ("pool2_binary", "bitmax_binary"),
    ("pool3_binary", "bitmax_binary"),
    ("pool3_binary", "pool2_binary"),
    ("binary_token_scale_fp16_cuda", "bitmax_binary"),
    ("binary_token_scale_u4_cuda", "binary_token_scale_fp16_cuda"),
    ("int4_int8q_dp4a", "bitmax_binary"),
]


def _rows(path: Path) -> dict[str, list]:
    data = json.loads(path.read_text())
    out = {}
    for row in data.get("results", []):
        vec = row.get("per_query_ndcg_at_10")
        if vec is not None:
            out[row["implementation"]] = vec
    return out


def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/significance-suite.json"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    per_dataset: list[dict[str, list]] = []
    for path in sorted(args.results_dir.glob("paper-vidore-*colqwen2*-r3.json")):
        rows = _rows(path)
        pool3_path = path.with_name(path.name.replace("paper-", "paper-pool3-", 1))
        if pool3_path.exists():
            rows.update({k: v for k, v in _rows(pool3_path).items() if k != "dense_fp16_baseline"})
        per_dataset.append(rows)

    rng = np.random.default_rng(20260704)
    results = []
    for a, b in PAIRS:
        deltas = []
        for rows in per_dataset:
            if a not in rows or b not in rows:
                continue
            for x, y in zip(rows[a], rows[b]):
                if x is not None and y is not None:
                    deltas.append(x - y)
        if not deltas:
            continue
        arr = np.asarray(deltas)
        boots = np.array([
            arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(args.bootstrap)
        ])
        wins = int((arr > 1e-9).sum())
        losses = int((arr < -1e-9).sum())
        results.append({
            "a": a,
            "b": b,
            "n_queries": len(arr),
            "mean_delta": float(arr.mean()),
            "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)),
            "wins": wins,
            "losses": losses,
            "ties": len(arr) - wins - losses,
            "sign_test_p": _sign_test_p(wins, losses),
        })
        print(
            f"{a:32s} vs {b:28s} n={len(arr):5d} Δ={arr.mean():+.4f} "
            f"CI95 [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}] "
            f"w/l/t {wins}/{losses}/{len(arr)-wins-losses} p={_sign_test_p(wins, losses):.2g}"
        )

    args.output.write_text(json.dumps({"pairs": results, "bootstrap": args.bootstrap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
