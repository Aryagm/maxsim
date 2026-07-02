"""Summarize pareto-v2 run_retrieval JSONs: frontier table + paired stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("results", [])


def summarize(path: Path) -> None:
    rows = _rows(path)
    dense = next((r for r in rows if r["implementation"] == "dense_fp16_baseline"), None)
    per_query = {}
    print(f"\n=== {path.name} (queries={rows[0]['query_count']}, docs={rows[0]['docs']}) ===")
    header = f"{'variant':34s} {'ndcg@10':>8s} {'d_dense':>8s} {'lat_ms':>9s} {'bytes':>11s} {'comp32':>7s} {'device':>14s}"
    print(header)
    for row in rows:
        name = row.get("variant") or row["implementation"]
        ndcg = row.get("ndcg_at_10", 0.0)
        delta = row.get("quality_delta_vs_dense_ndcg_at_10")
        comp = row.get("doc_memory_compression_vs_fp32")
        pq = row.get("per_query_ndcg_at_10")
        if pq:
            per_query[name] = pq
        print(
            f"{name:34s} {ndcg:8.4f} "
            f"{(f'{delta:+.4f}' if delta is not None else '     ---'):>8s} "
            f"{row['latency_ms']:9.2f} {row['doc_storage_bytes']:11d} "
            f"{(f'{comp:.1f}x' if comp else '---'):>7s} "
            f"{row.get('requested_device', '?'):>14s}"
            + (f"  kernel={row['maxsim_kernel_variant']}" if "maxsim_kernel_variant" in row else "")
        )

    def paired(a: str, b: str) -> None:
        if a not in per_query or b not in per_query:
            return
        deltas = [
            (x or 0.0) - (y or 0.0)
            for x, y in zip(per_query[a], per_query[b])
            if x is not None and y is not None
        ]
        arr = np.asarray(deltas)
        wins = int(np.sum(arr > 1e-9))
        losses = int(np.sum(arr < -1e-9))
        print(
            f"  paired {a} vs {b}: mean {arr.mean():+.4f}, wins {wins}, losses {losses}, "
            f"unchanged {len(arr) - wins - losses}"
        )

    paired("binary_token_scale_cuda", "binary")
    paired("binary_token_scale_cuda", "binary_token_scale")
    paired("binary_token_scale_fp16_cuda", "binary_token_scale_cuda")
    paired("binary_token_scale_cuda", "int4_symmetric_per_tensor")
    paired("pool2_binary_token_scale_cuda", "binary_token_scale_cuda")
    paired("pool2_binary", "binary")

    if dense is not None:
        checks = {r.get("variant") or r["implementation"]: r["score_checksum"] for r in rows}
        ts_cuda = checks.get("binary_token_scale_cuda")
        ts_ref = checks.get("binary_token_scale")
        plain = checks.get("binary")
        if ts_cuda is not None and ts_ref is not None and plain is not None:
            drift_ref = abs(ts_cuda - ts_ref) / max(abs(ts_ref), 1e-9)
            drift_plain = abs(ts_cuda - plain) / max(abs(plain), 1e-9)
            verdict = "OK (matches CPU token-scale ref)" if drift_ref < 1e-3 and drift_plain > 1e-3 else "SUSPECT"
            print(f"  checksum gate: vs token_scale ref drift {drift_ref:.2e}, vs plain binary drift {drift_plain:.2e} -> {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.inputs:
        summarize(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
