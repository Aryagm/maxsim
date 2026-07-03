"""Aggregate paper-suite JSONs into per-dataset and mean tables (markdown)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TIERS = [
    ("dense_fp16_baseline", "dense fp16"),
    ("bitmax_binary", "binary"),
    ("binary_token_scale_fp16_cuda", "balanced (fp16 scales)"),
    ("binary_token_scale_u4_cuda", "compact (u4 scales)"),
    ("int4_int8q_dp4a", "max_quality (int4 dp4a)"),
    ("pool2_binary", "max_compression (pool2)"),
]


def _rows(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["implementation"]: row for row in data.get("results", [])}


def per_dataset(paths: list[Path]) -> None:
    aggregate = defaultdict(list)
    print("\n## Per-dataset NDCG@10 (full test splits)\n")
    header = ["dataset"] + [label for _, label in TIERS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for path in sorted(paths):
        rows = _rows(path)
        name = path.stem.replace("paper-vidore-", "").replace("-r3", "")
        cells = [name]
        for key, _ in TIERS:
            row = rows.get(key)
            if row is None:
                cells.append("—")
                continue
            ndcg = row.get("ndcg_at_10")
            cells.append(f"{ndcg:.4f}")
            aggregate[key].append(ndcg)
        print("| " + " | ".join(cells) + " |")
    print("| **mean** | " + " | ".join(
        f"**{np.mean(aggregate[key]):.4f}**" if aggregate.get(key) else "—" for key, _ in TIERS
    ) + " |")

    print("\n## Mean latency / storage (same runs)\n")
    print("| tier | mean latency ms | mean compression vs fp32 |")
    print("|---|---|---|")
    lat = defaultdict(list)
    comp = defaultdict(list)
    for path in paths:
        rows = _rows(path)
        for key, _ in TIERS:
            if key in rows:
                lat[key].append(rows[key]["latency_ms"])
                if rows[key].get("doc_memory_compression_vs_fp32"):
                    comp[key].append(rows[key]["doc_memory_compression_vs_fp32"])
    for key, label in TIERS:
        if lat.get(key):
            comp_txt = f"{np.mean(comp[key]):.1f}x" if comp.get(key) else "—"
            print(f"| {label} | {np.mean(lat[key]):.1f} | {comp_txt} |")


def comparison(path: Path) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    print(f"\n## {path.stem}\n")
    print("| implementation | NDCG@10 | R@10 | best ms | p95 ms | compression vs fp32 |")
    print("|---|---|---|---|---|---|")
    for row in data.get("results", []):
        if row.get("status") and row["status"] != "ok":
            continue
        print(
            f"| {row['implementation']} | {row.get('ndcg_at_10', float('nan')):.4f} "
            f"| {row.get('recall_at_10', float('nan')):.4f} | {row.get('latency_ms', float('nan')):.1f} "
            f"| {row.get('latency_p95_ms', float('nan')):.1f} "
            f"| {row.get('doc_memory_compression_vs_fp32', 0) or 0:.1f}x |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    colqwen = sorted(args.results_dir.glob("paper-vidore-*colqwen2*-r3.json"))
    colpali = sorted(args.results_dir.glob("paper-colpali-*-r3.json"))
    if colqwen:
        print("# ColQwen2 (primary model)")
        per_dataset(colqwen)
    if colpali:
        print("\n# ColPali (generality check)")
        per_dataset(colpali)
    comparison(args.results_dir / "paper-unique-10k-comparison.json")
    comparison(args.results_dir / "paper-docscale-25k-comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
