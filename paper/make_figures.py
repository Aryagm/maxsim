"""Generate the paper figures from the committed benchmark ledger.

Reads docs/benchmark_results/raw/*.json and writes PDFs into paper/figures/.
Run from the repository root:  python paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("docs/benchmark_results/raw")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "binary": "#2a78d6",
    "ts": "#1baf7a",
    "u4": "#eda100",
    "int4": "#008300",
    "pool2": "#4a3aa7",
    "pool3": "#e34948",
    "dense": "#8a94a0",
}
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


def fig_scale_flip():
    scales = ["256 docs\n(DocVQA)", "suite\n(0.5–1.7k)", "10,171 docs\n(mixed)", "25,000 docs\n(stress)"]
    series = {
        "int4": ("int4+dp4a", [0.0093, 0.0048, 0.0152, None]),
        "pool3": ("pool3 (96$\\times$)", [-0.0133, -0.0024, 0.0112, 0.0238]),
        "pool2": ("pool2 (64$\\times$)", [-0.0074, -0.0012, 0.0008, 0.0039]),
        "ts": ("fp16 token scales", [0.0092, -0.0013, -0.0059, -0.0120]),
        "u4": ("u4 token scales", [0.0126, -0.0015, -0.0077, -0.0135]),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    x = range(len(scales))
    for key, (label, ys) in series.items():
        xs = [i for i, v in zip(x, ys) if v is not None]
        vs = [v for v in ys if v is not None]
        ax.plot(xs, vs, "-o", color=COLORS[key], label=label, linewidth=1.6, markersize=4)
    ax.axhline(0, color="#666", linewidth=0.9, linestyle="--")
    ax.text(len(scales) - 1.02, 0.0008, "plain binary", fontsize=7.5, color="#666", ha="right")
    ax.set_xticks(list(x), scales)
    ax.set_ylabel(r"$\Delta$ NDCG@10 vs. plain binary")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "scale_flip.pdf")


def _final_rows():
    data = json.loads((RAW / "paper-unique-10k-final.json").read_text())
    return {r["implementation"]: r for r in data["results"] if not r.get("status") or r["status"] == "ok"}


def fig_pareto_10k():
    rows = _final_rows()
    pts = [
        ("dense", "dense_fp16_vectorized", 2.0, "dense fp16"),
        ("int4", "bitmax_int4_dp4a", 8.0, "int4+dp4a"),
        ("ts", "bitmax_binary_token_scale", 28.4, "fp16 scales"),
        ("u4", "bitmax_binary_token_scale_u4", 31.0, "u4 scales"),
        ("binary", "bitmax_binary", 32.0, "binary"),
        ("pool2", "bitmax_pooled_binary", 63.9, "pool2"),
        ("pool3", "bitmax_pooled_binary3", 95.9, "pool3"),
    ]
    dense_y = rows["dense_fp16_vectorized"]["ndcg_at_10"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhspan(dense_y - 0.01, dense_y, color="#008300", alpha=0.08, lw=0)
    ax.axhline(dense_y, color="#666", linewidth=0.9, linestyle="--")
    offsets = {"ts": (-6, -13), "u4": (-28, 5), "binary": (14, 7), "pool2": (0, -13), "pool3": (0, 8),
               "int4": (0, 8), "dense": (30, 8)}
    for key, impl, comp, label in pts:
        y = rows[impl]["ndcg_at_10"]
        ax.scatter([comp], [y], s=42, color=COLORS[key], zorder=3, edgecolor="white", linewidth=1)
        dx, dy = offsets[key]
        ax.annotate(label, (comp, y), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=7.5, color=COLORS[key], fontweight="bold")
    fp = rows.get("fast_plaid")
    if fp:
        ax.scatter([3.4], [fp["ndcg_at_10"]], s=42, marker="s", color="#b0651a", zorder=3,
                   edgecolor="white", linewidth=1)
        ax.annotate("fast-plaid", (3.4, fp["ndcg_at_10"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color="#b0651a", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128], ["2×", "4×", "8×", "16×", "32×", "64×", "128×"])
    ax.set_xlabel("compression vs. fp32 storage (log)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0.46, 0.525)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_10k.pdf")


def fig_latency_10k():
    rows = _final_rows()
    order = [
        ("dense fp16 (loop impl.)", "dense_fp16_baseline", "dense", 0.45),
        ("dense fp16 (vectorized)", "dense_fp16_vectorized", "dense", 1.0),
        ("int4 fp32-query", "bitmax_int4", "int4", 0.45),
        ("int4+dp4a", "bitmax_int4_dp4a", "int4", 1.0),
        ("fp16 token scales", "bitmax_binary_token_scale", "ts", 1.0),
        ("binary", "bitmax_binary", "binary", 1.0),
        ("pool3", "bitmax_pooled_binary3", "pool3", 1.0),
        ("pool2", "bitmax_pooled_binary", "pool2", 1.0),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for i, (label, impl, key, alpha) in enumerate(order):
        v = rows[impl]["latency_ms"]
        y = len(order) - i
        ax.plot([0.3, v / 1000], [y, y], color="#dddddd", linewidth=1, zorder=1)
        ax.scatter([v / 1000], [y], s=40, color=COLORS[key], alpha=alpha, zorder=3,
                   edgecolor="white", linewidth=1)
        txt = f"{v/1000:.2f} s" if v < 10000 else f"{v/1000:.0f} s"
        ax.annotate(txt, (v / 1000, y), textcoords="offset points", xytext=(8, -3), fontsize=7.5, color="#444")
    ax.set_yticks([len(order) - i for i in range(len(order))], [o[0] for o in order], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 400)
    ax.set_xlabel("latency for 256 queries × 10,171 documents (s, log)")
    fig.tight_layout()
    fig.savefig(OUT / "latency_10k.pdf")


def fig_per_dataset():
    keys = [("dense_fp16_baseline", "dense", "dense fp16"),
            ("int4_int8q_dp4a", "int4", "int4+dp4a"),
            ("bitmax_binary", "binary", "binary"),
            ("pool3_binary", "pool3", "pool3")]
    rows_by_ds = []
    for path in sorted(RAW.glob("paper-vidore-*colqwen2*-r3.json")):
        data = json.loads(path.read_text())
        rows = {r["implementation"]: r for r in data["results"]}
        p3 = path.with_name(path.name.replace("paper-", "paper-pool3-", 1))
        if p3.exists():
            rows.update({r["implementation"]: r for r in json.loads(p3.read_text())["results"]
                         if r["implementation"] != "dense_fp16_baseline"})
        name = path.stem.replace("paper-vidore-", "").replace("-colqwen2", "").replace("-r3", "")
        name = name.replace("-test", "").replace("syntheticdocqa", "SynthDocQA")
        rows_by_ds.append((name, rows))
    rows_by_ds.sort(key=lambda t: -t[1]["dense_fp16_baseline"]["ndcg_at_10"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (name, rows) in enumerate(rows_by_ds):
        y = len(rows_by_ds) - i
        vals = [rows[k]["ndcg_at_10"] for k, _, _ in keys if k in rows]
        ax.plot([min(vals), max(vals)], [y, y], color="#dddddd", linewidth=2, zorder=1)
        for k, ckey, _ in keys:
            if k in rows:
                ax.scatter([rows[k]["ndcg_at_10"]], [y], s=26, color=COLORS[ckey], zorder=3,
                           edgecolor="white", linewidth=0.8)
    ax.set_yticks([len(rows_by_ds) - i for i in range(len(rows_by_ds))],
                  [n for n, _ in rows_by_ds], fontsize=7.5)
    ax.set_xlabel("NDCG@10")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[c], label=l) for _, c, l in keys]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "per_dataset.pdf")


if __name__ == "__main__":
    fig_scale_flip()
    fig_pareto_10k()
    fig_latency_10k()
    fig_per_dataset()
    print("figures written to", OUT)
