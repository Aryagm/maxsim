"""Generate the paper figures from the benchmark artifact ledger.

Reads docs/benchmark_results/raw/*.json and writes PDFs into paper/figures/
plus PNG twins into docs/figures/ for the README. Run from the repo root:
    python paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RAW = Path("docs/benchmark_results/raw")
OUT_PDF = Path("paper/figures")
OUT_PNG = Path("docs/figures")
OUT_PDF.mkdir(parents=True, exist_ok=True)
OUT_PNG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "binary": "#2a78d6",
    "ts": "#0e9668",
    "u4": "#c98500",
    "int4": "#008300",
    "int4_token": "#007d8a",
    "residual": "#7b579b",
    "pool2": "#4a3aa7",
    "pool3": "#d43d3c",
    "dense": "#5c6672",
    "ext": "#8a5a1e",
}
MARKERS = {"binary": "o", "ts": "s", "u4": "D", "int4": "^", "pool2": "v", "pool3": "P",
           "dense": "o", "ext": "X"}
LABELS = {
    "binary": "binary (32×)",
    "ts": "fp16 token scales (28.4×)",
    "u4": "u4 token scales (31×)",
    "int4": "int4 + dp4a (8×)",
    "pool2": "pool2 binary (63.9×)",
    "pool3": "pool3 binary (95.9×)",
    "dense": "dense fp16",
}
plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": "#e3e5e8",
    "grid.linewidth": 0.55,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name: str):
    fig.savefig(OUT_PDF / f"{name}.pdf")
    fig.savefig(OUT_PNG / f"{name}.png", dpi=220)
    plt.close(fig)


def _mark(ax, x, y, key, size=46, alpha=1.0, zorder=3):
    ax.scatter([x], [y], s=size, marker=MARKERS[key], color=COLORS[key], alpha=alpha,
               edgecolor="white", linewidth=1.1, zorder=zorder)


def _final_rows():
    data = json.loads((RAW / "paper-unique-10k-final.json").read_text())
    return {r["implementation"]: r for r in data["results"] if not r.get("status") or r["status"] == "ok"}


# ---------------------------------------------------------------- fig: formats
def fig_format_layout():
    """Byte budget per stored 128-d token, log-width bars with exact values."""
    tiers = [
        ("fp32 (reference)", 512.0, "dense", "1.0×"),
        ("fp16 dense", 256.0, "dense", "2×"),
        ("residual int4 (q4 + q4)", 136.0, "residual", "3.76×"),
        ("int4 + fp32 token scale", 68.0, "int4_token", "7.52×"),
        ("int4 (per-tensor scale)", 64.0, "int4", "8×"),
        ("binary + fp16 scale", 18.0, "ts", "28.4×"),
        ("binary + u4 scale", 16.5, "u4", "31×"),
        ("binary signs", 16.0, "binary", "32×"),
        ("pool2 binary$^{\\dagger}$", 8.0, "pool2", "63.9×"),
        ("pool3 binary$^{\\dagger}$", 5.33, "pool3", "95.9×"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    ax.grid(axis="y", visible=False)
    import numpy as np
    for i, (label, size_b, key, comp) in enumerate(tiers):
        y = len(tiers) - i
        ax.barh(y, np.log2(size_b), height=0.62, color=COLORS[key],
                alpha=1.0 if i > 1 else 0.45, edgecolor="white", linewidth=0.8)
        val = f"{size_b:g} B" if size_b >= 1 else f"{size_b:.2f} B"
        ax.annotate(f"{val}  ({comp})", (np.log2(size_b), y), textcoords="offset points",
                    xytext=(6, -2.6), fontsize=7.2, color="#40444a")
        ax.annotate(label, (-0.15, y), ha="right", va="center", fontsize=7.6,
                    color="#40444a", annotation_clip=False)
    ax.set_yticks([])
    ax.set_xticks([np.log2(v) for v in (4, 16, 64, 256)], ["4 B", "16 B", "64 B", "256 B"])
    ax.set_xlim(0, 10.4)
    ax.set_xlabel("stored bytes per 128-dim token (log scale)")
    save(fig, "format_layout")


# ---------------------------------------------------------------- fig: scale flip
def fig_scale_flip():
    scales = ["256 docs\nDocVQA slice", "6.7k docs\nsuite (10 sets)", "10,171 docs\nunique mixed",
              "25,000 docs\nstress"]
    series = [
        ("int4", [0.0093, 0.0048, 0.0152, None]),
        ("pool3", [-0.0133, -0.0024, 0.0112, 0.0238]),
        ("pool2", [-0.0074, -0.0021, 0.0008, 0.0039]),
        ("ts", [0.0092, -0.0015, -0.0059, -0.0120]),
        ("u4", [0.0126, -0.0015, -0.0077, -0.0135]),
    ]
    suite_ci = {"ts": (-0.0026, -0.0004), "pool2": (-0.0040, -0.0003), "pool3": (-0.0057, -0.0017),
                "int4": (0.0014, 0.0054)}
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = list(range(len(scales)))
    ax.axhspan(-0.0005, 0.0005, color="#9aa1a9", alpha=0.14, lw=0)
    ax.axhline(0, color="#5c6672", linewidth=0.9, linestyle=(0, (4, 3)))
    for key, ys in series:
        xs = [i for i, v in zip(x, ys) if v is not None]
        vs = [v for v in ys if v is not None]
        ax.plot(xs, vs, "-", color=COLORS[key], linewidth=1.5, zorder=2)
        for xi, vi in zip(xs, vs):
            _mark(ax, xi, vi, key, size=30)
        end_x, end_v = xs[-1], vs[-1]
        ax.annotate(f"{end_v:+.4f}", (end_x, end_v), textcoords="offset points",
                    xytext=(8, -2.5), fontsize=6.6, color=COLORS[key], fontweight="bold")
        if key in suite_ci:
            lo, hi = suite_ci[key]
            ax.plot([1, 1], [lo, hi], color=COLORS[key], linewidth=2.6, alpha=0.32,
                    solid_capstyle="round", zorder=1)
    ax.text(2.55, 0.0012, "= plain binary", fontsize=7, color="#5c6672", ha="right")
    ax.set_xticks(x, scales, fontsize=7.5)
    ax.set_xlim(-0.2, 3.62)
    ax.set_ylabel(r"$\Delta$ NDCG@10 relative to plain binary")
    handles = [Line2D([], [], marker=MARKERS[k], linestyle="-", color=COLORS[k],
                      markersize=5, label=LABELS[k]) for k, _ in series]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left", handlelength=1.6,
              columnspacing=1.0, borderaxespad=0.1)
    save(fig, "scale_flip")


# ---------------------------------------------------------------- fig: pareto
def fig_pareto_10k():
    rows = _final_rows()
    pts = [
        ("dense", "dense_fp16_vectorized", 2.0, "dense fp16", (8, -18, "left")),
        ("int4", "bitmax_int4_dp4a", 8.0, "int4+dp4a", (0, 9, "center")),
        ("ts", "bitmax_binary_token_scale", 28.4, "fp16 scales", (-8, -14, "center")),
        ("u4", "bitmax_binary_token_scale_u4", 31.0, "u4 scales", (-36, 2, "center")),
        ("binary", "bitmax_binary", 32.0, "binary", (18, 7, "center")),
        ("pool2", "bitmax_pooled_binary", 63.9, "pool2", (0, -15, "center")),
        ("pool3", "bitmax_pooled_binary3", 95.9, "pool3", (0, 9, "center")),
    ]
    dense_y = rows["dense_fp16_vectorized"]["ndcg_at_10"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhspan(dense_y - 0.01, dense_y, color="#008300", alpha=0.07, lw=0)
    ax.axhline(dense_y, color="#5c6672", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(126, dense_y + 0.0012, "dense fp16", fontsize=7, color="#5c6672", ha="right")
    ax.text(13.5, dense_y - 0.0093, "within 0.01 of dense", fontsize=7, color="#3d7a3d",
            ha="left", style="italic")
    # pareto-efficient frontier among measured points (fast-plaid, int4, pool3)
    frontier = [(3.4, rows["fast_plaid"]["ndcg_at_10"]),
                (8.0, rows["bitmax_int4_dp4a"]["ndcg_at_10"]),
                (95.9, rows["bitmax_pooled_binary3"]["ndcg_at_10"])]
    fx, fy = [], []
    for i, (cx, cy) in enumerate(frontier):
        if i:
            fx.append(cx); fy.append(frontier[i - 1][1])
        fx.append(cx); fy.append(cy)
    ax.plot(fx, fy, color="#b9bec5", linewidth=1.1, linestyle="-", zorder=1)
    ax.text(17, 0.5035, "efficient frontier", fontsize=6.6, color="#8a919a", rotation=-4)
    for key, impl, comp, label, (dx, dy, ha) in pts:
        row = rows[impl]
        y = row["ndcg_at_10"]
        lat = row["latency_ms"] / 1000
        _mark(ax, comp, y, key, size=52)
        ax.annotate(f"{label}\n{y:.4f} · {lat:.2f} s", (comp, y), textcoords="offset points",
                    xytext=(dx, dy if dy > 0 else dy - 8), ha=ha, fontsize=6.6,
                    color=COLORS[key], fontweight="bold", linespacing=1.15)
    fp = rows.get("fast_plaid")
    if fp:
        _mark(ax, 3.4, fp["ndcg_at_10"], "ext", size=52)
        ax.annotate(f"fast-plaid\n{fp['ndcg_at_10']:.4f} · lat. n/c", (3.4, fp["ndcg_at_10"]),
                    textcoords="offset points", xytext=(2, 9), ha="center", fontsize=6.6,
                    color=COLORS["ext"], fontweight="bold", linespacing=1.15)
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128], ["2×", "4×", "8×", "16×", "32×", "64×", "128×"])
    ax.minorticks_off()
    ax.set_xlabel("compression vs. fp32 storage (log scale)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0.462, 0.532)
    ax.set_xlim(1.7, 135)
    save(fig, "pareto_10k")


# ---------------------------------------------------------------- fig: latency
def fig_latency_10k():
    rows = _final_rows()
    dense_v = rows["dense_fp16_vectorized"]["latency_ms"]
    order = [
        ("dense fp16 (loop impl.)", "dense_fp16_baseline", "dense", 0.4, False),
        ("dense fp16 (vectorized)", "dense_fp16_vectorized", "dense", 1.0, False),
        ("int4 fp32-query", "bitmax_int4", "int4", 0.4, False),
        ("int4 + dp4a", "bitmax_int4_dp4a", "int4", 1.0, True),
        ("fp16 token scales", "bitmax_binary_token_scale", "ts", 1.0, True),
        ("binary", "bitmax_binary", "binary", 1.0, True),
        ("pool3 binary", "bitmax_pooled_binary3", "pool3", 1.0, True),
        ("pool2 binary", "bitmax_pooled_binary", "pool2", 1.0, True),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.95))
    ax.grid(axis="y", visible=False)
    for i, (label, impl, key, alpha, speedup) in enumerate(order):
        v = rows[impl]["latency_ms"] / 1000
        y = len(order) - i
        ax.plot([0.28, v], [y, y], color="#e3e5e8", linewidth=1.2, zorder=1)
        _mark(ax, v, y, key, size=44, alpha=alpha)
        txt = f"{v:.2f} s" if v < 10 else f"{v:.0f} s"
        if speedup:
            txt += f"   ({dense_v/1000/v:.1f}× dense)"
        ax.annotate(txt, (v, y), textcoords="offset points", xytext=(9, -2.6),
                    fontsize=7, color="#40444a")
    ax.set_yticks([len(order) - i for i in range(len(order))], [o[0] for o in order], fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlim(0.28, 700)
    ax.set_xticks([1, 10, 100], ["1 s", "10 s", "100 s"])
    ax.minorticks_off()
    ax.set_xlabel("latency, 256 queries × 10,171 documents (log scale)")
    save(fig, "latency_10k")


# ------------------------------------------------------- fig: per-dataset deltas
def fig_per_dataset():
    """Delta-from-dense per dataset: differences readable at every magnitude."""
    keys = [("int4_int8q_dp4a", "int4"), ("bitmax_binary", "binary"), ("pool3_binary", "pool3")]
    natural, synthetic = [], []
    for path in sorted(RAW.glob("paper-vidore-*colqwen2*-r3.json")):
        data = json.loads(path.read_text())
        rows = {r["implementation"]: r for r in data["results"]}
        p3 = path.with_name(path.name.replace("paper-", "paper-pool3-", 1))
        if p3.exists():
            rows.update({r["implementation"]: r for r in json.loads(p3.read_text())["results"]
                         if r["implementation"] != "dense_fp16_baseline"})
        stem = path.stem.replace("paper-vidore-", "")
        pretty = {
            "docvqa-test-colqwen2-limit500-r3": "DocVQA",
            "infovqa-test-colqwen2-limit500-r3": "InfoVQA",
            "arxivqa-test-colqwen2-limit500-r3": "ArxivQA",
            "tabfquad-test-colqwen2-limit500-r3": "TabFQuAD",
            "tatdqa-test-colqwen2-limit1663-r3": "TAT-DQA",
            "syntheticdocqa-ai-colqwen2-limit1000-r3": "SynthDocQA/AI",
            "syntheticdocqa-energy-colqwen2-limit1000-r3": "SynthDocQA/Energy",
            "syntheticdocqa-government-colqwen2-limit1000-r3": "SynthDocQA/Gov.",
            "syntheticdocqa-healthcare-colqwen2-limit1000-r3": "SynthDocQA/Health",
            "syntheticdocqa-shift-colqwen2-limit1000-r3": "SynthDocQA/Shift",
        }.get(stem, stem)
        dense = rows["dense_fp16_baseline"]["ndcg_at_10"]
        deltas = {ck: rows[k]["ndcg_at_10"] - dense for k, ck in keys if k in rows}
        (synthetic if "Synth" in pretty else natural).append((pretty, dense, deltas))
    natural.sort(key=lambda t: -t[1])
    synthetic.sort(key=lambda t: -t[1])
    groups = natural + synthetic
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color="#5c6672", linewidth=0.9, linestyle=(0, (4, 3)))
    n = len(groups)
    gap = 0.9
    for i, (name, dense, deltas) in enumerate(groups):
        y = n - i + (gap if i < len(natural) else 0)
        vals = list(deltas.values())
        ax.plot([min(vals + [0]), max(vals + [0])], [y, y], color="#e6e8eb", linewidth=2.2,
                solid_capstyle="round", zorder=1)
        for ck, v in deltas.items():
            _mark(ax, v, y, ck, size=30)
        ax.annotate(f"{name}   (dense {dense:.3f})", (-0.0255, y), ha="right", va="center",
                    fontsize=7.2, color="#40444a", annotation_clip=False)
    ax.text(0.0005, n + gap + 0.75, "better than dense →", fontsize=6.6, color="#8a919a")
    ax.text(-0.0005, n + gap + 0.75, "← worse", fontsize=6.6, color="#8a919a", ha="right")
    ax.set_yticks([])
    ax.set_xlim(-0.025, 0.011)
    ax.set_xlabel(r"$\Delta$ NDCG@10 vs. dense fp16 (same dataset)")
    handles = [Line2D([], [], marker=MARKERS[c], linestyle="", color=COLORS[c], markersize=5,
                      label=LABELS[c]) for _, c in keys]
    ax.legend(handles=handles, fontsize=7, loc="lower left", handlelength=1.2)
    save(fig, "per_dataset")


# ---------------------------------------------------------------- fig: forest
def fig_forest():
    data = json.loads((RAW / "significance-suite.json").read_text())["pairs"]
    pretty = {
        "int4_int8q_dp4a": "int4+dp4a", "dense_fp16_baseline": "dense",
        "bitmax_binary": "binary", "pool2_binary": "pool2", "pool3_binary": "pool3",
        "binary_token_scale_fp16_cuda": "fp16 scales", "binary_token_scale_u4_cuda": "u4 scales",
    }
    color_of = {"int4+dp4a": "int4", "binary": "binary", "pool2": "pool2", "pool3": "pool3",
                "fp16 scales": "ts", "u4 scales": "u4", "dense": "dense"}
    rows = [(f"{pretty[p['a']]}  vs  {pretty[p['b']]}", p, color_of[pretty[p["a"]]]) for p in data]
    fig, ax = plt.subplots(figsize=(5.6, 0.28 * len(rows) + 0.7))
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color="#5c6672", linewidth=0.9, linestyle=(0, (4, 3)))
    for i, (label, p, ckey) in enumerate(rows):
        y = len(rows) - i
        ax.plot([p["ci95_low"], p["ci95_high"]], [y, y], color=COLORS[ckey], linewidth=1.7,
                solid_capstyle="round", zorder=2)
        _mark(ax, p["mean_delta"], y, ckey, size=30)
        sig = "" if p["sign_test_p"] < 0.05 else "  (n.s.)"
        ax.annotate(f"{p['mean_delta']:+.4f}{sig}", (p["ci95_high"], y), textcoords="offset points",
                    xytext=(8, -2.6), fontsize=6.8, color="#40444a")
        ax.annotate(label, (-0.0125, y), ha="right", va="center", fontsize=7.2,
                    color="#40444a", annotation_clip=False)
    ax.set_yticks([])
    ax.set_xlim(-0.0122, 0.0085)
    ax.set_xlabel(r"paired $\Delta$ NDCG@10 with 95% bootstrap CI  (8,443 queries, full suite)")
    save(fig, "forest")


if __name__ == "__main__":
    fig_format_layout()
    fig_scale_flip()
    fig_pareto_10k()
    fig_latency_10k()
    fig_per_dataset()
    fig_forest()
    print("figures written to", OUT_PDF, "and", OUT_PNG)
