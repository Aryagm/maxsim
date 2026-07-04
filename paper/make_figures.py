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
from matplotlib.lines import Line2D

RAW = Path("docs/benchmark_results/raw")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "binary": "#2a78d6",
    "ts": "#0e9668",
    "u4": "#c98500",
    "int4": "#008300",
    "pool2": "#4a3aa7",
    "pool3": "#d43d3c",
    "dense": "#5c6672",
    "ext": "#8a5a1e",
}
MARKERS = {"binary": "o", "ts": "s", "u4": "D", "int4": "^", "pool2": "v", "pool3": "P", "dense": "o", "ext": "X"}
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


def _mark(ax, x, y, key, size=46, alpha=1.0, zorder=3):
    ax.scatter([x], [y], s=size, marker=MARKERS[key], color=COLORS[key], alpha=alpha,
               edgecolor="white", linewidth=1.1, zorder=zorder)


def fig_scale_flip():
    scales = ["256 docs\nDocVQA slice", "6.7k docs\nsuite (10 sets)", "10,171 docs\nunique mixed", "25,000 docs\nstress"]
    series = [
        ("int4", [0.0093, 0.0048, 0.0152, None]),
        ("pool3", [-0.0133, -0.0024, 0.0112, 0.0238]),
        ("pool2", [-0.0074, -0.0021, 0.0008, 0.0039]),
        ("ts", [0.0092, -0.0015, -0.0059, -0.0120]),
        ("u4", [0.0126, -0.0015, -0.0077, -0.0135]),
    ]
    # suite-level 95% CIs from the paired bootstrap (significance-suite.json)
    suite_ci = {"ts": (-0.0026, -0.0004), "pool2": (-0.0040, -0.0003), "pool3": (-0.0057, -0.0017),
                "int4": (0.0014, 0.0054)}
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = list(range(len(scales)))
    ax.axhspan(-0.0005, 0.0005, color="#9aa1a9", alpha=0.14, lw=0)
    ax.axhline(0, color="#5c6672", linewidth=0.9, linestyle=(0, (4, 3)))
    for key, ys in series:
        xs = [i for i, v in zip(x, ys) if v is not None]
        vs = [v for v in ys if v is not None]
        ax.plot(xs, vs, "-", color=COLORS[key], linewidth=1.5, zorder=2)
        for xi, vi in zip(xs, vs):
            _mark(ax, xi, vi, key, size=30)
        if key in suite_ci:
            lo, hi = suite_ci[key]
            ax.plot([1, 1], [lo, hi], color=COLORS[key], linewidth=2.6, alpha=0.32,
                    solid_capstyle="round", zorder=1)
    ax.text(3.02, 0.0012, "= plain binary", fontsize=7, color="#5c6672", ha="right")
    ax.set_xticks(x, scales, fontsize=7.5)
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylabel(r"$\Delta$ NDCG@10 relative to plain binary")
    handles = [Line2D([], [], marker=MARKERS[k], linestyle="-", color=COLORS[k],
                      markersize=5, label=LABELS[k]) for k, _ in series]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left", handlelength=1.6,
              columnspacing=1.0, borderaxespad=0.1)
    fig.savefig(OUT / "scale_flip.pdf")


def _final_rows():
    data = json.loads((RAW / "paper-unique-10k-final.json").read_text())
    return {r["implementation"]: r for r in data["results"] if not r.get("status") or r["status"] == "ok"}


def fig_pareto_10k():
    rows = _final_rows()
    pts = [
        ("dense", "dense_fp16_vectorized", 2.0, "dense fp16", (8, -18, "left")),
        ("int4", "bitmax_int4_dp4a", 8.0, "int4+dp4a", (0, 9, "center")),
        ("ts", "bitmax_binary_token_scale", 28.4, "fp16 scales", (-8, -14, "center")),
        ("u4", "bitmax_binary_token_scale_u4", 31.0, "u4 scales", (-34, 2, "center")),
        ("binary", "bitmax_binary", 32.0, "binary", (16, 7, "center")),
        ("pool2", "bitmax_pooled_binary", 63.9, "pool2", (0, -15, "center")),
        ("pool3", "bitmax_pooled_binary3", 95.9, "pool3", (0, 9, "center")),
    ]
    dense_y = rows["dense_fp16_vectorized"]["ndcg_at_10"]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axhspan(dense_y - 0.01, dense_y, color="#008300", alpha=0.07, lw=0)
    ax.axhline(dense_y, color="#5c6672", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(126, dense_y + 0.0012, "dense fp16", fontsize=7, color="#5c6672", ha="right")
    ax.text(13, dense_y - 0.0093, "within 0.01 of dense", fontsize=7, color="#3d7a3d",
            ha="left", style="italic")
    for key, impl, comp, label, (dx, dy, ha) in pts:
        row = rows[impl]
        y = row["ndcg_at_10"]
        lat = row["latency_ms"] / 1000
        _mark(ax, comp, y, key, size=52)
        ax.annotate(f"{label}\n{lat:.2f} s", (comp, y), textcoords="offset points",
                    xytext=(dx, dy if dy > 0 else dy - 8), ha=ha, fontsize=6.8,
                    color=COLORS[key], fontweight="bold", linespacing=1.1)
    fp = rows.get("fast_plaid")
    if fp:
        _mark(ax, 3.4, fp["ndcg_at_10"], "ext", size=52)
        ax.annotate("fast-plaid\n(latency n/c)", (3.4, fp["ndcg_at_10"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6.8, color=COLORS["ext"],
                    fontweight="bold", linespacing=1.1)
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128], ["2×", "4×", "8×", "16×", "32×", "64×", "128×"])
    ax.minorticks_off()
    ax.set_xlabel("compression vs. fp32 storage (log scale)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0.462, 0.53)
    ax.set_xlim(1.7, 135)
    fig.savefig(OUT / "pareto_10k.pdf")


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
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
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
    fig.savefig(OUT / "latency_10k.pdf")


def fig_per_dataset():
    keys = [("dense_fp16_baseline", "dense"), ("int4_int8q_dp4a", "int4"),
            ("bitmax_binary", "binary"), ("pool3_binary", "pool3")]
    natural, synthetic = [], []
    for path in sorted(RAW.glob("paper-vidore-*colqwen2*-r3.json")):
        data = json.loads(path.read_text())
        rows = {r["implementation"]: r for r in data["results"]}
        p3 = path.with_name(path.name.replace("paper-", "paper-pool3-", 1))
        if p3.exists():
            rows.update({r["implementation"]: r for r in json.loads(p3.read_text())["results"]
                         if r["implementation"] != "dense_fp16_baseline"})
        stem = path.stem.replace("paper-vidore-", "").replace("-colqwen2", "").replace("-r3", "")
        pretty = {
            "docvqa-test-limit500": "DocVQA (500)", "infovqa-test-limit500": "InfoVQA (500)",
            "arxivqa-test-limit500": "ArxivQA (500)", "tabfquad-test-limit500": "TabFQuAD (280 q)",
            "tatdqa-test-limit1663": "TAT-DQA (1,663 q)",
            "syntheticdocqa-ai-limit1000": "SynthDocQA / AI",
            "syntheticdocqa-energy-limit1000": "SynthDocQA / Energy",
            "syntheticdocqa-government-limit1000": "SynthDocQA / Gov.",
            "syntheticdocqa-healthcare-limit1000": "SynthDocQA / Health",
            "syntheticdocqa-shift-limit1000": "SynthDocQA / Shift",
        }.get(stem, stem)
        (synthetic if "Synth" in pretty else natural).append((pretty, rows))
    natural.sort(key=lambda t: -t[1]["dense_fp16_baseline"]["ndcg_at_10"])
    synthetic.sort(key=lambda t: -t[1]["dense_fp16_baseline"]["ndcg_at_10"])
    groups = natural + synthetic
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.grid(axis="y", visible=False)
    n = len(groups)
    gap = 0.9
    for i, (name, rows) in enumerate(groups):
        y = n - i + (gap if i < len(natural) else 0)
        vals = [rows[k]["ndcg_at_10"] for k, _ in keys if k in rows]
        ax.plot([min(vals), max(vals)], [y, y], color="#d5d8dc", linewidth=2.4,
                solid_capstyle="round", zorder=1)
        for k, ckey in keys:
            if k in rows:
                _mark(ax, rows[k]["ndcg_at_10"], y, ckey, size=26)
        ax.annotate(name, (-0.015, y), ha="right", va="center", fontsize=7.4,
                    color="#40444a", annotation_clip=False)
    ax.axhline(n - len(natural) + 1 + gap / 2, color="#e3e5e8", linewidth=0.7)
    ax.text(1.0, n - len(natural) + 1 + gap / 2 + 0.15, "hard synthetic query sets",
            fontsize=6.6, color="#8a919a", ha="right", style="italic")
    ax.set_yticks([])
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("NDCG@10")
    handles = [Line2D([], [], marker=MARKERS[c], linestyle="", color=COLORS[c], markersize=5,
                      label=LABELS[c]) for _, c in keys]
    ax.legend(handles=handles, fontsize=7, loc="center right", handlelength=1.2)
    fig.savefig(OUT / "per_dataset.pdf")


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
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
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
    fig.savefig(OUT / "forest.pdf")


if __name__ == "__main__":
    fig_scale_flip()
    fig_pareto_10k()
    fig_latency_10k()
    fig_per_dataset()
    fig_forest()
    print("figures written to", OUT)
