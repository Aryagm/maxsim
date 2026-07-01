from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "docs/benchmark_results/raw/unique-public-10k-rich/"
    "vidore-mixed-public-unique-colqwen2-limit10000-comparison.json"
)
DEFAULT_OUTPUT_DIR = Path("docs/benchmark_results/plots")

LABELS = {
    "dense_fp16_baseline": "Dense fp16",
    "faiss_gpu_mean_pool_flat_ip": "FAISS pooled",
    "cuvs_gpu_mean_pool_flat_ip": "cuVS pooled",
    "fast_plaid": "fast-plaid",
    "bitmax_binary": "bitmax binary",
    "bitmax_binary_q40": "bitmax q40",
    "bitmax_int4": "bitmax int4",
}

FAMILIES = {
    "dense_fp16_baseline": "Dense MaxSim",
    "faiss_gpu_mean_pool_flat_ip": "Single-vector pooled",
    "cuvs_gpu_mean_pool_flat_ip": "Single-vector pooled",
    "fast_plaid": "Compressed late interaction",
    "bitmax_binary": "bitmax",
    "bitmax_binary_q40": "bitmax",
    "bitmax_int4": "bitmax",
}

POINT_COLORS = {
    "dense_fp16_baseline": "#374151",
    "faiss_gpu_mean_pool_flat_ip": "#f59e0b",
    "cuvs_gpu_mean_pool_flat_ip": "#d97706",
    "fast_plaid": "#2563eb",
    "bitmax_binary": "#059669",
    "bitmax_binary_q40": "#14b8a6",
    "bitmax_int4": "#7c3aed",
}

MARKETING_ORDER = (
    "bitmax_binary",
    "bitmax_int4",
    "fast_plaid",
    "dense_fp16_baseline",
)

ANNOTATION_OFFSETS = {
    "dense_fp16_baseline": (-72, 14),
    "faiss_gpu_mean_pool_flat_ip": (8, 12),
    "cuvs_gpu_mean_pool_flat_ip": (8, -22),
    "fast_plaid": (-30, 18),
    "bitmax_binary": (10, -18),
    "bitmax_binary_q40": (10, -42),
    "bitmax_int4": (10, 14),
}

LATENCY_OFFSETS = {
    "dense_fp16_baseline": (-92, -22),
    "faiss_gpu_mean_pool_flat_ip": (10, 18),
    "cuvs_gpu_mean_pool_flat_ip": (10, -18),
    "fast_plaid": (-96, 22),
    "bitmax_binary": (18, -25),
    "bitmax_binary_q40": (18, -58),
    "bitmax_int4": (18, -30),
}

STORAGE_OFFSETS = {
    "dense_fp16_baseline": (-78, 22),
    "faiss_gpu_mean_pool_flat_ip": (-116, 18),
    "cuvs_gpu_mean_pool_flat_ip": (-110, -34),
    "fast_plaid": (-54, 34),
    "bitmax_binary": (18, -22),
    "bitmax_binary_q40": (18, -58),
    "bitmax_int4": (18, 22),
}

SPEEDUP_OFFSETS = {
    "faiss_gpu_mean_pool_flat_ip": (18, 30),
    "cuvs_gpu_mean_pool_flat_ip": (-20, -46),
    "fast_plaid": (-42, 34),
    "bitmax_binary": (20, -8),
    "bitmax_binary_q40": (20, -40),
    "bitmax_int4": (20, 28),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate seaborn benchmark figures for bitmax.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    args = parser.parse_args(argv)

    pd, sns, plt, ticker = _import_viz()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip())

    frame, metadata = _load_frame(args.input, pd)
    _configure_theme(sns, plt)
    _plot_marketing_scorecard(frame, metadata, output_dir, formats, sns, plt)
    _plot_latency_quality(frame, metadata, output_dir, formats, sns, plt, ticker)
    _plot_storage_quality(frame, metadata, output_dir, formats, sns, plt, ticker)
    _plot_speedup_quality_loss(frame, metadata, output_dir, formats, sns, plt, ticker)
    print(f"wrote {len(formats) * 4} figure files to {output_dir}")


def _import_viz():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - exercised by users without viz extra
        raise SystemExit("Install visualization dependencies with: python -m pip install -e '.[viz]'") from exc
    return pd, sns, plt, ticker


def _load_frame(path: Path, pd):
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for row in payload["results"]:
        if row.get("status") != "ok":
            continue
        implementation = str(row["implementation"])
        rows.append(
            {
                "implementation": implementation,
                "label": LABELS.get(implementation, implementation),
                "family": FAMILIES.get(implementation, "Other"),
                "latency_p95_ms": float(row["latency_p95_ms"]),
                "latency_p95_s": float(row["latency_p95_ms"]) / 1000.0,
                "speedup_vs_dense_fp16": float(row.get("speedup_vs_dense_fp16") or 1.0),
                "doc_storage_kib_per_doc": float(row["doc_storage_bytes"]) / float(row["docs"]) / 1024.0,
                "doc_memory_compression_vs_fp32": float(row["doc_memory_compression_vs_fp32"]),
                "recall_at_10": float(row["recall_at_10"]),
                "ndcg_at_10": float(row["ndcg_at_10"]),
                "quality_delta_vs_dense_ndcg_at_10": float(row["quality_delta_vs_dense_ndcg_at_10"]),
                "ndcg_loss_vs_dense": -float(row["quality_delta_vs_dense_ndcg_at_10"]),
            }
        )
    frame = pd.DataFrame(rows)
    dense_ndcg = float(frame.loc[frame["implementation"] == "dense_fp16_baseline", "ndcg_at_10"].iloc[0])
    frame["quality_retained_pct"] = frame["ndcg_at_10"] / dense_ndcg * 100.0
    metadata = {
        "docs": int(payload["dataset"]["docs"]),
        "queries": int(payload["dataset"]["queries"]),
        "doc_tokens": int(payload["dataset"]["doc_tokens"]),
        "name": str(payload["dataset"]["name"]),
    }
    return frame, metadata


def _configure_theme(sns, plt) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "axes.titlesize": 23,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "svg.hashsalt": "bitmax-benchmark-plots",
        }
    )


def _plot_marketing_scorecard(frame, metadata, output_dir: Path, formats, sns, plt) -> None:
    scorecard = _ordered_frame(frame, MARKETING_ORDER).copy()
    binary = scorecard.loc[scorecard["implementation"] == "bitmax_binary"].iloc[0]
    dense = scorecard.loc[scorecard["implementation"] == "dense_fp16_baseline"].iloc[0]
    specs = (
        (
            "quality_retained_pct",
            "Accuracy",
            "NDCG@10 retained, higher is better",
            (0, 108),
            lambda value: f"{value:.1f}%",
        ),
        (
            "latency_p95_s",
            "Latency",
            "P95 seconds, lower is better",
            (0, 85),
            lambda value: f"{value:.2f}s",
        ),
        (
            "doc_storage_kib_per_doc",
            "Size",
            "KiB per document, lower is better",
            (0, 210),
            lambda value: f"{value:.1f} KiB",
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), sharey=True)
    labels = list(scorecard["label"])
    for idx, (ax, (metric, title, xlabel, xlim, formatter)) in enumerate(zip(axes, specs, strict=True)):
        sns.barplot(data=scorecard, x=metric, y="label", order=labels, color="#d1d5db", ax=ax)
        for patch, row in zip(ax.patches, scorecard.itertuples(index=False), strict=True):
            patch.set_facecolor(POINT_COLORS[str(row.implementation)])
            patch.set_edgecolor("white")
            patch.set_linewidth(1.2)
            width = patch.get_width()
            label_x = width + (xlim[1] - xlim[0]) * 0.018
            ax.text(
                label_x,
                patch.get_y() + patch.get_height() / 2,
                formatter(float(width)),
                va="center",
                ha="left",
                fontsize=13,
                fontweight="bold",
                color="#111827",
            )
        ax.set_xlim(*xlim)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=14)
        ax.set_xlabel(xlabel, fontsize=12, color="#4b5563")
        ax.set_ylabel("")
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.9)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=11, colors="#6b7280")
        ax.tick_params(axis="y", labelsize=13, colors="#111827")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False, length=0)

    fig.suptitle(
        "Near-dense quality with much lower latency and storage",
        x=0.025,
        y=0.965,
        ha="left",
        fontsize=25,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.025,
        0.905,
        f"bitmax binary keeps {binary.quality_retained_pct:.1f}% of dense NDCG@10, while cutting P95 latency "
        f"from {dense.latency_p95_s:.1f}s to {binary.latency_p95_s:.2f}s and storage from "
        f"{dense.doc_storage_kib_per_doc:.1f} KiB/doc to {binary.doc_storage_kib_per_doc:.1f} KiB/doc.",
        ha="left",
        fontsize=15,
        color="#374151",
    )
    fig.text(
        0.025,
        0.035,
        f"{metadata['docs']:,} unique docs, {metadata['queries']} queries on RTX 4090. "
        "Pooled single-vector baselines are omitted here because they lose most "
        "late-interaction quality; see the full Pareto plots for that context.",
        fontsize=10.5,
        color="#6b7280",
    )
    fig.tight_layout(rect=(0.02, 0.09, 1.0, 0.88), w_pad=2.6)
    _save(fig, output_dir / "unique10k_marketing_scorecard", formats)
    plt.close(fig)


def _plot_latency_quality(frame, metadata, output_dir: Path, formats, sns, plt, ticker) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=frame,
        x="latency_p95_ms",
        y="ndcg_at_10",
        hue="implementation",
        size="doc_memory_compression_vs_fp32",
        sizes=(180, 980),
        palette=POINT_COLORS,
        legend=False,
        edgecolor="white",
        linewidth=1.4,
        ax=ax,
    )
    _annotate_points(ax, frame, "latency_p95_ms", "ndcg_at_10", LATENCY_OFFSETS)
    dense = float(frame.loc[frame["implementation"] == "dense_fp16_baseline", "ndcg_at_10"].iloc[0])
    ax.axhline(dense, color="#9ca3af", linestyle=(0, (4, 4)), linewidth=1.2)
    ax.text(22, dense + 0.012, "Dense quality reference", color="#6b7280", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 110000)
    ax.set_ylim(0.0, 0.58)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_latency_ms))
    ax.set_xlabel("P95 latency, log scale")
    ax.set_ylabel("NDCG@10")
    ax.set_title("10k unique docs: bitmax keeps late-interaction quality while cutting latency")
    ax.text(
        0.01,
        -0.18,
        f"{metadata['docs']:,} unique docs, {metadata['queries']} queries, {metadata['doc_tokens']:,} doc tokens. "
        "Bubble size = fp32 document-storage reduction.",
        transform=ax.transAxes,
        fontsize=10,
        color="#6b7280",
    )
    _save(fig, output_dir / "unique10k_latency_quality_pareto", formats)
    plt.close(fig)


def _plot_storage_quality(frame, metadata, output_dir: Path, formats, sns, plt, ticker) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=frame,
        x="doc_memory_compression_vs_fp32",
        y="ndcg_at_10",
        hue="implementation",
        size="latency_p95_s",
        sizes=(180, 980),
        palette=POINT_COLORS,
        legend=False,
        edgecolor="white",
        linewidth=1.4,
        ax=ax,
    )
    _annotate_points(ax, frame, "doc_memory_compression_vs_fp32", "ndcg_at_10", STORAGE_OFFSETS)
    ax.set_xscale("log")
    ax.set_xlim(1.4, 1400)
    ax.set_ylim(0.0, 0.56)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}x"))
    ax.axvline(32, color="#059669", linestyle=(0, (4, 4)), linewidth=1.2)
    ax.text(34, 0.045, "32x binary target", color="#047857", fontsize=11)
    ax.set_xlabel("Document-storage reduction vs fp32, log scale")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Storage vs quality: pooled vectors are tiny, but lose late-interaction signal")
    ax.text(
        0.01,
        -0.18,
        f"{metadata['docs']:,} unique docs. Bubble size = P95 latency, so smaller bubbles are faster.",
        transform=ax.transAxes,
        fontsize=10,
        color="#6b7280",
    )
    _save(fig, output_dir / "unique10k_storage_quality_pareto", formats)
    plt.close(fig)


def _plot_speedup_quality_loss(frame, metadata, output_dir: Path, formats, sns, plt, ticker) -> None:
    non_dense = frame[frame["implementation"] != "dense_fp16_baseline"].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=non_dense,
        x="speedup_vs_dense_fp16",
        y="ndcg_loss_vs_dense",
        hue="implementation",
        size="doc_memory_compression_vs_fp32",
        sizes=(180, 980),
        palette=POINT_COLORS,
        legend=False,
        edgecolor="white",
        linewidth=1.4,
        ax=ax,
    )
    _annotate_points(ax, non_dense, "speedup_vs_dense_fp16", "ndcg_loss_vs_dense", SPEEDUP_OFFSETS)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 180000)
    ax.set_ylim(-0.015, 0.55)
    ax.axhline(0.0, color="#9ca3af", linestyle=(0, (4, 4)), linewidth=1.2)
    ax.axhspan(-0.015, 0.025, color="#ecfdf5", zorder=0)
    ax.text(1.0, 0.012, "Near-dense quality band", color="#047857", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}x"))
    ax.set_xlabel("Speedup vs dense fp16, log scale")
    ax.set_ylabel("NDCG@10 loss vs dense fp16")
    ax.set_title("Speedup vs quality loss: the usable frontier is compressed late interaction")
    ax.text(
        0.01,
        -0.18,
        "Lower is better on the y-axis. Bubble size = fp32 document-storage reduction.",
        transform=ax.transAxes,
        fontsize=10,
        color="#6b7280",
    )
    _save(fig, output_dir / "unique10k_speedup_quality_loss", formats)
    plt.close(fig)


def _ordered_frame(frame, implementations: tuple[str, ...]):
    return frame.set_index("implementation").loc[list(implementations)].reset_index()


def _annotate_points(ax, frame, x_col: str, y_col: str, offsets: dict[str, tuple[int, int]] | None = None) -> None:
    for _, row in frame.iterrows():
        impl = str(row["implementation"])
        dx, dy = (offsets or ANNOTATION_OFFSETS).get(impl, ANNOTATION_OFFSETS.get(impl, (8, 8)))
        ax.annotate(
            str(row["label"]),
            (float(row[x_col]), float(row[y_col])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            color="#111827",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.7, "shrinkA": 0, "shrinkB": 5},
        )


def _format_latency_ms(value: float, _: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}s"
    return f"{value:g}ms"


def _save(fig, base_path: Path, formats: tuple[str, ...]) -> None:
    for fmt in formats:
        kwargs = {"metadata": {"Date": None}} if fmt == "svg" else {}
        fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=220, **kwargs)


if __name__ == "__main__":
    main()
