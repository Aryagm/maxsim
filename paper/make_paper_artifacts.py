"""Generate the model-aware paper tables and figures.

The 2026-07-15 archive is the authoritative source for new model, production,
scale, PLAID, and reducer results. Historical ColQwen2 dense/binary/pool rows
are joined from the repository legacy ledger because the follow-up archive only
contains its new int4/int8 controls.

Run from the repository root:
    .venv/bin/python paper/make_paper_artifacts.py
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "benchmark-results" / "paper-20260715"
LEGACY = ROOT / "docs" / "benchmark_results" / "raw"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"
README_FIGURES = ROOT / "docs" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
README_FIGURES.mkdir(parents=True, exist_ok=True)


FORMATS = [
    "dense_fp16",
    "binary",
    "binary_token_scale_u4",
    "pool3_binary",
    "int4_per_tensor",
    "int4_per_token",
    "int8_per_token",
]

COMP_FP32 = {
    "dense_fp16": 2.0,
    "binary": 32.0,
    "binary_token_scale_u4": 31.03,
    "pool3_binary": 95.9,
    "int4_per_tensor": 8.0,
    "int4_per_token": 7.53,
    "int8_per_token": 3.88,
    "int4_residual": 3.76,
}

LABELS = {
    "dense_fp16": "dense fp16",
    "binary": "binary",
    "binary_token_scale_u4": "scaled binary",
    "pool3_binary": "pool3",
    "int4_per_tensor": "int4/tensor",
    "int4_per_token": "int4/token",
    "int8_per_token": "int8/token",
    "int4_residual": "residual int4",
}

SHORT_LABELS = {
    "dense_fp16": "Dense",
    "binary": "Binary",
    "binary_token_scale_u4": "Scaled B1",
    "pool3_binary": "Pool3",
    "int4_per_tensor": "I4/tensor",
    "int4_per_token": "I4/token",
    "int8_per_token": "I8/token",
    "int4_residual": "Residual",
}

COLORS = {
    "dense_fp16": "#5F6872",
    "binary": "#2F6B9A",
    "binary_token_scale_u4": "#D08C2F",
    "pool3_binary": "#B64B3C",
    "int4_per_tensor": "#16857A",
    "int4_per_token": "#5A7D3B",
    "int8_per_token": "#7C5AA6",
    "int4_residual": "#9A6677",
}

MARKERS = {
    "dense_fp16": "o",
    "binary": "s",
    "binary_token_scale_u4": "D",
    "pool3_binary": "P",
    "int4_per_tensor": "^",
    "int4_per_token": "v",
    "int8_per_token": "X",
    "int4_residual": "h",
}

DATASET_LABELS = {
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "arxivqa": "ArxivQA",
    "docvqa": "DocVQA",
    "infovqa": "InfoVQA",
    "tabfquad": "TabFQuAD",
    "tatdqa": "TAT-DQA",
    "syntheticdocqa-ai": "Synth/AI",
    "syntheticdocqa-energy": "Synth/Energy",
    "syntheticdocqa-government": "Synth/Gov.",
    "syntheticdocqa-healthcare": "Synth/Health",
    "syntheticdocqa-shift": "Synth/Shift",
}

LEGACY_IMPL = {
    "dense_fp16_baseline": "dense_fp16",
    "bitmax_binary": "binary",
    "binary_token_scale_u4_cuda": "binary_token_scale_u4",
    "pool3_binary": "pool3_binary",
}

MODE_TO_FORMAT = {
    "dense": "dense_fp16",
    "binary": "binary",
    "binary_token_scale_u4": "binary_token_scale_u4",
    "pooled_binary": "pool3_binary",
    "int4": "int4_per_tensor",
    "int4_per_token": "int4_per_token",
    "int4_residual": "int4_residual",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": "#E1E5E8",
        "grid.linewidth": 0.55,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def require_complete(data: dict, path: Path) -> None:
    status = data.get("status")
    if status not in (None, "complete", "ok"):
        raise ValueError(f"{path}: incomplete status {status!r}")


def dataset_key(name: str) -> str:
    value = name.lower().replace("_test_subsampled", "").replace("_test", "")
    value = value.replace("shiftproject", "syntheticdocqa-shift")
    value = value.replace("syntheticdocqa_artificial_intelligence", "syntheticdocqa-ai")
    value = value.replace("syntheticdocqa_government_reports", "syntheticdocqa-government")
    value = value.replace("syntheticdocqa_healthcare_industry", "syntheticdocqa-healthcare")
    value = value.replace("syntheticdocqa_", "syntheticdocqa-")
    for key in sorted(DATASET_LABELS, key=len, reverse=True):
        if key in value:
            return key
    raise ValueError(f"unrecognized dataset name: {name}")


def load_quality(path: Path) -> tuple[str, dict[str, float], int, int]:
    data = read_json(path)
    require_complete(data, path)
    name = data["dataset"]["name"]
    key = dataset_key(name)
    rows = {}
    for row in data["results"]:
        if row.get("status") not in (None, "complete", "ok"):
            raise ValueError(f"{path}: incomplete row {row.get('format')}")
        rows[row["format"]] = float(row["ndcg_at_10"])
    return (
        key,
        rows,
        int(data["dataset"]["docs"]),
        int(data["dataset"].get("queries_evaluated", data["config"]["query_count"])),
    )


def load_suite(pattern: str, base: Path) -> dict[str, dict[str, float]]:
    suite = {}
    for path in sorted(base.glob(pattern)):
        key, rows, _, _ = load_quality(path)
        if key in suite:
            raise ValueError(f"duplicate quality dataset {key} in {pattern}")
        suite[key] = rows
    if not suite:
        raise ValueError(f"no results matched {base / pattern}")
    return suite


def load_colqwen2() -> dict[str, dict[str, float]]:
    suite: dict[str, dict[str, float]] = {}
    for path in sorted(LEGACY.glob("paper-vidore-*-colqwen2-*-r3.json")):
        data = read_json(path)
        key = dataset_key(data["dataset"]["name"])
        rows = suite.setdefault(key, {})
        for row in data["results"]:
            fmt = LEGACY_IMPL.get(row["implementation"])
            if fmt:
                rows[fmt] = float(row["ndcg_at_10"])

        pool_path = path.with_name(path.name.replace("paper-", "paper-pool3-", 1))
        if pool_path.exists():
            for row in read_json(pool_path)["results"]:
                fmt = LEGACY_IMPL.get(row["implementation"])
                if fmt:
                    rows[fmt] = float(row["ndcg_at_10"])

    new_paths = list(
        (ARCHIVE / "systems" / "results").glob(
            "quality-vidore-*-colqwen2-*-new-int4.json"
        )
    )
    new_paths += [
        ARCHIVE
        / "models"
        / "results"
        / "quality-vidore-tabfquad-colqwen2-new-int4.json"
    ]
    for path in sorted(new_paths):
        key, rows, _, _ = load_quality(path)
        suite.setdefault(key, {}).update(rows)

    expected = set(DATASET_LABELS) - {"fiqa", "nfcorpus", "scifact"}
    if set(suite) != expected:
        missing = sorted(expected - set(suite))
        extra = sorted(set(suite) - expected)
        raise ValueError(f"ColQwen2 join mismatch; missing={missing}, extra={extra}")
    for key, rows in suite.items():
        missing = set(FORMATS) - set(rows)
        if missing:
            raise ValueError(f"ColQwen2 {key} missing formats: {sorted(missing)}")
    return suite


def suite_means(suite: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        fmt: float(np.mean([rows[fmt] for rows in suite.values()]))
        for fmt in FORMATS
    }


def quality_suites() -> dict[str, dict[str, dict[str, float]]]:
    suites = {
        "Jina": load_suite(
            "quality-beir-*-jina-colbert-v2-full.json",
            ARCHIVE / "models" / "results",
        ),
        "GTE": load_suite(
            "quality-gte-*-full.json", ARCHIVE / "systems" / "results"
        ),
        "ColPali": load_suite(
            "quality-colpali-v1.3-*-full.json", ARCHIVE / "models" / "results"
        ),
        "ColQwen2": load_colqwen2(),
    }
    expected_counts = {"Jina": 3, "GTE": 3, "ColPali": 4, "ColQwen2": 10}
    actual = {name: len(rows) for name, rows in suites.items()}
    if actual != expected_counts:
        raise ValueError(f"quality suite count mismatch: {actual}")
    return suites


def tex_table(path: Path, columns: str, header: list[str], rows: list[list[str]]) -> None:
    lines = [
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_model_matrix(suites: dict[str, dict[str, dict[str, float]]]) -> None:
    rows = []
    for name in ("Jina", "GTE", "ColPali", "ColQwen2"):
        means = suite_means(suites[name])
        dense = means["dense_fp16"]
        cells = [name, f"{dense:.4f}"]
        for fmt in FORMATS[1:]:
            retention = 100.0 * means[fmt] / dense
            cells.append(f"{means[fmt]:.4f} ({retention:.1f}\\%)")
        rows.append(cells)
    tex_table(
        TABLES / "model_matrix.tex",
        "lrrrrrrr",
        [
            "model",
            "dense",
            "binary",
            "scaled B1",
            "pool3",
            "I4/tensor",
            "I4/token",
            "I8/token",
        ],
        rows,
    )


def write_per_dataset_tables(
    suites: dict[str, dict[str, dict[str, float]]]
) -> None:
    header = [
        "model / dataset",
        "dense",
        "binary",
        "scaled B1",
        "pool3",
        "I4/tensor",
        "I4/token",
        "I8/token",
    ]
    text_rows = []
    for model in ("Jina", "GTE"):
        for key in ("fiqa", "nfcorpus", "scifact"):
            values = suites[model][key]
            text_rows.append(
                [f"{model} / {DATASET_LABELS[key]}"]
                + [f"{values[fmt]:.4f}" for fmt in FORMATS]
            )
    tex_table(
        TABLES / "per_dataset_text.tex",
        "lrrrrrrr",
        header,
        text_rows,
    )

    visual_rows = []
    colpali_order = ("arxivqa", "docvqa", "infovqa", "tabfquad")
    colqwen_order = (
        "arxivqa",
        "docvqa",
        "infovqa",
        "tabfquad",
        "tatdqa",
        "syntheticdocqa-ai",
        "syntheticdocqa-energy",
        "syntheticdocqa-government",
        "syntheticdocqa-healthcare",
        "syntheticdocqa-shift",
    )
    for model, order in (("ColPali", colpali_order), ("ColQwen2$^\\dagger$", colqwen_order)):
        for key in order:
            values = suites[model.replace("$^\\dagger$", "")][key]
            visual_rows.append(
                [f"{model} / {DATASET_LABELS[key]}"]
                + [f"{values[fmt]:.4f}" for fmt in FORMATS]
            )
    tex_table(
        TABLES / "per_dataset_visual.tex",
        "lrrrrrrr",
        header,
        visual_rows,
    )


def production_data() -> dict[str, list[dict]]:
    paths = {
        "Visual 10k": ARCHIVE
        / "systems"
        / "results"
        / "production-visual10k-2kq.json",
        "FiQA 57k": ARCHIVE / "systems" / "results" / "production-fiqa57k.json",
    }
    output = {}
    for corpus, path in paths.items():
        data = read_json(path)
        require_complete(data, path)
        cases = []
        for case in data["cases"]:
            if case["case_id"].startswith("int4_residual_cascade"):
                continue
            fmt = MODE_TO_FORMAT.get(case["mode"])
            if not fmt:
                continue
            quality = case["quality"]
            storage = case["storage"]
            cases.append(
                {
                    "format": fmt,
                    "ndcg": float(quality["ndcg_at_10_mean"]),
                    "recall": float(quality["recall_at_10_mean"]),
                    "p50": float(case["latency"]["p50_ms"]),
                    "p95": float(case["latency"]["p95_ms"]),
                    "qps": float(case["throughput"]["queries_per_second"]),
                    "encoded": int(storage["encoded_bytes"]),
                    "serialized": storage.get("serialized_bytes"),
                }
            )
        dense = next(row for row in cases if row["format"] == "dense_fp16")
        for row in cases:
            row["speedup"] = dense["p50"] / row["p50"]
            row["compression"] = dense["encoded"] / row["encoded"]
        output[corpus] = cases
    return output


def write_production_table(production: dict[str, list[dict]]) -> None:
    rows = []
    order = [
        "dense_fp16",
        "binary",
        "binary_token_scale_u4",
        "pool3_binary",
        "int4_per_tensor",
        "int4_per_token",
        "int4_residual",
    ]
    for corpus in ("Visual 10k", "FiQA 57k"):
        by_format = {row["format"]: row for row in production[corpus]}
        for fmt in order:
            row = by_format[fmt]
            rows.append(
                [
                    corpus,
                    SHORT_LABELS[fmt],
                    f"{row['ndcg']:.4f}",
                    f"{row['p50']:.2f}",
                    f"{row['p95']:.2f}",
                    f"{row['qps']:.1f}",
                    f"{row['encoded'] / 2**20:.1f}",
                    f"{row['speedup']:.2f}$\\times$",
                    f"{row['compression']:.1f}$\\times$",
                ]
            )
    tex_table(
        TABLES / "production_end_to_end.tex",
        "llrrrrrrr",
        [
            "corpus",
            "format",
            "\\ndcg{}",
            "p50 ms",
            "p95 ms",
            "QPS",
            "MiB",
            "speedup",
            "compr.",
        ],
        rows,
    )


def plaid_data() -> dict[str, dict]:
    paths = {
        "Visual 10k": ARCHIVE
        / "systems"
        / "results"
        / "plaid-visual10k-2kq.json",
        "FiQA 57k": ARCHIVE / "systems" / "results" / "plaid-fiqa57k.json",
    }
    output = {}
    for corpus, path in paths.items():
        data = read_json(path)
        require_complete(data, path)
        output[corpus] = {
            "ndcg": float(data["quality"]["ndcg_at_10"]),
            "recall": float(data["quality"]["recall_at_10"]),
            "p50": float(data["search"]["batch1"]["latency_p50_ms"]),
            "p95": float(data["search"]["batch1"]["latency_p95_ms"]),
            "bytes": int(data["build"]["index_bytes"]),
            "build_s": float(data["build"]["latency_ms"]) / 1000.0,
        }
    return output


def write_plaid_table(
    production: dict[str, list[dict]], plaid: dict[str, dict]
) -> None:
    rows = []
    selected = (
        "dense_fp16",
        "binary",
        "pool3_binary",
        "int4_per_tensor",
        "int4_per_token",
        "int4_residual",
    )
    for corpus in ("Visual 10k", "FiQA 57k"):
        by_format = {row["format"]: row for row in production[corpus]}
        rows.append(
            [
                corpus,
                "FastPLAID",
                f"{plaid[corpus]['ndcg']:.4f}",
                f"{plaid[corpus]['recall']:.4f}",
                f"{plaid[corpus]['p50']:.2f}",
                f"{plaid[corpus]['bytes'] / 1e9:.3f}",
            ]
        )
        for fmt in selected:
            row = by_format[fmt]
            rows.append(
                [
                    corpus,
                    f"\\sysname{{}} {SHORT_LABELS[fmt]}",
                    f"{row['ndcg']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['p50']:.2f}",
                    f"{row['encoded'] / 1e9:.3f}",
                ]
            )
    tex_table(
        TABLES / "plaid_comparison.tex",
        "llrrrr",
        ["corpus", "system", "\\ndcg{}", "\\recall{}", "p50 ms", "index GB"],
        rows,
    )


def reducer_data() -> dict[int, dict]:
    output = {}
    for docs in (512, 4096):
        path = ARCHIVE / "systems" / "results" / f"reducers-n{docs}.json"
        data = read_json(path)
        if not data.get("gate_passed"):
            raise ValueError(f"{path}: reducer gate failed")
        rows = [
            row
            for row in data["rows"]
            if row["implementation"] == "shared_reducer_policy"
        ]
        output[docs] = {"raw": data, "rows": rows}
    return output


def write_reducer_table(reducers: dict[int, dict]) -> None:
    rows = []
    reducer_order = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")
    for docs in (512, 4096):
        for fmt in ("binary", "int4"):
            entries = reducers[docs]["rows"]
            for reducer in reducer_order:
                full = next(
                    row
                    for row in entries
                    if row["format"] == fmt
                    and row["reducer"] == reducer
                    and row["scope"] == "full"
                )
                cand = next(
                    row
                    for row in entries
                    if row["format"] == fmt
                    and row["reducer"] == reducer
                    and row["scope"] == "candidates"
                )
                error = max(
                    float(full["max_abs_delta_vs_torch"]),
                    float(cand["max_abs_delta_vs_torch"]),
                )
                rows.append(
                    [
                        f"{docs:,}",
                        fmt,
                        reducer.replace("_", "\\_"),
                        f"{full['latency_median_ms']:.3f}",
                        f"{cand['latency_median_ms']:.3f}",
                        f"{full['latency_median_ms'] / cand['latency_median_ms']:.2f}$\\times$",
                        f"{error:.1e}",
                    ]
                )
    tex_table(
        TABLES / "reducers_generalization.tex",
        "rrlrrrr",
        [
            "docs",
            "format",
            "reducer",
            "full ms",
            "cand. ms",
            "speedup",
            "max error",
        ],
        rows,
    )


def scale_data() -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    output: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for modality in ("visual", "fiqa"):
        grouped: dict[int, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for path in sorted(
            (ARCHIVE / "systems" / "scale" / modality).glob(
                "nested-scale-seed*-n*.json"
            )
        ):
            data = read_json(path)
            results = data["results"]
            for row in results:
                fmt = MODE_TO_FORMAT.get(row.get("mode"))
                if row["implementation"] == "dense_fp16_vectorized":
                    fmt = "dense_fp16"
                if not fmt:
                    continue
                grouped[int(row["docs"])][fmt].append(row)

        normalized = {}
        for docs, by_format in grouped.items():
            normalized[docs] = {}
            dense_latency = float(
                np.mean(
                    [
                        row["latency_p50_ms"]
                        for row in by_format["dense_fp16"]
                    ]
                )
            )
            for fmt, rows in by_format.items():
                ndcg = np.array([row["ndcg_at_10"] for row in rows], dtype=float)
                latency = np.array(
                    [row["latency_p50_ms"] for row in rows], dtype=float
                )
                normalized[docs][fmt] = {
                    "ndcg_mean": float(ndcg.mean()),
                    "ndcg_min": float(ndcg.min()),
                    "ndcg_max": float(ndcg.max()),
                    "latency_mean": float(latency.mean()),
                    "speedup": dense_latency / float(latency.mean()),
                }
            if len(by_format["dense_fp16"]) != 3:
                raise ValueError(f"{modality} n={docs}: expected three scale seeds")
        output[modality] = normalized
    return output


def write_scale_table(scale: dict[str, dict]) -> None:
    rows = []
    for modality, label in (("visual", "Visual"), ("fiqa", "FiQA")):
        for docs in sorted(scale[modality]):
            current = scale[modality][docs]
            dense = current["dense_fp16"]
            binary = current["binary"]
            pool = current["pool3_binary"]
            rows.append(
                [
                    label,
                    f"{docs:,}",
                    f"{dense['ndcg_mean']:.4f}",
                    f"{binary['ndcg_mean']:.4f}",
                    f"{binary['speedup']:.2f}$\\times$",
                    f"{pool['ndcg_mean']:.4f}",
                    f"{pool['speedup']:.2f}$\\times$",
                ]
            )
    tex_table(
        TABLES / "scale_summary.tex",
        "lrrrrrr",
        [
            "corpus",
            "docs",
            "dense",
            "binary",
            "B1 speed",
            "pool3",
            "P3 speed",
        ],
        rows,
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(
        FIGURES / f"{name}.pdf",
        metadata={
            "Creator": "MaxSim paper artifact generator",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(README_FIGURES / f"{name}.png", dpi=220)
    plt.close(fig)


def figure_format_layout() -> None:
    values = [
        ("fp32 reference", 512.0, "dense_fp16", 1.0),
        ("fp16 dense", 256.0, "dense_fp16", 2.0),
        ("residual int4", 136.0, "int4_residual", 3.76),
        ("int8 + token scale", 132.0, "int8_per_token", 3.88),
        ("int4 + token scale", 68.0, "int4_per_token", 7.53),
        ("int4 + tensor scale", 64.0, "int4_per_tensor", 8.0),
        ("scaled binary", 16.5, "binary_token_scale_u4", 31.03),
        ("binary", 16.0, "binary", 32.0),
        ("pool3 binary", 16.0 / 3.0, "pool3_binary", 95.9),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(values))
    ax.barh(
        y,
        [value for _, value, _, _ in values],
        color=[COLORS[fmt] for _, _, fmt, _ in values],
        height=0.62,
    )
    ax.set_xscale("log")
    ax.set_yticks(y, [name for name, _, _, _ in values])
    ax.invert_yaxis()
    ax.set_xlabel("bytes per original 128-d document token (log scale)")
    ax.set_xlim(3.5, 720)
    for index, (_, value, _, compression) in enumerate(values):
        byte_text = f"{value:.2f}" if value < 10 else f"{value:g}"
        ax.text(
            value * 1.06,
            index,
            f"{byte_text} B  ({compression:.1f}x)",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="y", visible=False)
    save_figure(fig, "format_layout")


def figure_heatmap(suites: dict[str, dict[str, dict[str, float]]]) -> None:
    names = ("Jina", "GTE", "ColPali", "ColQwen2")
    formats = FORMATS[1:]
    matrix = []
    for name in names:
        means = suite_means(suites[name])
        matrix.append([100.0 * means[fmt] / means["dense_fp16"] for fmt in formats])
    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    image = ax.imshow(
        matrix_np,
        cmap="cividis",
        vmin=65,
        vmax=100,
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(formats)),
        [SHORT_LABELS[fmt] for fmt in formats],
        rotation=24,
        ha="right",
    )
    ax.set_yticks(np.arange(len(names)), names)
    ax.grid(False)
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            value = matrix_np[i, j]
            color = "white" if value < 88 else "#111417"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("macro NDCG retention")
    ax.set_title("The best aggressive format depends on the encoder")
    save_figure(fig, "model_format_heatmap")


def figure_production(production: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    order = [
        "dense_fp16",
        "binary",
        "binary_token_scale_u4",
        "pool3_binary",
        "int4_per_tensor",
        "int4_per_token",
        "int4_residual",
    ]
    for ax, corpus in zip(axes, ("Visual 10k", "FiQA 57k")):
        by_format = {row["format"]: row for row in production[corpus]}
        for fmt in order:
            row = by_format[fmt]
            size = 28 + 62 * math.sqrt(row["encoded"] / max(v["encoded"] for v in by_format.values()))
            ax.scatter(
                row["p50"],
                row["ndcg"],
                s=size,
                color=COLORS[fmt],
                marker=MARKERS[fmt],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            offsets = {
                "Visual 10k": {
                    "dense_fp16": (5, 8),
                    "binary": (5, 5),
                    "binary_token_scale_u4": (5, -11),
                    "pool3_binary": (5, 5),
                    "int4_per_tensor": (5, -12),
                    "int4_per_token": (5, 7),
                    "int4_residual": (5, -11),
                },
                "FiQA 57k": {
                    "dense_fp16": (-7, 8),
                    "binary": (5, 6),
                    "binary_token_scale_u4": (5, -13),
                    "pool3_binary": (5, 6),
                    "int4_per_tensor": (5, 8),
                    "int4_per_token": (5, 8),
                    "int4_residual": (7, -11),
                },
            }
            alignment = (
                "right"
                if corpus == "FiQA 57k" and fmt == "dense_fp16"
                else "left"
            )
            ax.annotate(
                SHORT_LABELS[fmt],
                (row["p50"], row["ndcg"]),
                xytext=offsets[corpus][fmt],
                textcoords="offset points",
                fontsize=7.2,
                ha=alignment,
            )
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_xlabel("single-query p50 latency (ms, log)")
        ax.set_ylabel("NDCG@10")
        ax.set_title(corpus)
        ax.margins(x=0.13, y=0.15)
    fig.text(
        0.5,
        -0.01,
        "Marker area monotonically encodes resident encoded bytes.",
        ha="center",
        fontsize=7.5,
        color="#5F6872",
    )
    fig.tight_layout()
    save_figure(fig, "production_frontier")


def figure_scale(scale: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex="row")
    shown = (
        "dense_fp16",
        "binary",
        "pool3_binary",
        "int4_per_token",
        "int4_residual",
    )
    for row_idx, (modality, label) in enumerate(
        (("visual", "Visual"), ("fiqa", "FiQA"))
    ):
        sizes = np.array(sorted(scale[modality]))
        for fmt in shown:
            means = np.array(
                [scale[modality][size][fmt]["ndcg_mean"] for size in sizes]
            )
            lows = np.array(
                [scale[modality][size][fmt]["ndcg_min"] for size in sizes]
            )
            highs = np.array(
                [scale[modality][size][fmt]["ndcg_max"] for size in sizes]
            )
            speeds = np.array(
                [scale[modality][size][fmt]["speedup"] for size in sizes]
            )
            axes[row_idx, 0].plot(
                sizes,
                means,
                color=COLORS[fmt],
                marker=MARKERS[fmt],
                linewidth=1.4,
                markersize=4,
                label=SHORT_LABELS[fmt],
            )
            axes[row_idx, 0].fill_between(
                sizes, lows, highs, color=COLORS[fmt], alpha=0.10
            )
            axes[row_idx, 1].plot(
                sizes,
                speeds,
                color=COLORS[fmt],
                marker=MARKERS[fmt],
                linewidth=1.4,
                markersize=4,
            )
        axes[row_idx, 0].set_xscale("log")
        axes[row_idx, 1].set_xscale("log")
        axes[row_idx, 0].set_ylabel(f"{label}\nNDCG@10")
        axes[row_idx, 1].set_ylabel(f"{label}\nspeedup")
        axes[row_idx, 1].axhline(1.0, color="#8B9298", linewidth=0.8)
    axes[0, 0].set_title("Quality across scale")
    axes[0, 1].set_title("Latency relative to dense fp16")
    axes[1, 0].set_xlabel("documents")
    axes[1, 1].set_xlabel("documents")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "scale_consistency")


def figure_reducers(reducers: dict[int, dict]) -> None:
    data = reducers[4096]["raw"]["summary"]["candidate_vs_full_speedup"]
    order = ("maxsim", "weighted_maxsim", "topk2", "topk4", "smoothsim")
    labels = ("MaxSim", "Weighted", "TopK2", "TopK4", "SmoothSim")
    binary = [data["binary"][name] for name in order]
    int4 = [data["int4"][name] for name in order]
    errors = [
        row["max_abs_delta_vs_torch"]
        for payload in reducers.values()
        for row in payload["rows"]
    ]

    x = np.arange(len(order))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].bar(x - width / 2, binary, width, color=COLORS["binary"], label="binary")
    axes[0].bar(
        x + width / 2, int4, width, color=COLORS["int4_per_tensor"], label="int4"
    )
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("candidate-only speedup")
    axes[0].set_title("512 candidates from 4,096 documents")

    full_rows = [
        row
        for row in reducers[4096]["rows"]
        if row["scope"] == "full"
    ]
    for fmt, offset in (("binary", -width / 2), ("int4", width / 2)):
        values = [
            next(
                row["latency_median_ms"]
                for row in full_rows
                if row["format"] == fmt and row["reducer"] == reducer
            )
            for reducer in order
        ]
        axes[1].bar(
            x + offset,
            values,
            width,
            color=COLORS["binary" if fmt == "binary" else "int4_per_tensor"],
        )
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("full-scan median (ms)")
    axes[1].set_title(f"Max PyTorch error {max(errors):.2e}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, "reducer_generality")


def validate_manifest_shape() -> None:
    required = [
        ARCHIVE / "ARCHIVE_MANIFEST.sha256",
        ARCHIVE / "models" / "result-checksums.sha256",
        ARCHIVE / "systems" / "result-checksums.sha256",
        ARCHIVE / "systems" / "results" / "production-fiqa57k.json",
        ARCHIVE / "systems" / "results" / "production-visual10k-2kq.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing archive inputs:\n" + "\n".join(missing))
    statuses = (
        ARCHIVE / "models" / "case-status.tsv",
        ARCHIVE / "systems" / "case-status.tsv",
    )
    for path in statuses:
        text = path.read_text().lower()
        if "\tfailed" in text or "\terror" in text:
            raise ValueError(f"{path}: failed archive cases")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
    validate_manifest_shape()
    suites = quality_suites()
    production = production_data()
    plaid = plaid_data()
    reducers = reducer_data()
    scale = scale_data()

    write_model_matrix(suites)
    write_per_dataset_tables(suites)
    write_production_table(production)
    write_plaid_table(production, plaid)
    write_reducer_table(reducers)
    write_scale_table(scale)

    figure_format_layout()
    figure_heatmap(suites)
    figure_production(production)
    figure_scale(scale)
    figure_reducers(reducers)

    print("generated 7 tables and 5 figures from the archived result bundle")


if __name__ == "__main__":
    main()
