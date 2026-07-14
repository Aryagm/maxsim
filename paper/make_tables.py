"""Generate the paper's exact-value LaTeX tables from the artifact ledger.

Writes paper/tables/*.tex; main.tex \\input{}s them. Run from the repo root:
    python paper/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

RAW = Path("docs/benchmark_results/raw")
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)

TIERS = [
    ("dense_fp16_baseline", "dense fp16"),
    ("int4_int8q_dp4a", "int4$+$dp4a"),
    ("bitmax_binary", "binary"),
    ("binary_token_scale_fp16_cuda", "fp16 scales"),
    ("binary_token_scale_u4_cuda", "u4 scales"),
    ("pool2_binary", "pool2"),
    ("pool3_binary", "pool3"),
]

PRETTY_DS = {
    "docvqa": "DocVQA", "infovqa": "InfoVQA", "arxivqa": "ArxivQA",
    "tabfquad": "TabFQuAD", "tatdqa": "TAT-DQA",
    "syntheticdocqa-ai": "SynthDocQA/AI", "syntheticdocqa-energy": "SynthDocQA/Energy",
    "syntheticdocqa-government": "SynthDocQA/Gov.", "syntheticdocqa-healthcare": "SynthDocQA/Health",
    "syntheticdocqa-shift": "SynthDocQA/Shift",
}


def _dataset_rows(pattern: str):
    out = []
    for path in sorted(RAW.glob(pattern)):
        data = json.loads(path.read_text())
        rows = {r["implementation"]: r for r in data["results"]}
        p3 = path.with_name(path.name.replace("paper-", "paper-pool3-", 1))
        if p3.exists():
            rows.update({r["implementation"]: r for r in json.loads(p3.read_text())["results"]
                         if r["implementation"] != "dense_fp16_baseline"})
        stem = path.stem.replace("paper-vidore-", "")
        key = next((k for k in PRETTY_DS if stem.startswith(k)), stem)
        meta = data["results"][0]
        out.append((PRETTY_DS.get(key, key), rows, int(meta["docs"]), int(meta["query_count"])))
    return out


def table_per_dataset(pattern: str, filename: str, caption_note: str):
    datasets = _dataset_rows(pattern)
    lines = [
        "\\begin{tabular}{l r r " + "r " * len(TIERS) + "}",
        "\\toprule",
        "dataset & docs & queries & " + " & ".join(label for _, label in TIERS) + " \\\\",
        "\\midrule",
    ]
    sums = [0.0] * len(TIERS)
    counts = [0] * len(TIERS)
    for name, rows, docs, queries in datasets:
        cells = [name, f"{docs:,}", f"{queries:,}"]
        for i, (impl, _) in enumerate(TIERS):
            row = rows.get(impl)
            if row is None:
                cells.append("---")
            else:
                v = row["ndcg_at_10"]
                cells.append(f"{v:.4f}")
                sums[i] += v
                counts[i] += 1
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    means = ["\\textbf{mean}", "", ""] + [
        f"\\textbf{{{sums[i]/counts[i]:.4f}}}" if counts[i] else "---" for i in range(len(TIERS))
    ]
    lines.append(" & ".join(means) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / filename).write_text("\n".join(lines) + "\n")
    print("wrote", filename, f"({len(datasets)} datasets){caption_note}")


def table_significance():
    pairs = json.loads((RAW / "significance-suite.json").read_text())["pairs"]
    pretty = {
        "int4_int8q_dp4a": "int4$+$dp4a", "dense_fp16_baseline": "dense",
        "bitmax_binary": "binary", "pool2_binary": "pool2", "pool3_binary": "pool3",
        "binary_token_scale_fp16_cuda": "fp16 scales", "binary_token_scale_u4_cuda": "u4 scales",
    }
    lines = [
        "\\begin{tabular}{l r r r r}",
        "\\toprule",
        "comparison & $\\Delta$ \\ndcg{} & 95\\% CI & win / loss / tie & sign-test $p$ \\\\",
        "\\midrule",
    ]
    for p in pairs:
        pv = p["sign_test_p"]
        pv_txt = f"{pv:.2g}" if pv >= 1e-4 else f"$<10^{{{int(f'{pv:.0e}'.split('e')[1])}}}$"
        lines.append(
            f"{pretty[p['a']]} vs.\\ {pretty[p['b']]} & ${p['mean_delta']:+.4f}$ & "
            f"$[{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]$ & "
            f"{p['wins']} / {p['losses']} / {p['ties']:,} & {pv_txt} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "significance.tex").write_text("\n".join(lines) + "\n")
    print("wrote significance.tex", f"({len(pairs)} pairs)")


def table_kernels():
    gate = json.loads((RAW / "dim128-gate-sweep.json").read_text())["results"]
    lines = [
        "\\begin{tabular}{l r r r r r}",
        "\\toprule",
        "shape & docs & tokens/doc & generic (ms) & unrolled (ms) & speedup \\\\",
        "\\midrule",
    ]
    for r in gate:
        lines.append(
            f"{r['name'].replace('_', '\\_')} & {r['docs']:,} & {r['tokens_per_doc']} & "
            f"{r['generic_median_ms']:.2f} & {r['unrolled_median_ms']:.2f} & "
            f"{r['unrolled_speedup']:.2f}$\\times$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "kernels.tex").write_text("\n".join(lines) + "\n")
    print("wrote kernels.tex", f"({len(gate)} shapes)")


def table_production_cuda():
    data = json.loads(
        (RAW / "cuda-step1-rtx4090-20260714.json").read_text()
    )
    if data.get("schema_version") != 2:
        raise ValueError("production CUDA artifact must use schema version 2")
    if not data.get("gate_passed") or len(data["cases"]) != 1:
        raise ValueError("production CUDA artifact must contain one passing case")
    metadata = data.get("metadata", {})
    required_metadata = (
        "run_utc",
        "git_commit",
        "source_archive_sha256",
        "source_archive_path",
        "vast_instance_id",
        "container_image",
        "build_command",
        "benchmark_command",
        "nvcc_version",
        "residual_reducer_warps",
        "residual_reducer_force_warps",
    )
    missing_metadata = [name for name in required_metadata if not metadata.get(name)]
    if missing_metadata:
        raise ValueError(
            "production CUDA artifact is missing metadata: "
            + ", ".join(missing_metadata)
        )
    expected_config = {
        "doc_counts": [4096],
        "candidate_counts": [32, 128, 512, 2048, 4096],
        "min_doc_tokens": 64,
        "max_doc_tokens": 192,
        "batch": 4,
        "query_tokens": 32,
        "k": 10,
        "warmup": 5,
        "repeat": 20,
        "runs": 3,
        "seed": 20260713,
        "dim": 128,
        "correctness_only": False,
        "performance_gates_enabled": True,
    }
    if data.get("config") != expected_config:
        raise ValueError("production CUDA artifact does not match the release matrix")
    if (
        metadata.get("gpu") != "NVIDIA GeForce RTX 4090"
        or metadata.get("compute_capability") != [8, 9]
    ):
        raise ValueError("production CUDA artifact must come from an RTX 4090/SM89")
    if (
        metadata.get("residual_reducer_warps") != 8
        or metadata.get("residual_reducer_force_warps") != -1
    ):
        raise ValueError("production CUDA artifact must use adaptive residual routing")

    timings = data["cases"][0]["timings"]
    rows = {
        (row["operation"], row.get("candidate_count")): row
        for row in timings
    }
    full_rows = [row for row in timings if row.get("scope") == "full"]
    if not full_rows or any(
        row.get("max_abs_error_vs_reference_scores") is None
        for row in full_rows
    ):
        raise ValueError("every full-scan row must include measured parity")
    cascade_rows = [
        row for row in timings if row["operation"] == "int4_residual_cascade"
    ]
    if not cascade_rows or any(
        row.get("max_abs_error_vs_full_scores") is None for row in cascade_rows
    ):
        raise ValueError("every cascade row must include full-score parity")
    full_cascade = next(
        (row for row in cascade_rows if row.get("candidate_count") == 4096),
        None,
    )
    if full_cascade is None or full_cascade.get("indices_exact_vs_stable_full") is not True:
        raise ValueError("full-budget cascade must exactly match stable full top-k")
    spec = [
        ("int4_per_token_full_fp32", None, "Per-token int4 full scan (fp32 query)"),
        ("int4_per_token_topk_int8", None, "Per-token int4 top-$k$ (int8 query)"),
        ("int4_residual_full_fp32", None, "Residual int4 full scan"),
        ("int4_residual_candidates_fp32", 512, "Residual int4, 512 candidates"),
        ("int4_residual_cascade", 512, "Prefix scan $+$ 512-candidate cascade"),
    ]
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Production operation & P50 (ms) & P95 (ms) \\\\",
        "\\midrule",
    ]
    for operation, candidate_count, label in spec:
        row = rows[(operation, candidate_count)]
        lines.append(
            f"{label} & {row['latency_p50_ms']:.2f} & "
            f"{row['latency_p95_ms']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "production_cuda.tex").write_text("\n".join(lines) + "\n")
    print("wrote production_cuda.tex")


def table_tenk():
    data = json.loads((RAW / "paper-unique-10k-final.json").read_text())
    rows = {r["implementation"]: r for r in data["results"] if not r.get("status") or r["status"] == "ok"}
    dense_v = rows["dense_fp16_vectorized"]["latency_ms"]
    spec = [
        ("dense_fp16_baseline", "dense fp16 (loop impl.)", "2$\\times$", False),
        ("dense_fp16_vectorized", "dense fp16 (vectorized)", "2$\\times$", True),
        ("fast_plaid", "fast-plaid (tuned)$^{\\ast}$", "3.4$\\times$", None),
        ("faiss_gpu_mean_pool_flat_ip", "FAISS GPU mean-pool", "746$\\times$", False),
        ("bitmax_int4_dp4a", "\\sysname{} int4$+$dp4a", "8$\\times$", True),
        ("bitmax_binary", "\\sysname{} binary", "32$\\times$", True),
        ("bitmax_binary_token_scale", "\\sysname{} fp16 token scales", "28.4$\\times$", True),
        ("bitmax_binary_token_scale_u4", "\\sysname{} u4 token scales", "31$\\times$", True),
        ("bitmax_pooled_binary", "\\sysname{} pool2 binary", "63.9$\\times$", True),
        ("bitmax_pooled_binary3", "\\sysname{} pool3 binary", "95.9$\\times$", True),
    ]
    lines = [
        "\\begin{tabular}{l r r r r r r}",
        "\\toprule",
        "implementation & \\ndcg{} & R@10 & MRR@10 & latency & speedup & compr. \\\\",
        "\\midrule",
    ]
    for impl, label, comp, spd in spec:
        r = rows.get(impl)
        if r is None:
            continue
        lat = r["latency_ms"] / 1000
        lat_txt = "n/c" if spd is None else (f"{lat:.2f}\\,s" if lat < 100 else f"{lat:.0f}\\,s")
        spd_txt = {None: "n/c", False: "---"}.get(spd, f"{dense_v / r['latency_ms']:.1f}$\\times$")
        lines.append(
            f"{label} & {r['ndcg_at_10']:.4f} & {r['recall_at_10']:.4f} & "
            f"{r.get('mrr_at_10', 0):.4f} & {lat_txt} & {spd_txt} & {comp} \\\\"
        )
        if impl == "faiss_gpu_mean_pool_flat_ip":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tenk.tex").write_text("\n".join(lines) + "\n")
    print("wrote tenk.tex")


def table_text_beir():
    tiers = [
        ("dense_fp16_baseline", "dense fp16"),
        ("int4_int8q_dp4a", "int4$+$dp4a"),
        ("bitmax_binary", "binary"),
        ("binary_token_scale_fp16_cuda", "fp16 scales"),
        ("binary_token_scale_u4_cuda", "u4 scales"),
        ("pool2_binary", "pool2"),
        ("pool3_binary", "pool3"),
    ]
    pretty = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA-2018"}
    lines = [
        "\\begin{tabular}{l r r " + "r " * len(tiers) + "}",
        "\\toprule",
        "dataset & docs & queries & " + " & ".join(label for _, label in tiers) + " \\\\",
        "\\midrule",
    ]
    for path in sorted(RAW.glob("text-beir-*-r3.json")):
        data = json.loads(path.read_text())
        rows = {r["implementation"]: r for r in data["results"]}
        ds = path.stem.replace("text-beir-", "").split("-gte-")[0]
        cells = [pretty.get(ds, ds), f"{int(data['dataset']['docs']):,}", f"{int(data['dataset']['queries']):,}"]
        for impl, _ in tiers:
            row = rows.get(impl)
            cells.append(f"{row['ndcg_at_10']:.4f}" if row else "---")
        lines.append(" & ".join(cells) + " \\\\")
    # fiqa via the exact CPU-simulation evaluator (harness loop-dense is
    # impractical at 57k docs); scored identically, single evaluation
    fiqa = RAW / "exp-eval-beir-fiqa-gte-moderncolbert.json"
    if fiqa.exists():
        vals = json.loads(fiqa.read_text())["ndcg_at_10"]
        remap = {
            "dense_fp16_baseline": "dense_fp32", "int4_int8q_dp4a": "int4_sim",
            "bitmax_binary": "binary", "pool2_binary": "pool2_binary", "pool3_binary": "pool3_binary",
        }
        cells = ["FiQA-2018$^{\\ast}$", "57,638", "648"]
        for impl, _ in tiers:
            key = remap.get(impl)
            cells.append(f"{vals[key]:.4f}" if key and key in vals else "---")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "text_beir.tex").write_text("\n".join(lines) + "\n")
    print("wrote text_beir.tex")


if __name__ == "__main__":
    table_per_dataset("paper-vidore-*colqwen2*-r3.json", "per_dataset_colqwen2.tex", " colqwen2")
    table_per_dataset("paper-vidore-*colpali*-r3.json", "per_dataset_colpali.tex", " colpali")
    table_significance()
    table_kernels()
    table_production_cuda()
    table_tenk()
    table_text_beir()
