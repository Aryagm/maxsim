"""Generate the paper's exact-value LaTeX tables from the committed ledger.

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
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "text_beir.tex").write_text("\n".join(lines) + "\n")
    print("wrote text_beir.tex")


if __name__ == "__main__":
    table_per_dataset("paper-vidore-*colqwen2*-r3.json", "per_dataset_colqwen2.tex", " colqwen2")
    table_per_dataset("paper-vidore-*colpali*-r3.json", "per_dataset_colpali.tex", " colpali")
    table_significance()
    table_kernels()
    table_tenk()
    table_text_beir()
