"""Build text-ColBERT embedding caches for BEIR datasets in the harness schema.

Embeds a BEIR corpus + test queries with a PyLate-compatible ColBERT model and
writes the .npz consumed by benchmarks/run_retrieval.py (doc_embeddings /
doc_offsets / query_embeddings / query_offsets / qrels / ids).

Example:
    python -m benchmarks.build_beir_colbert_embeddings \
        --dataset scifact --model lightonai/GTE-ModernColBERT-v1 \
        --output caches/beir-scifact-gte-moderncolbert.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa")


def _load_beir(dataset: str):
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{dataset}-qrels", split="test")
    return corpus, queries, qrels


def _doc_text(row) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def _flatten(embeddings) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(e, dtype=np.float32) for e in embeddings]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([a.shape[0] for a in arrays], out=offsets[1:])
    return np.concatenate(arrays, axis=0), offsets


def _batch_fidelity_check(model, texts, batch_size: int, is_query: bool) -> float:
    sample = texts[: min(8, len(texts))]
    solo = model.encode(sample, is_query=is_query, batch_size=1, show_progress_bar=False)
    batched = model.encode(sample, is_query=is_query, batch_size=batch_size, show_progress_bar=False)
    worst = 1.0
    for a, b in zip(solo, batched):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        rows = min(a.shape[0], b.shape[0])
        cos = np.sum(a[:rows] * b[:rows], axis=1) / (
            np.linalg.norm(a[:rows], axis=1) * np.linalg.norm(b[:rows], axis=1) + 1e-9
        )
        worst = min(worst, float(cos.min()))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=BEIR_DATASETS)
    parser.add_argument("--model", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-docs", type=int, default=0, help="0 = full corpus")
    args = parser.parse_args()

    from pylate import models

    corpus, queries, qrels = _load_beir(args.dataset)

    relevant = {}
    for row in qrels:
        if float(row["score"]) > 0:
            relevant.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    doc_ids = [str(r["_id"]) for r in corpus]
    if args.limit_docs:
        # keep every relevant doc, fill the remainder deterministically
        needed = set().union(*relevant.values()) if relevant else set()
        keep = [d for d in doc_ids if d in needed]
        keep += [d for d in doc_ids if d not in needed][: max(0, args.limit_docs - len(keep))]
        keep_set = set(keep)
        corpus = corpus.filter(lambda r: str(r["_id"]) in keep_set)
        doc_ids = [str(r["_id"]) for r in corpus]
    doc_pos = {d: i for i, d in enumerate(doc_ids)}

    query_rows = [r for r in queries if str(r["_id"]) in relevant]
    query_ids = [str(r["_id"]) for r in query_rows]
    print(f"{args.dataset}: {len(doc_ids)} docs, {len(query_ids)} test queries")

    model = models.ColBERT(model_name_or_path=args.model, device=args.device)

    doc_texts = [_doc_text(r) for r in corpus]
    worst = _batch_fidelity_check(model, doc_texts, args.batch_size, is_query=False)
    print(f"batch-fidelity worst token cosine (docs, batch {args.batch_size} vs 1): {worst:.6f}")
    if worst < 0.999:
        raise SystemExit(f"builder is not batch-faithful at batch {args.batch_size}; use --batch-size 1")

    doc_embs = model.encode(doc_texts, is_query=False, batch_size=args.batch_size, show_progress_bar=True)
    query_embs = model.encode(
        [r["text"] for r in query_rows], is_query=True, batch_size=args.batch_size, show_progress_bar=True
    )

    doc_flat, doc_offsets = _flatten(doc_embs)
    query_flat, query_offsets = _flatten(query_embs)

    qrels_matrix = np.zeros((len(query_ids), len(doc_ids)), dtype=np.float32)
    for qi, qid in enumerate(query_ids):
        for did in relevant[qid]:
            if did in doc_pos:
                qrels_matrix[qi, doc_pos[did]] = 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset_name=np.array(f"beir-{args.dataset}-{Path(args.model).name}"),
        doc_embeddings=doc_flat,
        doc_offsets=doc_offsets,
        query_embeddings=query_flat,
        query_offsets=query_offsets,
        qrels=qrels_matrix,
        query_ids=np.asarray(query_ids),
        doc_ids=np.asarray(doc_ids),
    )
    print(f"wrote {args.output} (docs {doc_flat.shape}, queries {query_flat.shape}, dim {doc_flat.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
