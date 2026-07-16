"""Build text-ColBERT embedding caches for BEIR datasets in the harness schema.

Embeds a BEIR corpus + test queries with a PyLate-compatible ColBERT model and
writes the .npz consumed by benchmarks/run_retrieval.py (doc_embeddings /
doc_offsets / query_embeddings / query_offsets / qrels / ids).

Example:
    python -m benchmarks.build_beir_colbert_embeddings \
        --dataset scifact --model lightonai/GTE-ModernColBERT-v1 \
        --output caches/beir-scifact-gte-moderncolbert.npz

    python -m benchmarks.build_beir_colbert_embeddings \
        --dataset scifact --model jinaai/jina-colbert-v2 \
        --output caches/beir-scifact-jina-colbert-v2.npz
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa")
JINA_COLBERT_V2 = "jinaai/jina-colbert-v2"
CANONICAL_COLBERT_V2 = "colbert-ir/colbertv2.0"


def _model_profile(model_name: str) -> tuple[str, dict[str, Any]]:
    """Return the PyLate options required by a known checkpoint family."""
    normalized = model_name.rstrip("/").lower()
    if normalized == JINA_COLBERT_V2.lower():
        return (
            "jina-colbert-v2",
            {
                "query_prefix": "[QueryMarker]",
                "document_prefix": "[DocumentMarker]",
                "attend_to_expansion_tokens": True,
                "trust_remote_code": True,
            },
        )
    if normalized == CANONICAL_COLBERT_V2.lower():
        # PyLate reads the canonical markers and expansion behavior from the
        # Stanford ColBERT checkpoint metadata.
        return "canonical-colbert-v2", {}
    return "generic-pylate-colbert", {}


def _load_colbert_model(models, model_name: str, *, device: str | None, fallback_model: str | None):
    attempts = [model_name]
    if fallback_model and fallback_model != model_name:
        attempts.append(fallback_model)

    failures: list[tuple[str, Exception]] = []
    for candidate in attempts:
        profile, options = _model_profile(candidate)
        try:
            model = models.ColBERT(model_name_or_path=candidate, device=device, **options)
            return model, candidate, profile, options, failures
        except Exception as exc:  # model loading can fail in transformers or remote code
            failures.append((candidate, exc))
            if candidate != attempts[-1]:
                print(
                    f"failed to load {candidate!r} ({type(exc).__name__}: {exc}); "
                    f"falling back to {attempts[-1]!r}",
                    flush=True,
                )
    candidate, exc = failures[-1]
    raise RuntimeError(f"failed to load ColBERT model {candidate!r}") from exc


def _atomic_savez_compressed(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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
    parser.add_argument(
        "--fallback-model",
        default=None,
        help=(
            "Checkpoint to try if the requested model cannot load. "
            f"{JINA_COLBERT_V2} defaults to {CANONICAL_COLBERT_V2}."
        ),
    )
    parser.add_argument(
        "--no-model-fallback",
        action="store_true",
        help="Fail instead of falling back when the requested model cannot load.",
    )
    parser.add_argument(
        "--graded-qrels",
        action="store_true",
        help="Preserve BEIR relevance grades instead of the historical positive/binary cache convention.",
    )
    args = parser.parse_args()

    from pylate import models

    corpus, queries, qrels = _load_beir(args.dataset)

    relevant: dict[str, dict[str, float]] = {}
    for row in qrels:
        score = float(row["score"])
        if score > 0:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            query_relevance = relevant.setdefault(qid, {})
            query_relevance[did] = max(score, query_relevance.get(did, 0.0))

    doc_ids = [str(r["_id"]) for r in corpus]
    if args.limit_docs:
        # keep every relevant doc, fill the remainder deterministically
        needed = set().union(*(set(values) for values in relevant.values())) if relevant else set()
        keep = [d for d in doc_ids if d in needed]
        keep += [d for d in doc_ids if d not in needed][: max(0, args.limit_docs - len(keep))]
        keep_set = set(keep)
        corpus = corpus.filter(lambda r: str(r["_id"]) in keep_set)
        doc_ids = [str(r["_id"]) for r in corpus]
    doc_pos = {d: i for i, d in enumerate(doc_ids)}

    query_rows = [r for r in queries if str(r["_id"]) in relevant]
    query_ids = [str(r["_id"]) for r in query_rows]
    print(f"{args.dataset}: {len(doc_ids)} docs, {len(query_ids)} test queries")

    fallback_model = args.fallback_model
    if fallback_model is None and args.model.rstrip("/").lower() == JINA_COLBERT_V2.lower():
        fallback_model = CANONICAL_COLBERT_V2
    if args.no_model_fallback:
        fallback_model = None
    model, resolved_model, model_profile, model_options, failures = _load_colbert_model(
        models,
        args.model,
        device=args.device,
        fallback_model=fallback_model,
    )
    print(f"model: {resolved_model} ({model_profile})", flush=True)

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
        for did, score in relevant[qid].items():
            if did in doc_pos:
                qrels_matrix[qi, doc_pos[did]] = np.float32(score if args.graded_qrels else 1.0)

    _atomic_savez_compressed(
        args.output,
        builder_schema_version=np.array(2, dtype=np.int64),
        dataset_name=np.array(f"beir-{args.dataset}-{Path(resolved_model).name}"),
        dataset_id=np.array(f"BeIR/{args.dataset}"),
        model_requested=np.array(args.model),
        model_resolved=np.array(resolved_model),
        model_profile=np.array(model_profile),
        model_options_json=np.array(json.dumps(model_options, sort_keys=True)),
        model_fallback_used=np.array(resolved_model != args.model),
        qrels_semantics=np.array("graded_original" if args.graded_qrels else "binary_positive"),
        model_load_failures_json=np.array(
            json.dumps(
                [{"model": name, "error_type": type(exc).__name__, "error": str(exc)} for name, exc in failures],
                sort_keys=True,
            )
        ),
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
