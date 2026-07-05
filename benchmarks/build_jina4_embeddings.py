"""Multi-vector caches from jina-embeddings-v4 (unified text+image space).

Two modes:
  --vidore <dataset>   embed a ViDoRe image dataset (text query -> page image)
  --beir <dataset>     embed a BEIR text dataset (text query -> text doc)

jina-v4 emits late-interaction token embeddings when output is multivector;
this gives a second model family for BOTH modalities in one embedding space.

    python -m benchmarks.build_jina4_embeddings --vidore docvqa_test_subsampled --limit 500 --output caches/jina4-docvqa500.npz
    python -m benchmarks.build_jina4_embeddings --beir scifact --output caches/jina4-scifact.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _flatten(embeddings):
    arrays = [np.asarray(e, dtype=np.float32) for e in embeddings]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([a.shape[0] for a in arrays], out=offsets[1:])
    return np.concatenate(arrays, axis=0), offsets


def _load_model():
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4", trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vidore")
    parser.add_argument("--beir")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = _load_model()

    def enc_texts(texts, prompt_name):
        out = []
        for i in range(0, len(texts), 8):
            embs = model.encode_text(
                texts=texts[i : i + 8], task="retrieval", prompt_name=prompt_name,
                return_multivector=True,
            )
            out.extend(np.asarray(e.to(np.float32) if hasattr(e, "to") else e, dtype=np.float32) for e in
                       (x.float().cpu().numpy() if hasattr(x, "cpu") else x for x in embs))
            if (i // 8) % 25 == 0:
                print(f"  text {i}/{len(texts)}", flush=True)
        return out

    def enc_images(images):
        out = []
        for i, img in enumerate(images):
            embs = model.encode_image(images=[img], task="retrieval", return_multivector=True)
            x = embs[0]
            out.append(x.float().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, dtype=np.float32))
            if (i + 1) % 50 == 0:
                print(f"  image {i + 1}/{len(images)}", flush=True)
        return out

    if args.vidore:
        from datasets import load_dataset

        ds = load_dataset(f"vidore/{args.vidore}", split="test")
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        images = [r["image"] for r in ds]
        queries = [r["query"] for r in ds]
        doc_ids = [f"doc-{i}" for i in range(len(images))]
        query_ids = [f"q-{i}" for i in range(len(queries))]
        doc_embs = enc_images(images)
        q_embs = enc_texts(queries, "query")
        qrels = np.eye(len(queries), len(images), dtype=np.float32)
        name = f"jina4-vidore-{args.vidore}"
    else:
        from benchmarks.build_beir_colbert_embeddings import _load_beir, _doc_text

        corpus, queries_ds, qrels_ds = _load_beir(args.beir)
        relevant = {}
        for row in qrels_ds:
            if float(row["score"]) > 0:
                relevant.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
        doc_ids = [str(r["_id"]) for r in corpus]
        doc_pos = {d: i for i, d in enumerate(doc_ids)}
        query_rows = [r for r in queries_ds if str(r["_id"]) in relevant]
        query_ids = [str(r["_id"]) for r in query_rows]
        doc_embs = enc_texts([_doc_text(r) for r in corpus], "passage")
        q_embs = enc_texts([r["text"] for r in query_rows], "query")
        qrels = np.zeros((len(query_ids), len(doc_ids)), dtype=np.float32)
        for qi, qid in enumerate(query_ids):
            for did in relevant[qid]:
                if did in doc_pos:
                    qrels[qi, doc_pos[did]] = 1.0
        name = f"jina4-beir-{args.beir}"

    doc_flat, doc_offsets = _flatten(doc_embs)
    q_flat, q_offsets = _flatten(q_embs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset_name=np.array(name),
        doc_embeddings=doc_flat,
        doc_offsets=doc_offsets,
        query_embeddings=q_flat,
        query_offsets=q_offsets,
        qrels=qrels,
        query_ids=np.asarray(query_ids),
        doc_ids=np.asarray(doc_ids),
    )
    print(f"wrote {args.output} (docs {doc_flat.shape}, queries {q_flat.shape}, dim {doc_flat.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
