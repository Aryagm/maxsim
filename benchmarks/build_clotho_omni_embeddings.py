"""Text->audio retrieval cache: Clotho eval clips embedded with ColQwen-Omni.

Clotho evaluation split: 1,045 audio clips, 5 human captions each. Docs are
the audio clips (multi-vector audio-token embeddings from
vidore/colqwen-omni-v0.1), queries are the captions, and the qrel for a
caption is exactly its source clip. Output is the harness .npz schema.

    python -m benchmarks.build_clotho_omni_embeddings --output caches/clotho-eval-colqwen-omni.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CLOTHO_EVAL_URL = "https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z?download=1"
CLOTHO_CAPTIONS_URL = "https://zenodo.org/records/4783391/files/clotho_captions_evaluation.csv?download=1"


def _flatten(embeddings):
    arrays = [np.asarray(e, dtype=np.float32) for e in embeddings]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([a.shape[0] for a in arrays], out=offsets[1:])
    return np.concatenate(arrays, axis=0), offsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("clotho-data"))
    parser.add_argument("--model", default="vidore/colqwen-omni-v0.1")
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument("--captions-per-clip", type=int, default=1)
    args = parser.parse_args()

    import csv
    import subprocess

    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "clotho_audio_evaluation.7z"
    captions_csv = args.data_dir / "clotho_captions_evaluation.csv"
    audio_dir = args.data_dir / "evaluation"
    if not captions_csv.exists():
        subprocess.run(["curl", "-sL", "-o", str(captions_csv), CLOTHO_CAPTIONS_URL], check=True)
    if not audio_dir.exists():
        if not archive.exists():
            subprocess.run(["curl", "-sL", "-o", str(archive), CLOTHO_EVAL_URL], check=True)
        subprocess.run(["7z", "x", str(archive), f"-o{args.data_dir}", "-y"], check=True)

    rows = list(csv.DictReader(open(captions_csv)))
    if args.limit_clips:
        rows = rows[: args.limit_clips]
    print(f"clotho eval: {len(rows)} clips", flush=True)

    import torch
    from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor

    model = ColQwen2_5Omni.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = ColQwen2_5OmniProcessor.from_pretrained(args.model)

    import librosa

    doc_ids, doc_embs = [], []
    for i, row in enumerate(rows):
        path = audio_dir / row["file_name"]
        audio, _sr = librosa.load(str(path), sr=16000, mono=True)
        batch = processor.process_audios([audio]).to(model.device)
        with torch.no_grad():
            emb = model(**batch)
        doc_embs.append(emb[0].to(torch.float32).cpu().numpy())
        doc_ids.append(row["file_name"])
        if (i + 1) % 50 == 0:
            print(f"  audio {i + 1}/{len(rows)}", flush=True)

    queries, query_ids, rel = [], [], []
    for ci, row in enumerate(rows):
        for k in range(1, args.captions_per_clip + 1):
            queries.append(row[f"caption_{k}"])
            query_ids.append(f"{row['file_name']}#c{k}")
            rel.append(ci)
    q_embs = []
    for i in range(0, len(queries), 8):
        batch = processor.process_queries(queries[i : i + 8]).to(model.device)
        with torch.no_grad():
            emb = model(**batch)
        q_embs.extend(e.to(torch.float32).cpu().numpy() for e in emb)
        if (i // 8 + 1) % 25 == 0:
            print(f"  query batch {i // 8 + 1}", flush=True)

    doc_flat, doc_offsets = _flatten(doc_embs)
    q_flat, q_offsets = _flatten(q_embs)
    qrels = np.zeros((len(queries), len(doc_ids)), dtype=np.float32)
    for qi, ci in enumerate(rel):
        qrels[qi, ci] = 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset_name=np.array("clotho-eval-colqwen-omni"),
        doc_embeddings=doc_flat,
        doc_offsets=doc_offsets,
        query_embeddings=q_flat,
        query_offsets=q_offsets,
        qrels=qrels,
        query_ids=np.asarray(query_ids),
        doc_ids=np.asarray(doc_ids),
    )
    print(f"wrote {args.output} (docs {doc_flat.shape}, queries {q_flat.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
