# Retrieval Accuracy And Size Design

## Goal

Improve bitmax's retrieval quality while preserving the storage and GPU-latency advantages of int8-query/binary-document late interaction.

## Scope

This pass focuses on real retrieval measurements, not another synthetic-only kernel loop. The primary benchmark is a ViDoRe/ColQwen-style embedding slice on the persistent project-owned VAST RTX 4090 worker. The work remains a kernel-library effort: no index, database, service, or search engine is added.

## Success Metrics

- Report every variant with latency, bytes/doc, recall@1, recall@10, MRR@10, and NDCG@10 on the same retrieval slice.
- Keep the raw binary default unless a variant improves NDCG/recall with acceptable storage and latency cost.
- Prefer same-size variants first: calibrated binary thresholds keep one bit per dimension and can use the existing binary CUDA scorer after offline packing.
- Promote larger variants only if they show a clear quality gain: per-token scale is the first larger candidate because it keeps roughly 25-28x fp32 compression on blog-shaped documents.

## Candidate Variants

1. **Raw binary docs**
   Current default. Storage is one bit per dimension. It is the latency and size baseline.

2. **Per-document scale**
   Already implemented on CUDA. Adds one scale per document and applies it before top-k. It may recover document-level magnitude with tiny storage overhead.

3. **Calibrated binary thresholds**
   Offline thresholds change how document signs are packed, but the stored documents remain one bit per dimension. This is the best first accuracy target because runtime CUDA scoring stays unchanged.

4. **Per-token scale**
   Stores one magnitude scale per document token and scores `scale[token] * dot(query, sign(token))` before MaxSim. This costs more storage than binary thresholds but can recover token-level magnitude.

5. **Ternary docs**
   Stores `{-1, 0, +1}` using two bits per dimension. This is deferred until same-size binary thresholds and scale variants are measured, because it halves the storage advantage versus binary docs.

## Data Flow

1. Build a ViDoRe embedding `.npz` on the VAST worker with `benchmarks/build_vidore_embeddings.py`.
2. Run dense fp16, raw binary, doc-scale binary, thresholded binary, and reference scale variants through `benchmarks/run_retrieval.py`.
3. If a reference variant wins, add the smallest GPU kernel or packing change needed to measure it on CUDA.
4. Write benchmark JSON artifacts under `benchmark-results/` and summarize only measured rows in docs.

## Guardrails

- Use the existing project-owned VAST worker `43248165` while it remains reachable.
- Do not destroy any VAST instance except IDs in `.vast/bitmax-instances.jsonl` whose live label starts with `bitmax-v0-`.
- Add tests before promoting any new packing/scoring API.
- Do not change the default format unless a measured Pareto win is clear.

## Initial Decision

Start with real retrieval measurement and same-size calibrated binary thresholding. Only after that, implement per-token scale on GPU if the retrieval artifact shows enough quality upside to justify the storage cost.
