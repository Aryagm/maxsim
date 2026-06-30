# Binary-First Accuracy And Usability Design

## Goal

Make `bitmax` more usable and more accurate while preserving the main value
proposition: one-bit document storage for roughly 32x fp32 corpus-side
compression. Add int4 as a separate experimental backend for users who prefer
accuracy over maximum compression.

## Priorities

1. Keep raw binary signs as the stable default.
2. Promote zero-threshold per-dimension centroid calibration as the primary
   experimental 32x accuracy path.
3. Optimize centroid top-k for CUDA by moving query weighting onto the resident
   GPU path.
4. Add persistence helpers so packed corpora and centroid calibration can be
   saved and loaded without rebuilding.
5. Run larger VAST retrieval slices before making broader claims.
6. Add int4 as an experimental selectable backend, not as a replacement for
   binary.

## Architecture

### Persistence

Add top-level `bitmax.save_packed(path, packed, calibration=None, metadata=None)`
and `bitmax.load_packed(path)`. Loading returns a small bundle object with
`packed`, optional `centroid_calibration`, and user metadata. The file format is
NumPy `.npz` with a schema version and explicit arrays for packed bytes,
offsets, scale, and calibration vectors.

Persistence supports CPU `PackedDocs`. CUDA handles are intentionally not saved
directly; callers should save CPU-packed data and call `bitmax.to_device` after
loading.

### Fused Centroid CUDA Top-K

The current centroid path transforms query vectors on the host, then calls the
regular binary CUDA top-k path. Add a resident CUDA method that accepts raw
queries plus per-dimension centroid weights and applies the weights on device
before scoring. Top-k ordering is unchanged by the centroid constant, so the
kernel only needs weighted binary scores; Python can restore returned top-k
scores with the existing small host-side correction.

### Larger Evaluation

Use the persistent project-owned VAST RTX 4090 worker. Build and evaluate
ViDoRe/ColQwen2 slices at 256 first, then 1000 if the 256 run passes and cost
remains reasonable. Keep using project ledger cleanup rules and do not destroy
the persistent worker unless explicitly asked.

### Int4 Experimental Backend

Add `bitmax.experimental` int4 pack/scoring helpers. Start with symmetric
per-tensor int4 because the quality sweep showed it was the strongest simple
8x-compression reference. The int4 backend must report storage and retrieval
metrics separately from binary; users should be able to choose binary for max
compression or int4 for higher quality.

## Success Criteria

- Packed binary + centroid calibration can be saved, loaded, scored, and moved
  to CUDA.
- CUDA centroid top-k produces the same indices and scores as the existing
  centroid reference path.
- Larger VAST retrieval artifacts report latency, storage reduction, recall@1,
  recall@10, MRR@10, and NDCG@10 for dense, raw binary, centroid binary, and
  int4 reference/backend when available.
- Binary centroid remains the recommended high-compression path unless a
  same-storage calibration variant beats it on larger slices.
- Int4 is documented as an accuracy/storage option, not as a replacement for
  32x binary.

## Non-Goals

- No vector database, index builder, shard manager, or serving API.
- No broad VAST cleanup. Only instances in the project ledger with
  `bitmax-v0-` labels may ever be destroyed.
- No int8 optimization focus beyond using it as a quality reference.
