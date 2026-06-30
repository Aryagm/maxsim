# GPU Optimization Notes

## 2026-06-30: Batched Block-Parallel CUDA MaxSim

Change: `70f6ed1` adds `maxsim_cuda_batch`, pads ragged retrieval queries into a
single batch for CUDA scoring, and replaces the previous one-thread-per-document
CUDA kernel with a 128-thread block per `(query, document)` score.

Root cause addressed:

- The previous retrieval path launched one CUDA kernel per query.
- Each call re-copied packed document bytes and offsets to the GPU.
- The CUDA kernel scored each document serially in one thread, leaving most GPU
  lanes idle for ColQwen/ColPali-style documents with hundreds of vectors.

Measured on a VAST RTX 4090 instance created and destroyed by this project:

| benchmark | previous bitmax CUDA | batched bitmax CUDA | quality |
| --- | ---: | ---: | --- |
| ViDoRe DocVQA + ColQwen2, 64 queries/docs | 1041.22 ms | 3.47 ms | recall@10 0.875, NDCG@10 0.737 unchanged |
| synthetic dim128/q16/doc32/docs512 | not comparable on same GPU | 0.214 ms | exact vs reference |
| synthetic dim128/q32/doc64/docs1024 | not comparable on same GPU | 0.326 ms | exact vs reference |
| synthetic dim256/q32/doc64/docs1024 | not comparable on same GPU | 0.582 ms | exact vs reference |

The retrieval benchmark still shows the same binary-sign quality gap relative to
dense fp16 on the 64-page sample:

- Dense fp16: recall@10 0.890625, NDCG@10 0.788359.
- Bitmax binary: recall@10 0.875, NDCG@10 0.737492.
- Document memory remains 16x smaller than fp16 and 32x smaller than fp32.

Important caveat: one synthetic run on the PyTorch 2.4 VAST image reported
`numpy_torch_equivalent` CPU baselines before optional retrieval dependencies
were installed, because the installed PyTorch build did not advertise support
for the RTX 4090 architecture in that process. Treat the synthetic bitmax CUDA
latencies as kernel timing signal, not final speedup claims against torch CUDA.

Next GPU optimization hypotheses:

1. Keep packed document buffers resident on GPU across calls instead of copying
   `packed` and `offsets` for every query batch.
2. Add a fused CUDA `topk_maxsim` path to avoid copying full `[queries, docs]`
   scores back to host for reranking.
3. Split specialized kernels for `dim=128` and `dim=256` so the byte loop can be
   unrolled and query values can be cached more aggressively.
4. Add scale restoration variants and measure whether global/token scale narrows
   the dense-quality gap without losing most of the speedup.

## 2026-06-30: CUDA-Resident Packed Docs

Change: `7d009e5` adds `bitmax.to_device(packed, "cuda")`, backed by a
pybind-owned `CudaPackedDocs` handle. CUDA scoring can now keep packed document
bytes and offsets resident on GPU across repeated query batches.

Root cause addressed:

- `maxsim_cuda_batch` still copied packed document bytes and offsets from host to
  device for every scoring call.
- Retrieval benchmarks model document packing as offline work, so repeated
  online query batches should not pay document upload cost.

Measured on a VAST RTX 4090 instance created and destroyed by this project:

| benchmark | host-packed CUDA | resident-doc CUDA | speedup | correctness |
| --- | ---: | ---: | ---: | ---: |
| docvqa_like_64, batch64/q23/docs64/doc750/dim128 | 2.67 ms | 2.25 ms | 1.19x | 0.0 |
| rerank_512, batch32/q32/docs512/doc128/dim128 | 2.94 ms | 2.21 ms | 1.33x | 0.0 |
| rerank_1024_dim256, batch16/q32/docs1024/doc64/dim256 | 2.98 ms | 2.43 ms | 1.23x | 0.0 |

The benchmark artifact is
`benchmark-results/cuda-resident-docs-4090.json`. The attempted repeat of the
ViDoRe DocVQA download on this host failed due an incomplete Hugging Face mirror
download, so this pass used controlled retrieval-shaped synthetic tensors to
isolate document residency.

Next GPU optimization hypotheses:

1. Move query buffers and output buffers into reusable CUDA handles as well, so
   repeated online batches avoid `cudaMalloc` and host/device allocation churn.
2. Add fused CUDA top-k to avoid returning full score matrices when callers only
   need the top documents.
3. Start accuracy work with global and per-token scale restoration on the same
   ViDoRe/ColQwen2 benchmark slice.

## 2026-06-30: CUDA-Resident Fused Top-K

Change: CUDA-resident `PackedDocs` handles now expose `topk_batch`, and
`bitmax.topk_maxsim` uses it directly when the packed docs are already resident
on GPU. The fused path computes the full exact score matrix on device, selects
top-k on device with deterministic lower-doc-id tie breaking, and returns only
top-k scores and indices to the host.

Root cause addressed:

- Retrieval callers usually need only top-k documents, but the previous
  CUDA-resident path copied the full `[batch, docs]` float32 score matrix back to
  the host and sorted there.
- The cost gets worse for reranking shapes with thousands of candidate docs,
  where the selected top-k payload is much smaller than the full score matrix.

Measured on a VAST RTX 4090 instance created and destroyed by this project:

| benchmark | full scores + host top-k | fused CUDA top-k | speedup | host bytes returned | correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| topk_docvqa_like_64, batch64/q23/docs64/doc750/dim128/k10 | 2.57 ms | 2.21 ms | 1.16x | 16,384 -> 7,680 | exact |
| topk_rerank_512, batch32/q32/docs512/doc128/dim128/k10 | 3.22 ms | 2.30 ms | 1.40x | 65,536 -> 3,840 | exact |
| topk_rerank_4096, batch16/q32/docs4096/doc32/dim128/k10 | 8.13 ms | 3.55 ms | 2.29x | 262,144 -> 1,920 | exact |

The benchmark artifact is
`benchmark-results/cuda-resident-topk-4090.json`. Exactness here means the
fused top-k scores and indices matched the full score matrix plus host top-k for
all measured rows with `score_delta=0.0`.

Next GPU optimization hypotheses:

1. Move query buffers and top-k output buffers into reusable CUDA handles to cut
   repeated allocation overhead.
2. Specialize `dim=128` and `dim=256` kernels and cache query bytes/values more
   aggressively inside each `(query, document)` block.
3. Start accuracy work with global, per-token, and asymmetric scale restoration
   on the same retrieval-quality benchmark slices, then measure the latency cost.

## 2026-06-30: Reused CUDA-Resident Work Buffers

Change: `CudaPackedDocs` now keeps reusable device work buffers for query
values, full scores, top-k scores, and top-k indices. Buffers grow to the largest
shape seen by the handle and are reused by later `maxsim_batch` and `topk_batch`
calls.

Root cause addressed:

- The CUDA-resident document handle still allocated and freed query/output work
  buffers on every call.
- Repeated retrieval batches are the normal online workload, so steady-state
  calls should not pay repeated `cudaMalloc`/`cudaFree` overhead.

Measured on the same VAST RTX 4090 host, comparing `main` at `ddbb544` against
the buffer-reuse branch:

| benchmark | operation | before | after | speedup | correctness |
| --- | --- | ---: | ---: | ---: | --- |
| small_repeated_64, batch8/q8/docs64/doc16/dim128 | maxsim | 0.048 ms | 0.041 ms | 1.17x | checksum unchanged |
| small_repeated_64, batch8/q8/docs64/doc16/dim128/k10 | top-k | 0.070 ms | 0.058 ms | 1.21x | exact |
| docvqa_like_64, batch64/q23/docs64/doc750/dim128 | maxsim | 2.20 ms | 2.01 ms | 1.09x | checksum unchanged |
| docvqa_like_64, batch64/q23/docs64/doc750/dim128/k10 | top-k | 2.16 ms | 2.03 ms | 1.07x | exact |
| rerank_512, batch32/q32/docs512/doc128/dim128 | maxsim | 2.26 ms | 2.11 ms | 1.07x | checksum unchanged |
| rerank_512, batch32/q32/docs512/doc128/dim128/k10 | top-k | 2.27 ms | 2.13 ms | 1.06x | exact |
| rerank_4096, batch16/q32/docs4096/doc32/dim128 | maxsim | 3.40 ms | 3.38 ms | 1.01x | checksum unchanged |
| rerank_4096, batch16/q32/docs4096/doc32/dim128/k10 | top-k | 3.45 ms | 3.41 ms | 1.01x | exact |

The benchmark artifact is
`benchmark-results/cuda-buffer-reuse-4090.json`. The result is a real but
bounded win: it helps allocation-sensitive and medium retrieval shapes, while
large reranking is now dominated by the MaxSim scoring kernel itself.

Next GPU optimization hypotheses:

1. Specialize `dim=128` and `dim=256` kernels to reduce inner-loop overhead in
   the scoring-dominated cases.
2. Cache or transform query values per block so each document block rereads less
   query data from global memory.
3. Start accuracy work with global, per-token, and asymmetric scale restoration
   on the same retrieval-quality benchmark slices, then measure the latency cost.

## 2026-06-30: Gated Dim-128 CUDA Scoring Specialization

Change: CUDA-resident `PackedDocs` now routes small `dim=128` document sets
through a specialized unrolled scoring kernel. The specialized kernel preserves
the generic kernel's per-dimension accumulation order, unlike a query-byte LUT
prototype that was faster in some cases but introduced small float-order deltas.
The specialization is gated to `num_docs <= 128`; larger reranking shapes stay on
the generic kernel because always-on unrolling regressed those cases.

Root cause addressed:

- The generic CUDA scorer loops over runtime `dim / 8` bytes and per-byte bits
  for every document token.
- For `dim=128`, the byte and bit loop bounds are fixed, so the compiler can
  unroll the binary dot-product body.
- Measurement showed that the unrolled body helps 64-document, long-document
  retrieval batches but hurts high-document-count reranking unless gated.

Measured on the same VAST RTX 4090 host, comparing `main` at `281ed2a` against
the dim-128 specialization branch:

| benchmark | operation | kernel after | before | after | speedup | correctness |
| --- | --- | --- | ---: | ---: | ---: | --- |
| small_repeated_64, batch8/q8/docs64/doc16/dim128 | maxsim | dim128_unrolled | 0.044 ms | 0.039 ms | 1.11x | checksum unchanged |
| small_repeated_64, batch8/q8/docs64/doc16/dim128/k10 | top-k | dim128_unrolled | 0.062 ms | 0.056 ms | 1.10x | exact |
| docvqa_like_64, batch64/q23/docs64/doc750/dim128 | maxsim | dim128_unrolled | 2.22 ms | 1.15 ms | 1.92x | checksum unchanged |
| docvqa_like_64, batch64/q23/docs64/doc750/dim128/k10 | top-k | dim128_unrolled | 2.19 ms | 1.17 ms | 1.88x | exact |
| rerank_512, batch32/q32/docs512/doc128/dim128 | maxsim | generic | 2.11 ms | 2.11 ms | 1.00x | checksum unchanged |
| rerank_512, batch32/q32/docs512/doc128/dim128/k10 | top-k | generic | 2.13 ms | 2.13 ms | 1.00x | exact |
| rerank_4096, batch16/q32/docs4096/doc32/dim128 | maxsim | generic | 3.39 ms | 3.38 ms | 1.00x | checksum unchanged |
| rerank_4096, batch16/q32/docs4096/doc32/dim128/k10 | top-k | generic | 3.42 ms | 3.43 ms | 1.00x | exact |
| dim256_control_1024, batch16/q32/docs1024/doc64/dim256 | maxsim | generic | 2.32 ms | 2.33 ms | 1.00x | checksum unchanged |

The benchmark artifact is
`benchmark-results/cuda-dim128-unrolled-4090.json`. The key research result is
that shape-aware routing matters: an ungated dim-128 unrolled kernel improved
DocVQA-like shapes but slowed 512/4096-document reranking, so v0.1 keeps the
fast path only where current measurements support it.

Next GPU optimization hypotheses:

1. Try a query-byte LUT again only behind a shape gate and with explicit
   score-delta reporting, because it may help high-token documents but changes
   float accumulation order.
2. Add a separate dim-256 specialization and benchmark it with the same
   same-host baseline/candidate method before enabling it.
3. Start accuracy work with global, per-token, and asymmetric scale restoration
   on retrieval-quality benchmark slices, tracking both quality and kernel cost.

## 2026-06-30: Per-Document Scale Restoration Probe

Change: `pack_signs(..., scale="doc")` stores one `mean(abs(doc_tokens))` scale
per document. `maxsim` applies that vector after native scoring, and
`topk_maxsim` ranks after scale restoration. This is an optional accuracy mode;
the default remains unscaled signs.

Root cause addressed:

- Binary signs discard magnitude. A single global scale changes score magnitude
  but cannot change ranking within a query.
- Per-document scale can restore some document-level magnitude signal while
  keeping the packed sign representation unchanged.

Measured locally on a targeted retrieval fixture where two documents have
identical signs but different magnitudes:

| scale | recall@1 | NDCG@10 | note |
| --- | ---: | ---: | --- |
| none | 0.00 | 0.631 | lower doc id wins an unscaled tie |
| doc | 1.00 | 1.000 | scale restores dense ranking |

Measured on a VAST RTX 4090 with ViDoRe DocVQA + ColQwen2, 64 queries/docs:

| scale | latency | recall@1 | recall@10 | MRR@10 | NDCG@10 | speedup vs dense fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 1.89 ms | 0.609 | 0.875 | 0.695 | 0.737 | 134.1x |
| global | 1.89 ms | 0.609 | 0.875 | 0.695 | 0.737 | 129.8x |
| doc | 1.89 ms | 0.625 | 0.844 | 0.705 | 0.738 | 130.8x |

Artifacts:

- `benchmark-results/retrieval-doc-scale-targeted-none.json`
- `benchmark-results/retrieval-doc-scale-targeted-doc.json`
- `benchmark-results/retrieval-vidore-limit64-cuda-scale-comparison.json`

The result is mixed: per-document scale improved recall@1, MRR, and NDCG
slightly on the 64-query slice, but reduced recall@10. It should remain an
opt-in accuracy knob until broader slices show a stable win.

Next GPU optimization hypotheses:

1. Evaluate per-token or per-dimension scale variants on larger ViDoRe slices,
   using recall@1, recall@10, MRR, and NDCG as separate gates.
2. Keep unscaled signs as the default until a scale mode improves the broad
   quality profile without a large latency penalty.

## 2026-06-30: Resident CUDA Top-K With Per-Document Scale

Change: CUDA-resident `PackedDocs` upload per-document scale vectors and fused
`topk_maxsim` now applies those scales on GPU before selecting top-k. This keeps
scale-restored ranking on the fused CUDA path instead of materializing a full
score matrix for host-side scale and sort.

Measured on a VAST RTX 4090 instance created and destroyed by this project:

| benchmark | full scores + host scale top-k | resident scale fused top-k | speedup | host bytes returned | correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| docscale_docvqa_like_64, batch64/q23/docs64/doc750/dim128/k10 | 1.52 ms | 1.30 ms | 1.17x | 16,384 -> 7,680 | exact |
| docscale_rerank_512, batch32/q32/docs512/doc128/dim128/k10 | 2.99 ms | 2.30 ms | 1.30x | 65,536 -> 3,840 | exact |
| docscale_rerank_4096, batch16/q32/docs4096/doc32/dim128/k10 | 7.59 ms | 3.71 ms | 2.05x | 262,144 -> 1,920 | exact |

The benchmark artifact is
`benchmark-results/cuda-doc-scale-resident-4090.json`. Exactness here means
score deltas were zero and top-k indices matched full-score ranking.

## 2026-06-30: Experimental Pareto Variants

Change: benchmark-only reference implementations now cover:

- int8-query x ternary-doc scoring with a percentile threshold.
- int8-query x binary-doc with per-token magnitude scales.
- int8-query x binary-doc with per-16-dimension grouped scales.
- int8-query x binary-doc with calibrated per-dimension thresholds.

These are deliberately not public `bitmax` kernels yet. They are probes to find
accuracy/storage wins before writing more CUDA.

Local blog-shape reference artifact:
`benchmark-results/blog-baseline-blog-shape-variants.json`.

| variant | bytes/doc | fp32 storage reduction | latency | max score delta vs fp32 |
| --- | ---: | ---: | ---: | ---: |
| fp32 query/docs | 402,432 | 1.0x | 27.58 ms | 0.00 |
| int8 query/int8 docs | 100,608 | 4.0x | 46.38 ms | 286.00 |
| int8 query/binary docs | 12,576 | 32.0x | 111.05 ms | 2741.61 |
| binary query/binary docs | 12,576 | 32.0x | 116.68 ms | 3936.61 |
| int8 query/ternary docs | 25,152 | 16.0x | 46.89 ms | 2983.61 |
| int8 query/binary docs + token scale | 15,720 | 25.6x | 89.94 ms | 1222.35 |
| int8 query/binary docs + group scale 16 | 37,728 | 10.7x | 156.59 ms | 1179.47 |
| int8 query/binary docs + calibrated threshold | 12,576 | 32.0x | 83.48 ms | 2741.61 |

Local retrieval artifacts:

- `benchmark-results/retrieval-fixture-pareto-variants.json`
- `benchmark-results/retrieval-doc-scale-targeted-pareto-variants.json`

On the targeted magnitude fixture, raw binary preserves the 32x fp32 storage
reduction but misses the dense top-1 document because tied signs lose magnitude:
recall@1 = 0.0, NDCG@10 = 0.631. Per-document scale, per-token scale, grouped
scale, ternary thresholding, and calibrated thresholds all restore recall@1 and
NDCG@10 to 1.0 on that fixture. The storage cost differs: ternary uses 16x fp32
compression, while doc/token/group scale use 6.4x compression on the tiny
fixture because scale metadata dominates at three documents.

Interpretation:

1. The clearest accuracy signal is magnitude restoration, not ternary, on the
   synthetic/blog and targeted retrieval fixtures.
2. Per-token and group scale cut score error by roughly 55-57% versus raw
   int8-query x binary-doc scoring on the blog-shape fixture, but the current
   implementations are CPU reference paths and slower than dense fp32.
3. Calibrated thresholds did not improve synthetic score delta in the current
   median-threshold form; it remains a candidate only if held-out retrieval
   quality improves.
4. Keep unscaled binary docs as the default until a real retrieval slice shows a
   stable recall/NDCG gain after accounting for storage and GPU latency.

## 2026-06-30: GPU-First Dim128 Query LUT Top-K

Change: CUDA-resident `topk_maxsim` now routes int8/integer query tensors with
`dim=128` and more than 128 documents through a query-byte lookup-table kernel.
The LUT stores the 256 possible dot contributions for each `(batch, query token,
packed byte)` and turns each 128-dimensional binary dot product into 16 table
loads. The small-doc dim128 path remains on the prior unrolled kernel because
LUT construction overhead dominates there.

Measured on the persistent VAST RTX 4090 worker created by this project:

| benchmark | resident top-k | LUT direct | routed API | API speedup | correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| docvqa_like_64, batch64/q23/docs64/doc750/dim128/k10 | 1.287 ms | 2.551 ms | 1.304 ms | 0.99x | exact |
| rerank_512, batch32/q32/docs512/doc128/dim128/k10 | 2.222 ms | 1.403 ms | 1.440 ms | 1.54x | exact |
| rerank_4096, batch16/q32/docs4096/doc32/dim128/k10 | 3.622 ms | 2.210 ms | 2.220 ms | 1.63x | exact |
| blog_single_query, batch1/q33/docs1000/doc786/dim128/k10 | 0.832 ms | 0.485 ms | 0.485 ms | 1.71x | exact |
| docscale_blog_single_query, batch1/q33/docs1000/doc786/dim128/k10 | 0.887 ms | 0.480 ms | 0.487 ms | 1.82x | exact |

The benchmark artifact is
`benchmark-results/cuda-dim128-lut-topk-4090.json`. Exactness here means
`score_delta=0.0` and top-k indices matched the prior resident top-k path for
integer query tensors.

The same worker measured the blog-style shape against torch dense top-k:

| implementation | bytes/doc | median latency | speedup vs torch fp32 | speedup vs blog 3.71 ms row |
| --- | ---: | ---: | ---: | ---: |
| torch fp32 docs/query | 402,432 | 0.723 ms | 1.00x | n/a |
| torch fp16 docs/query | 201,216 | 0.381 ms | 1.90x | n/a |
| bitmax int8 query/binary docs LUT top-k | 12,576 | 0.463 ms | 1.56x | 8.02x |
| bitmax int8 query/binary docs + doc scale LUT top-k | 12,580 | 0.480 ms | 1.51x | 7.73x |

The benchmark artifact is
`benchmark-results/blog-shape-gpu-topk-4090.json`. This is the first measured
GPU result that beats the blog table's int8-query/binary-doc latency while
keeping the same 12.28 KiB/doc storage target. It is still a synthetic
blog-shape benchmark; real retrieval NDCG must be rerun on a ViDoRe slice before
claiming an accuracy improvement over the blog.

Streaming top-k status:

- The true streaming CUDA top-k kernel, which avoids materializing the full
  `[batch, docs]` score matrix, now compiles and passes CUDA correctness tests.
- Same-host timing showed it is slower than resident score-matrix top-k because
  per-document global top-k locking dominates.
- It is intentionally not routed by the public Python API.

## 2026-06-30: Real ViDoRe Accuracy/Size Frontier

Change: added an experimental one-bit per-dimension centroid calibration path
under `bitmax.experimental`. It still stores document tokens as packed signs,
but learns positive/negative centroids per dimension and transforms query
vectors before calling the existing CUDA binary MaxSim kernel. This is a
library-level accuracy knob, not a database/index feature.

Measured on the persistent project-owned VAST RTX 4090 worker with
`vidore/docvqa_test_subsampled:test:64` embedded by
`vidore/colqwen2-v1.0-hf`:

| implementation | total doc bytes | fp32 doc reduction | latency | recall@1 | recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense fp16 baseline | 12,290,048 | 2.0x | 264.42 ms | 0.688 | 0.891 | 0.756 | 0.788 |
| raw binary CUDA | 768,128 | 32.0x | 1.91 ms | 0.609 | 0.875 | 0.695 | 0.737 |
| binary + doc scale CUDA | 768,384 | 32.0x | 1.90 ms | 0.625 | 0.844 | 0.705 | 0.738 |
| ternary threshold CPU reference | 1,536,256 | 16.0x | 942.75 ms | 0.609 | 0.875 | 0.700 | 0.742 |
| binary + per-dim centroids CUDA | 769,664 | 31.9x | 1.91 ms | 0.609 | 0.891 | 0.700 | 0.746 |
| binary + Lloyd centroids CUDA | 769,664 | 31.9x | 1.90 ms | 0.641 | 0.812 | 0.695 | 0.723 |

Artifacts:

- `benchmark-results/retrieval-docvqa-colqwen2-limit64-centroid-api-r5.json`
- `benchmark-results/retrieval-docvqa-colqwen2-limit64-centroid-threshold-sweep.json`
- `benchmark-results/retrieval-docvqa-colqwen2-limit64-quant-quality-sweep.json`

Decision:

1. Promote zero-threshold per-dimension centroids as an experimental
   GPU-first path. It improved NDCG@10 from 0.737 to 0.746 and recovered dense
   recall@10 while keeping effectively the same 32x fp32 document compression
   and the same CUDA binary kernel latency.
2. Do not promote median/quantile threshold-only packing. The threshold sweep
   showed zero-threshold centroids were best; median and quantile thresholds
   reduced NDCG on this slice.
3. Do not write a per-token/group-scale GPU kernel yet. Those variants were
   worse than raw binary on NDCG@10 and used more storage.
4. Ternary is not currently Pareto-optimal: it used 2x the binary storage and
   was lower-quality than centroid binary on this real slice.

Reference quality frontier from the same slice:

| reference format | fp32 doc reduction | recall@1 | recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uint8 affine per-dim docs | 4.0x | 0.688 | 0.891 | 0.757 | 0.789 |
| int4 symmetric per-tensor docs | 8.0x | 0.656 | 0.875 | 0.737 | 0.771 |
| binary + per-dim centroids | 31.9x | 0.609 | 0.891 | 0.700 | 0.746 |
| int2 symmetric per-token docs | 14.2x | 0.594 | 0.812 | 0.678 | 0.711 |

The int8/int4 rows are quality-reference paths, not custom optimized kernels.
They define the next GPU research targets: int4 may be the better accuracy
option when 8x compression is acceptable, while centroid binary is the best
measured high-compression path so far.
