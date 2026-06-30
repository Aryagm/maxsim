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
