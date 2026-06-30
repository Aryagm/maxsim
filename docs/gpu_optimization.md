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
