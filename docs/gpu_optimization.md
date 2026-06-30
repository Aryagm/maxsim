# GPU Optimization Notes

## 2026-06-30: Batched Block-Parallel CUDA MaxSim

Change: `ef5ee2a` adds `maxsim_cuda_batch`, pads ragged retrieval queries into a
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
