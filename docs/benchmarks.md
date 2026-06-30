# Benchmark Ladder

Run benchmarks in order. Do not start larger VAST runs until the earlier gate
JSON exists and reports `"gate_passed": true`.

```bash
python benchmarks/run_synthetic.py --stage stage0 --output benchmark-results/stage0.json
python benchmarks/run_synthetic.py --stage cpu-smoke --output benchmark-results/cpu-smoke.json
python benchmarks/run_synthetic.py --stage cuda-smoke --output benchmark-results/cuda-smoke.json
python benchmarks/run_synthetic.py --stage cuda-sweep \
  --gate benchmark-results/cuda-smoke.json \
  --output benchmark-results/cuda-sweep.json
python benchmarks/run_synthetic.py --stage vast-large \
  --gate benchmark-results/cuda-sweep.json \
  --output benchmark-results/vast-large.json
```

Each JSON row includes implementation, shape, latency, docs/sec, bytes read,
score checksum, and correctness delta against the Python reference.

## Blog-Style Quantization Baseline

Use this benchmark to reproduce the blogpost-style late-interaction scoring
shape before changing kernels or formats. It compares fp32 query/docs,
int8 query/docs, int8 query with binary docs, binary query/docs, and
experimental ternary/scale/threshold variants.

```bash
python -m benchmarks.blog_baseline \
  --stage smoke \
  --output benchmark-results/blog-baseline-smoke.json
python -m benchmarks.blog_baseline \
  --stage blog-shape \
  --repeat 1 \
  --output benchmark-results/blog-baseline-blog-shape.json
```

Rows report `latency_ms`, `doc_storage_bytes_per_doc`,
`speedup_vs_fp32`, and `max_abs_delta_vs_reference`. The `blog-shape`
stage uses the blogpost scoring dimensions: 33 query tokens, 1000
documents, 786 document tokens per document, and 128 dimensions.

The current local reference artifact is
`benchmark-results/blog-baseline-blog-shape-variants.json`. It is a CPU/NumPy
signal benchmark, not an optimized kernel result. On that artifact:

- raw int8-query x binary-doc keeps the blog storage target at 12,576 bytes/doc
  (32x smaller than fp32) but is slower than local fp32 in the reference path.
- per-token scale lowers max score delta from 2741.61 to 1222.35 while keeping
  15,720 bytes/doc (25.6x smaller than fp32).
- grouped scale lowers max score delta to 1179.47 at 37,728 bytes/doc
  (10.7x smaller than fp32).
- ternary docs use 25,152 bytes/doc (16x smaller than fp32) but did not reduce
  score delta on this synthetic distribution.

## Schema v2

Top-level fields:

- `schema_version`: currently `2`.
- `stage`: benchmark stage name.
- `baselines`: expected baseline implementation names.
- `gate_passed`: true when all gate-blocking rows are within tolerance.
- `results`: per-implementation timing and correctness rows.

Rows with `gate_blocking: true` decide whether the stage can unlock the next
stage. PyTorch-style baseline rows are measurement rows, so they report
correctness deltas but do not block gates.

Native rows include:

- `speedup_vs_torch_fp16`
- `speedup_vs_torch_int8`
- `doc_memory_compression_vs_fp16`
- `doc_memory_compression_vs_fp32`
- `baseline_latency_ms`

The `torch_fp16_baseline` and `torch_int8_baseline` rows use vectorized dense
MaxSim over the uniform synthetic document layout and include:

- `baseline_backend`: `torch` when PyTorch is installed, otherwise
  `numpy_torch_equivalent`.
- `baseline_device`: `cuda` when the installed PyTorch build supports the
  current GPU architecture, otherwise `cpu`.
- `requested_baseline_device`: the requested benchmark device.
- `formula`: `dense_fp16_vectorized_maxsim` or
  `dense_int8_vectorized_doc_maxsim`.

## Retrieval Benchmarks

The retrieval benchmark measures the online reranking path over multi-vector
embeddings and qrels. It still does not build an index or database. The input is
an `.npz` file with:

- `doc_embeddings`: `[total_doc_tokens, dim]` float array.
- `doc_offsets`: `[num_docs + 1]` int64 offsets into `doc_embeddings`.
- `query_embeddings`: `[num_queries, query_tokens, dim]`, or flattened
  `[total_query_tokens, dim]` with `query_offsets`.
- `qrels`: dense `[num_queries, num_docs]` relevance matrix, or
  `relevant_doc_ids`: one positive doc index per query.
- Optional `dataset_name`, `query_ids`, and `doc_ids`.

Run the local retrieval smoke first:

```bash
python benchmarks/run_retrieval.py \
  --stage fixture-smoke \
  --output benchmark-results/retrieval-fixture.json
```

Run a real embedding file on CPU before renting a GPU:

```bash
python benchmarks/build_vidore_embeddings.py \
  --dataset vidore/docvqa_test_subsampled \
  --split test \
  --limit 16 \
  --model vidore/colqwen2-v1.0-hf \
  --output benchmark-results/vidore-docvqa-colqwen2.npz
python benchmarks/run_retrieval.py \
  --stage embeddings-smoke \
  --input benchmark-results/vidore-docvqa-colqwen2.npz \
  --output benchmark-results/retrieval-embeddings-smoke.json
```

Then run the same embedding file on CUDA only after the CPU gate passes:

```bash
python benchmarks/run_retrieval.py \
  --stage embeddings-cuda-smoke \
  --input benchmark-results/vidore-docvqa-colqwen2.npz \
  --gate benchmark-results/retrieval-embeddings-smoke.json \
  --output benchmark-results/retrieval-embeddings-cuda-smoke.json
```

Retrieval rows report latency, queries/sec, recall, MRR, NDCG, top-k agreement
with dense fp16 MaxSim, speedup for the bitmax row, and document-memory
compression. Dense fp16 uses the original document embeddings; bitmax packs the
same document embeddings to one-bit signs and scores with `bitmax.maxsim`.
Pass `--scale global` or `--scale doc` to test global or per-document scale
restoration. `doc` scale can improve ranking when magnitude differences between
documents carry useful signal. CUDA-resident packed docs upload stored doc-scale
vectors and apply them inside resident `maxsim` and fused `topk_maxsim`; CPU and
host-packed CUDA paths apply vector scales after native scoring.

CUDA-resident top-k kernel experiments can be run directly on a CUDA worker:

```bash
python -m benchmarks.run_cuda_topk \
  --stage lut-sweep \
  --output benchmark-results/cuda-dim128-lut-topk-4090.json
python -m benchmarks.run_cuda_topk \
  --stage blog-shape \
  --output benchmark-results/blog-shape-gpu-topk-4090.json
```

These stages require PyTorch with CUDA. `lut-sweep` compares the default
resident top-k path with the dim128 query-byte LUT path on retrieval-shaped
synthetic cases. `blog-shape` compares torch fp32/fp16 dense top-k with bitmax
int8-query/binary-doc top-k on the blog-style 33 x 1000 x 786 x 128 shape.

Pass `--variants all` to emit the experimental Pareto rows:

```bash
python benchmarks/run_retrieval.py \
  --stage fixture-smoke \
  --variants all \
  --output benchmark-results/retrieval-fixture-pareto-variants.json
python benchmarks/run_retrieval.py \
  --stage embeddings-smoke \
  --input benchmark-results/doc-scale-targeted.npz \
  --variants all \
  --output benchmark-results/retrieval-doc-scale-targeted-pareto-variants.json
```

The variant set is `binary`, `binary_doc_scale`, `ternary_threshold`,
`binary_token_scale`, `binary_group_scale_16`, `int4_symmetric_per_tensor`,
`binary_calibrated_threshold`, `binary_dim_centroid_zero`,
`binary_dim_centroid_q40`, and `binary_dim_centroid_lloyd`. Ternary and
token/group/threshold variants are benchmark-only reference implementations
until a retrieval-quality win justifies moving them into public kernels.
`binary_dim_centroid_zero` and `binary_dim_centroid_q40` use
`bitmax.experimental` helpers and reuse the existing CUDA binary MaxSim path
with query preprocessing. `int4_symmetric_per_tensor` uses the experimental
CUDA int4 backend when the stage device is CUDA.

Measured on the persistent project-owned VAST RTX 4090 worker with
`vidore/docvqa_test_subsampled:test:256` embedded by `vidore/colqwen2-v1.0-hf`:

| implementation | fp32 doc reduction | latency | speedup vs dense fp16 | recall@1 | recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense fp16 baseline | 2.00x | 3790.37 ms | 1.00x | 0.551 | 0.777 | 0.623 | 0.660 |
| raw binary CUDA | 32.00x | 49.05 ms | 77.28x | 0.547 | 0.754 | 0.615 | 0.649 |
| q40 centroid binary CUDA | 31.98x | 49.01 ms | 77.34x | 0.547 | 0.762 | 0.617 | 0.652 |
| int4 symmetric CUDA | 8.00x | 192.13 ms | 19.73x | 0.555 | 0.773 | 0.622 | 0.658 |

Artifact:
`benchmark-results/retrieval-docvqa-colqwen2-limit256-int4-q40-cuda-focused.json`.

## SDK CUDA Demo

The SDK demo is the main production-facing proof path. It uses `bitmax.Corpus`
and `bitmax.Reranker`, compares against dense fp16 CUDA on the same embedding
slice, and reports storage, latency, speedup, recall, MRR, and NDCG.

```bash
python -m examples.local_multivector_search \
  --input benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
  --device cuda \
  --modes binary,binary_q40,int4 \
  --repeat 5 \
  --output benchmark-results/sdk-demo-local-search-limit256-cuda.json
```

Measured on the persistent project-owned VAST RTX 4090 worker with
`vidore/docvqa_test_subsampled:test:256` embedded by `vidore/colqwen2-v1.0-hf`:

| implementation | fp32 doc reduction | latency | speedup vs dense fp16 | recall@1 | recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 2.00x | 1906.08 ms | 1.00x | 0.551 | 0.777 | 0.623 | 0.660 |
| SDK binary CUDA | 32.00x | 58.79 ms | 32.42x | 0.547 | 0.754 | 0.615 | 0.649 |
| SDK binary_q40 CUDA | 31.98x | 64.35 ms | 29.62x | 0.547 | 0.762 | 0.617 | 0.652 |
| SDK int4 CUDA | 8.00x | 144.19 ms | 13.22x | 0.555 | 0.773 | 0.622 | 0.658 |

Artifact: `benchmark-results/sdk-demo-local-search-limit256-cuda.json`.
