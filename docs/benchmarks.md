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

## Open-Source CUDA Comparison

Use this benchmark to compare the SDK against popular open-source GPU vector
search and multi-vector baselines on the same embedding slice:

```bash
python -m pip install ".[oss-bench]"
python -m benchmarks.compare_open_source \
  --input benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
  --device cuda \
  --implementations dense_fp16,faiss_pooled,faiss_token_dense_rerank,qdrant_multivector,cuvs_pooled,fast_plaid,colbert_plaid,vespa_multivector,bitmax_binary,bitmax_binary_q40,bitmax_int4 \
  --repeat 3 \
  --metric-ks 1,5,10 \
  --faiss-token-topn 512 \
  --allow-unavailable \
  --output benchmark-results/open-source-comparison-expanded-limit256-cuda.json
```

Every completed row preserves the older `latency_ms` and `ndcg_at_10` fields,
and also emits richer diagnostics:

- `latency_samples_ms`, `latency_mean_ms`, `latency_p50_ms`,
  `latency_p95_ms`, and `latency_p99_ms`.
- `recall_at_K`, `mrr_at_K`, `ndcg_at_K`, and
  `quality_delta_vs_dense_ndcg_at_K` for each cutoff in `--metric-ks`.
- `token_bucket_quality_at_K` for short, medium, and long pages, where buckets
  are assigned by the relevant document's page-token count on that dataset.

Rows are intentionally labeled by what they do:

- `faiss_gpu_mean_pool_flat_ip`: popular single-vector FAISS GPU flat inner
  product search over mean-pooled document/query vectors. This is extremely
  fast and tiny, but it is not late interaction.
- `cuvs_gpu_mean_pool_flat_ip`: NVIDIA cuVS GPU brute-force inner product over
  the same mean-pooled vectors. This is a CUDA-native single-vector baseline,
  not late interaction.
- `faiss_gpu_token_candidates_dense_rerank`: FAISS GPU flat search over all
  document token vectors to produce candidate documents, followed by dense fp16
  MaxSim reranking of those candidates. This recovers dense quality on the
  measured slice but stores full token vectors and is slower than dense fp16.
- `qdrant_multivector`: Qdrant local in-memory multivector MaxSim with full
  float32 token vectors. This measures an exact production-style multivector
  API path, not a GPU kernel path.
- `fast_plaid`: fast-plaid CUDA search over its own packed/indexed document
  embeddings. The table reports search latency only; `build_ms` is stored in
  the JSON artifact.
- `colbert_plaid`: not run in this embedding-array harness because ColBERT's
  PLAID stack owns the ColBERT text/model/index pipeline. Use `fast_plaid` here
  for the embedding-array PLAID-style comparison.
- `vespa_multivector`: not run in this local CUDA harness because Vespa needs a
  running service, schema, and ingestion benchmark.

Measured on the persistent project-owned VAST RTX 4090 worker with
`vidore/docvqa_test_subsampled:test:256` embedded by `vidore/colqwen2-v1.0-hf`:

| implementation | kind | fp32 doc reduction | latency | speedup vs dense fp16 | recall@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | dense baseline | 2.00x | 1926.10 ms | 1.00x | 0.777 | 0.660 |
| FAISS GPU mean-pool flat IP | open-source single-vector | 752.21x | 0.75 ms | 2552.09x | 0.414 | 0.290 |
| cuVS GPU mean-pool flat IP | open-source single-vector | 752.21x | 0.98 ms | 1959.79x | 0.414 | 0.290 |
| FAISS GPU token candidates + dense rerank | open-source token baseline | 0.67x | 2884.28 ms | 0.67x | 0.777 | 0.660 |
| Qdrant in-memory multivector | open-source multivector API | 1.00x | 33660.60 ms | 0.06x | 0.777 | 0.660 |
| fast-plaid CUDA | open-source PLAID-style | 3.37x | 1856.76 ms | 1.04x | 0.766 | 0.660 |
| bitmax binary CUDA | bitmax SDK | 32.00x | 62.28 ms | 30.93x | 0.754 | 0.649 |
| bitmax binary_q40 CUDA | bitmax SDK | 31.98x | 64.73 ms | 29.76x | 0.762 | 0.652 |
| bitmax int4 CUDA | bitmax SDK | 8.00x | 144.24 ms | 13.35x | 0.773 | 0.658 |

Artifact: `benchmark-results/open-source-comparison-expanded-limit256-cuda.json`.

Interpretation: FAISS and cuVS mean-pooling win raw speed and storage but lose
most of the multi-vector quality. FAISS token-candidate retrieval and Qdrant
exact multivector recover dense quality but are slower on this slice and keep
full-size vectors. fast-plaid keeps dense-level NDCG with modest compression,
but its search latency is near dense fp16 here. On this slice, `bitmax` is the
better compressed CUDA Pareto point for late-interaction reranking: it keeps
most of dense MaxSim quality while reducing document storage by 8-32x and
improving CUDA latency by 13-31x.

## Multi-Dataset Proof

Use the multi-dataset runner to test the same fixed public model embeddings
across multiple ViDoRe datasets. The runner is cache-first: it uses existing
`.npz` embedding files when present, and only creates missing embeddings when
`--build-missing` is passed.

```bash
python -m benchmarks.run_multidataset \
  --datasets docvqa,infovqa,arxivqa,tabfquad \
  --embedding-dir benchmark-results \
  --output-dir benchmark-results/multidataset-limit64 \
  --summary-output benchmark-results/multidataset-limit64-summary.json \
  --model vidore/colqwen2-v1.0-hf \
  --limit 64 \
  --device cuda \
  --repeat 3 \
  --metric-ks 1,5,10 \
  --allow-unavailable \
  --build-missing
```

For corpus scaling, pass multiple limits. The runner reuses existing caches and
only builds missing `.npz` files when `--build-missing` is present:

```bash
python -m benchmarks.run_multidataset \
  --datasets docvqa \
  --embedding-dir benchmark-results \
  --output-dir benchmark-results/multidataset-docvqa-scaling \
  --summary-output benchmark-results/multidataset-docvqa-scaling-summary.json \
  --model vidore/colqwen2-v1.0-hf \
  --limits 64,256,1024 \
  --device cuda \
  --repeat 5 \
  --metric-ks 1,5,10 \
  --allow-unavailable
```

Default runnable implementations are dense fp16, FAISS pooled, cuVS pooled,
fast-plaid, and the bitmax binary/q40/int4 SDK modes. Add `--include-slow` to
also run FAISS token-candidate dense rerank and Qdrant exact multivector.

This benchmark keeps `bitmax` scoped to the compression/scoring layer. The
embeddings are produced by the frozen public model named in `--model`, cached as
artifacts, and reused by every backend.

The summary JSON includes:

- `per_dataset`: full per-dataset comparison artifacts.
- `per_dataset_deltas`: recall/MRR/NDCG deltas versus dense fp16 for every
  measured cutoff.
- `aggregate_results`: mean best latency, mean P50/P95/P99 latency, mean
  quality metrics, storage compression, speedup, and token-bucket rollups.
- `scaling_results`: the same aggregate rows grouped by corpus limit.

Measured on the persistent project-owned VAST RTX 4090 worker across
`docvqa`, `infovqa`, `arxivqa`, and `tabfquad` with limit64 cached ColQwen2
embeddings:

| implementation | datasets | fp32 doc reduction | mean latency | mean P95 | geomean speedup vs dense | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 4 | 2.00x | 109.82 ms | 187.71 ms | 1.00x | 0.949 | 0.885 |
| FAISS token candidates + dense rerank | 4 | 0.67x | 224.56 ms | 228.28 ms | 0.49x | 0.949 | 0.885 |
| Qdrant in-memory multivector | 4 | 1.00x | 1782.25 ms | 2104.45 ms | 0.06x | 0.949 | 0.885 |
| bitmax int4 CUDA | 4 | 8.00x | 33.68 ms | 34.52 ms | 3.32x | 0.949 | 0.879 |
| fast-plaid CUDA | 4 | 3.22x | 182.29 ms | 219.26 ms | 0.60x | 0.941 | 0.878 |
| bitmax binary CUDA | 4 | 32.00x | 9.26 ms | 11.20 ms | 11.89x | 0.941 | 0.865 |
| bitmax binary_q40 CUDA | 4 | 31.92x | 10.70 ms | 11.02 ms | 10.27x | 0.930 | 0.858 |
| FAISS GPU mean-pool flat IP | 4 | 707.80x | 0.24 ms | 6.45 ms | 456.09x | 0.809 | 0.650 |
| cuVS GPU mean-pool flat IP | 4 | 707.80x | 0.44 ms | 6.86 ms | 246.54x | 0.809 | 0.650 |

Artifact: `benchmark-results/multidataset-limit64-rich-summary.json`.

Interpretation: the multi-dataset result supports the same product framing as
the single-slice result. Single-vector pooled baselines are tiny and fast but
lose retrieval quality. Full dense/exact multivector paths preserve quality but
are larger and slower. `bitmax` int4 is the most conservative compressed option
in this run, averaging only `-0.005` NDCG@10 versus dense while reducing storage
by 8x and improving latency by 3.32x. `bitmax` binary is the aggressive option,
reducing storage by 32x and improving latency by 11.89x with an average
`-0.020` NDCG@10 delta.

The same schema also records page-token bucket quality. On this four-dataset
run, binary's NDCG@10 delta is concentrated in the short-page bucket
(`-0.034`); long-page binary is effectively tied with dense (`+0.001`).

The scaling artifact `benchmark-results/multidataset-docvqa-scaling-rich-summary.json`
runs `docvqa` at limits 64 and 256. At limit256, `bitmax_int4` is 12.50x
faster than dense fp16 with NDCG@10 `0.658` vs dense `0.660`, while
`bitmax_binary` is 30.58x faster with NDCG@10 `0.649`.

For larger document-count scaling, use fixed-query mixed corpora built from
compatible real embedding caches. The combiner keeps document/query ids unique
and writes block-diagonal qrels:

```bash
python -m benchmarks.build_mixed_embeddings \
  --inputs benchmark-results/vidore-syntheticdocqa-energy-colqwen2-limit1000.npz \
           benchmark-results/vidore-syntheticdocqa-healthcare-colqwen2-limit1000.npz \
           benchmark-results/vidore-syntheticdocqa-government-colqwen2-limit1000.npz \
  --dataset-name vidore/mixed_syntheticdocqa:test:3000 \
  --output benchmark-results/vidore-mixed-syntheticdocqa-colqwen2-limit3000.npz
```

Measured with 256 fixed queries on the same VAST RTX 4090 worker:

| real docs | dense fp16 latency | fast-plaid latency | bitmax int4 latency | bitmax binary latency | binary speedup | dense NDCG@10 | int4 NDCG@10 | binary NDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 977 | 6886.31 ms | 4657.31 ms | 503.63 ms | 141.66 ms | 48.61x | 0.379 | 0.380 | 0.375 |
| 1,942 | 13517.36 ms | 7744.62 ms | 997.71 ms | 268.86 ms | 50.28x | 0.379 | 0.379 | 0.374 |
| 2,914 | 20339.91 ms | 12165.72 ms | 1416.06 ms | 376.26 ms | 54.06x | 0.379 | 0.379 | 0.374 |

Artifact: `benchmark-results/mixed-syntheticdocqa-docscale-rich-summary.json`.

For a fully unique larger-corpus comparison, use the mixed 5k embedding cache
directly with the open-source comparison runner:

```bash
python -m benchmarks.compare_open_source \
  --input benchmark-results/vidore-mixed-syntheticdocqa-colqwen2-limit5000.npz \
  --output benchmark-results/unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json \
  --device cuda \
  --implementations dense_fp16,faiss_pooled,cuvs_pooled,fast_plaid,bitmax_binary,bitmax_binary_q40,bitmax_int4 \
  --limit-queries 256 \
  --repeat 3 \
  --metric-ks 1,5,10 \
  --allow-unavailable
```

Measured with 256 queries and 4,882 unique documents on the VAST RTX 4090
worker:

| implementation | fp32 doc reduction | P95 latency | speedup vs dense fp16 | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 2.00x | 34.32s | 1.00x | 0.383 | 0.378 |
| fast-plaid CUDA | 3.45x | 21.76s | 1.63x | 0.387 | 0.379 |
| bitmax binary CUDA | 32.00x | 627ms | 54.53x | 0.387 | 0.374 |
| bitmax binary_q40 CUDA | 32.00x | 640ms | 53.38x | 0.383 | 0.375 |
| bitmax int4 CUDA | 8.00x | 2.38s | 14.34x | 0.383 | 0.379 |

Artifact:
`benchmark-results/unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json`.

For a larger fixed-query stress sweep, start from the mixed real 5k corpus and
expand document count with repeated non-positive real page embeddings. This is a
latency/memory and distractor-crowding stress test, not a claim that the 25k
corpus contains 25k unique pages:

```bash
python -m benchmarks.build_docscale_stress \
  --input benchmark-results/vidore-mixed-syntheticdocqa-colqwen2-limit5000.npz \
  --target-docs 25000 \
  --query-limit 256 \
  --dataset-name vidore/mixed_syntheticdocqa_docscale_stress:test:25000 \
  --output benchmark-results/vidore-mixed-syntheticdocqa-docscale-stress-colqwen2-limit25000.npz
```

Measured with 256 fixed queries on the VAST RTX 4090 worker:

| docs | dense fp16 mean | dense P95 | bitmax int4 mean | binary_q40 mean | binary mean | binary speedup | dense NDCG@10 | int4 NDCG@10 | binary_q40 NDCG@10 | binary NDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 35492.90 ms | 35623.81 ms | 2414.87 ms | 648.45 ms | 637.72 ms | 55.66x | 0.377 | 0.379 | 0.372 | 0.372 |
| 10,000 | 71541.32 ms | 71748.70 ms | 4763.87 ms | 1278.96 ms | 1253.83 ms | 57.06x | 0.376 | 0.378 | 0.371 | 0.369 |
| 25,000 | 174426.51 ms | 174690.07 ms | 11828.15 ms | 3161.95 ms | 3122.47 ms | 55.86x | 0.374 | 0.376 | 0.371 | 0.363 |

Artifact: `benchmark-results/docscale-stress-5k-10k-25k-rich-summary.json`.
The 5k and 10k comparison files also include FAISS/cuVS mean-pool rows; the
25k full comparison hit FAISS GPU temporary-memory OOM, so the final 25k
artifact is dense plus bitmax core modes. This does not affect the dense-vs-
bitmax scaling result.
