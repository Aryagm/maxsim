# bitmax

`bitmax` is a Python SDK and kernel library for compressed late-interaction
multi-vector search. Query tokens stay `int8`/`float16`/`float32`; document
tokens can be stored as 1-bit signs, q40 centroid-calibrated signs, or signed
int4 values.

This is not a vector database, RAG framework, or embedding model. The v0.1 goal
is to provide the compressed MaxSim scoring and reranking layer that those
systems can call.

## Current CUDA Evidence

Measured artifacts are committed under `docs/benchmark_results/raw/`; generated
embedding caches are intentionally ignored.

The strongest current unique-corpus result is a mixed ViDoRe/SyntheticDocQA
slice with 4,882 unique documents and 256 measured queries on a VAST RTX 4090:

| implementation | fp32 doc reduction | P95 latency | speedup vs dense fp16 | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 2.00x | 34.32s | 1.00x | 0.383 | 0.378 |
| fast-plaid CUDA | 3.45x | 21.76s | 1.63x | 0.387 | 0.379 |
| bitmax binary CUDA | 32.00x | 627ms | 54.53x | 0.387 | 0.374 |
| bitmax binary_q40 CUDA | 32.00x | 640ms | 53.38x | 0.383 | 0.375 |
| bitmax int4 CUDA | 8.00x | 2.38s | 14.34x | 0.383 | 0.379 |

Artifact:
`docs/benchmark_results/raw/unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json`.

The largest current result is a fixed-query document-count stress sweep on the
same VAST RTX 4090:

| docs | dense fp16 P95 | binary P95 | binary speedup | dense NDCG@10 | binary_q40 NDCG@10 | int4 NDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 35.62s | 645ms | 55.66x | 0.377 | 0.372 | 0.379 |
| 10,000 | 71.75s | 1.26s | 57.06x | 0.376 | 0.371 | 0.378 |
| 25,000 | 174.69s | 3.14s | 55.86x | 0.374 | 0.371 | 0.376 |

Interpretation:

- `binary` is the aggressive mode: 32x fp32 document compression and the lowest
  latency, with a small quality loss on current ViDoRe-style sweeps.
- `binary_q40` keeps the 32x-ish storage profile while improving the large-doc
  NDCG result versus raw binary.
- `int4` is the conservative mode: 8x fp32 document compression with quality
  very close to dense fp16 in the tracked runs.

The 25k row is a distractor-crowding stress test built from repeated
non-positive real page embeddings. It validates scoring and memory scaling; it
is not a claim that the 25k corpus contains 25k unique pages. See
`docs/benchmarks.md` and `docs/benchmark_results/README.md` for full commands,
artifacts, and caveats.

## Install for development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## SDK Use

```python
import bitmax

corpus = bitmax.Corpus.from_embeddings(
    doc_ids=doc_ids,
    embeddings=doc_embeddings,
    offsets=doc_offsets,
    mode="binary_q40",
    metadata={"model": "vidore/colqwen2-v1.0-hf"},
)
corpus.save("docs.bitmax.npz")

reranker = bitmax.Reranker.load("docs.bitmax.npz", device="cuda")
results = reranker.search(query_embeddings, k=10)

reranked = reranker.rerank(
    query_embeddings,
    candidate_ids=["doc-17", "doc-03", "doc-91"],
    k=3,
)
```

For an end-to-end example using any precomputed multi-vector embeddings:

```bash
python -m examples.rag_pipeline_sdk \
  --embeddings benchmark-results/vidore-docvqa-colqwen2-limit256.npz \
  --corpus benchmark-results/docvqa.binary_q40.bitmax.npz \
  --mode binary_q40 \
  --device cuda \
  --query-index 0 \
  --k 5
```

Modes:

- `binary`: fastest, 32x fp32 document compression.
- `binary_q40`: experimental 32x-ish accuracy mode using q40 centroid calibration.
- `int4`: experimental accuracy-first mode, 8x fp32 compression.

Lower-level kernels remain available when you need direct packed scoring:

```python
packed = bitmax.pack_signs(doc_embeddings, doc_offsets)
scores = bitmax.maxsim(query_embeddings, packed)
top_scores, top_indices = bitmax.topk_maxsim(query_embeddings, packed, k=10)
```

## API contract

- Query shapes: `[query_tokens, dim]` or `[batch, query_tokens, dim]`.
- Query dtypes: `int8`, `float16`, or `float32`; output scores are `float32`.
- Packed signs use little-endian bits inside each byte: `1 = +1`, `0 = -1`.
- `dim` must be divisible by 8. The current native CPU kernel is an exact
  scalar byte-LUT implementation; CUDA is optional with `BITMAX_BUILD_CUDA=1`.

## Benchmarks

Run the cheap signal-first benchmark ladder before any expensive GPU work:

```bash
python benchmarks/run_synthetic.py --stage stage0
python benchmarks/run_synthetic.py --stage cpu-smoke
```

Benchmark JSON uses schema version 2 and includes:

- `python_reference` correctness rows;
- `torch_fp16_baseline` and `torch_int8_baseline` rows;
- `bitmax_native` or `bitmax_cuda` rows with speedups vs the torch-style
  baselines;
- packed-document memory compression vs dense fp16/fp32 storage.

If PyTorch is installed, the baseline rows use PyTorch. Otherwise they use a
NumPy implementation of the same vectorized dense MaxSim formulas and mark
`baseline_backend` as `numpy_torch_equivalent`.

CUDA and larger VAST runs are gated by the earlier JSON results:
`cuda-smoke` unlocks `cuda-sweep`, and `cuda-sweep` unlocks `vast-large`.
Benchmark tables in this README should only contain measured numbers from
`benchmark-results/`.

Retrieval-level benchmarks consume multi-vector embedding `.npz` files and qrels
without building an index:

```bash
python benchmarks/run_retrieval.py --stage fixture-smoke
python benchmarks/build_vidore_embeddings.py \
  --dataset vidore/docvqa_test_subsampled \
  --limit 16 \
  --model vidore/colqwen2-v1.0-hf \
  --output benchmark-results/vidore-docvqa-colqwen2.npz
python benchmarks/run_retrieval.py \
  --stage embeddings-smoke \
  --input benchmark-results/vidore-docvqa-colqwen2.npz
```

Those rows report recall/MRR/NDCG, top-k agreement with dense fp16 MaxSim,
latency, speedup, and document-memory compression.

To inspect or rerun benchmark plans:

```bash
python -m benchmarks.reproduce --suite smoke
python -m benchmarks.reproduce --suite smoke --execute --refresh-ledger
python -m benchmarks.reproduce --suite cuda-unique-5k
python -m benchmarks.reproduce --suite cuda-docscale
```

`--execute` runs the plan; without it the command writes a dry-run JSON plan
with git SHA, environment, GPU info, commands, and expected artifacts.

Experimental Pareto variants can be measured with `--variants all`. Those rows
include ternary documents, per-token scale, grouped scale, int4 documents,
calibrated threshold references, and per-dimension centroid-calibrated binary
docs. Most are benchmark probes, not stable public kernels. The centroid and
int4 paths are exposed under `bitmax.experimental`:

```python
from bitmax.experimental import (
    dim_centroid_maxsim,
    fit_dim_centroid_calibration,
    int4_maxsim,
    int4_to_device,
    pack_dim_centroid_signs,
    pack_int4_symmetric,
)

calibration = fit_dim_centroid_calibration(docs)
packed, calibration = pack_dim_centroid_signs(docs, doc_offsets, calibration=calibration)
scores = dim_centroid_maxsim(query, packed, calibration)

packed_i4 = pack_int4_symmetric(docs, doc_offsets)
scores_i4 = int4_maxsim(query, int4_to_device(packed_i4), device="cuda")
```

CUDA top-k kernel experiments are available on CUDA workers:

```bash
python -m benchmarks.run_cuda_topk --stage lut-sweep
python -m benchmarks.run_cuda_topk --stage blog-shape
```

On a project-owned VAST RTX 4090, the dim128 int8-query LUT path measured
`0.463 ms` median latency on the blog-style 33 x 1000 x 786 x 128 top-k shape
with 12,576 bytes/doc, versus `0.723 ms` for torch fp32 dense top-k on the same
worker.

The SDK CUDA demo on `vidore/docvqa_test_subsampled:test:256` with ColQwen2
embeddings measured:

| implementation | fp32 doc reduction | latency | speedup vs dense fp16 | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 2.00x | 1906.08 ms | 1.00x | 0.777 | 0.660 |
| SDK binary CUDA | 32.00x | 58.79 ms | 32.42x | 0.754 | 0.649 |
| SDK binary_q40 CUDA | 31.98x | 64.35 ms | 29.62x | 0.762 | 0.652 |
| SDK int4 CUDA | 8.00x | 144.19 ms | 13.22x | 0.773 | 0.658 |

Artifact: `benchmark-results/sdk-demo-local-search-limit256-cuda.json`.

Against open-source baselines on the same slice:

| implementation | fp32 doc reduction | latency | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 2.00x | 1926.10 ms | 0.777 | 0.660 |
| FAISS GPU mean-pool flat IP | 752.21x | 0.75 ms | 0.414 | 0.290 |
| cuVS GPU mean-pool flat IP | 752.21x | 0.98 ms | 0.414 | 0.290 |
| FAISS GPU token candidates + dense rerank | 0.67x | 2884.28 ms | 0.777 | 0.660 |
| Qdrant in-memory multivector | 1.00x | 33660.60 ms | 0.777 | 0.660 |
| fast-plaid CUDA | 3.37x | 1856.76 ms | 0.766 | 0.660 |
| SDK binary_q40 CUDA | 31.98x | 64.73 ms | 0.762 | 0.652 |
| SDK int4 CUDA | 8.00x | 144.24 ms | 0.773 | 0.658 |

Artifact: `benchmark-results/open-source-comparison-expanded-limit256-cuda.json`.

Across four ViDoRe datasets at limit64 with frozen ColQwen2 embeddings:

| implementation | datasets | fp32 doc reduction | mean latency | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 4 | 2.00x | 110.94 ms | 0.949 | 0.885 |
| fast-plaid CUDA | 4 | 3.22x | 185.65 ms | 0.945 | 0.882 |
| SDK int4 CUDA | 4 | 8.00x | 33.05 ms | 0.949 | 0.879 |
| SDK binary CUDA | 4 | 32.00x | 9.19 ms | 0.941 | 0.865 |
| FAISS pooled GPU | 4 | 707.80x | 0.27 ms | 0.809 | 0.650 |

Artifact: `benchmark-results/multidataset-limit64-slow-summary.json`.

## VAST

VAST helpers live under `ops/vast/` and are also exposed as `bitmax-vast` after
installation. They enforce a ledger-based cleanup rule: destroy only instances
recorded in `.vast/bitmax-instances.jsonl` whose live label still starts with
`bitmax-v0-`.
