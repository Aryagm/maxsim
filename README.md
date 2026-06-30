# bitmax

`bitmax` is a Python SDK and kernel library for compressed late-interaction
multi-vector search. Query tokens stay `int8`/`float16`/`float32`; document
tokens can be stored as 1-bit signs, q40 centroid-calibrated signs, or signed
int4 values.

This is not a vector database, RAG framework, or embedding model. The v0.1 goal
is to provide the compressed MaxSim scoring and reranking layer that those
systems can call.

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

Against FAISS GPU on the same slice:

| implementation | fp32 doc reduction | latency | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| FAISS GPU mean-pool flat IP | 752.21x | 0.74 ms | 0.414 | 0.290 |
| FAISS GPU token candidates + dense rerank | 0.67x | 2784.23 ms | 0.777 | 0.660 |
| SDK binary_q40 CUDA | 31.98x | 64.24 ms | 0.762 | 0.652 |
| SDK int4 CUDA | 8.00x | 144.14 ms | 0.773 | 0.658 |

Artifact: `benchmark-results/open-source-comparison-limit256-cuda.json`.

## VAST

VAST helpers live under `ops/vast/` and are also exposed as `bitmax-vast` after
installation. They enforce a ledger-based cleanup rule: destroy only instances
recorded in `.vast/bitmax-instances.jsonl` whose live label still starts with
`bitmax-v0-`.
