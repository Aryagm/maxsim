# bitmax

`bitmax` is a small kernel library for exact asymmetric binary MaxSim scoring:
query tokens stay `int8`/`float16`/`float32`, stored document tokens are packed to
1-bit signs.

This is not a vector database, RAG framework, or embedding model. The v0.1 goal is
to provide the scoring primitive that those systems can call.

## Install for development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Minimal use

```python
import numpy as np
import bitmax

docs = np.random.randn(128, 128).astype("float32")
packed = bitmax.pack_signs(docs)

query = np.random.randint(-8, 8, size=(16, 128), dtype=np.int8)
scores = bitmax.maxsim(query, packed)
top_scores, top_indices = bitmax.topk_maxsim(query, packed, k=10)
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

## VAST

VAST helpers live under `ops/vast/` and are also exposed as `bitmax-vast` after
installation. They enforce a ledger-based cleanup rule: destroy only instances
recorded in `.vast/bitmax-instances.jsonl` whose live label still starts with
`bitmax-v0-`.
