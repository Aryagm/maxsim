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
documents carry useful signal, but it currently applies the vector scale outside
the fused CUDA top-k path.
