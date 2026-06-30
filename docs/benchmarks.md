# Benchmark Ladder

Run benchmarks in order. Do not start larger VAST runs until the earlier gate
JSON exists and reports `"gate_passed": true`.

```bash
python benchmarks/run_synthetic.py --stage stage0 --output benchmark-results/stage0.json
python benchmarks/run_synthetic.py --stage cpu-smoke --output benchmark-results/cpu-smoke.json
python benchmarks/run_synthetic.py --stage cuda-smoke --output benchmark-results/cuda-smoke.json
python benchmarks/run_synthetic.py --stage vast-large \
  --gate benchmark-results/cuda-smoke.json \
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

The `torch_fp16_baseline` and `torch_int8_baseline` rows include:

- `baseline_backend`: `torch` when PyTorch is installed, otherwise
  `numpy_torch_equivalent`.
- `formula`: `dense_fp16_maxsim` or `dense_int8_doc_maxsim`.
