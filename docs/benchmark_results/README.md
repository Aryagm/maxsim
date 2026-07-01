# Benchmark Result Ledger

This directory tracks numerical benchmark artifacts that should move with the
repo. The raw artifacts are mirrored from `benchmark-results/` into
`docs/benchmark_results/raw/` because `benchmark-results/` itself is ignored to
avoid accidentally committing generated embedding caches.

Last refreshed: 2026-06-30.

## What Is Tracked

- Raw JSON benchmark outputs and summary files.
- Generated benchmark plots that are small enough to keep in git.
- The exact result files referenced by `docs/benchmarks.md`.

Important summary artifacts:

- `raw/multidataset-limit64-rich-summary.json`
- `raw/multidataset-docvqa-scaling-rich-summary.json`
- `raw/mixed-syntheticdocqa-docscale-rich-summary.json`
- `raw/unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json`
- `raw/unique-public-10k-rich/vidore-mixed-public-unique-colqwen2-limit10000-comparison.json`
- `raw/docscale-stress-5k-10k-25k-rich-summary.json`
- `raw/open-source-comparison-expanded-limit256-cuda.json`
- `raw/sdk-demo-local-search-limit256-cuda.json`

## What Is Not Tracked

- `.npz` embedding caches and mixed/stress corpora. These are generated inputs
  and can be hundreds of MB to many GB each.
- VAST ledgers under `.vast/`, which can contain instance/account-specific
  operational state.

## Refresh Command

After running benchmarks, refresh the committed numerical artifact mirror with:

```bash
mkdir -p docs/benchmark_results/raw
rsync -av \
  --include='*/' \
  --include='*.json' \
  --include='*.png' \
  --exclude='*' \
  benchmark-results/ docs/benchmark_results/raw/
```

Run tests before committing the refreshed ledger:

```bash
.venv/bin/python -m pytest -q
```

## Headline Results

The current strongest unique-corpus run uses a mixed public ViDoRe/SyntheticDocQA
embedding cache with 10,171 unique documents and 256 measured queries on a VAST
RTX 4090.

| implementation | fp32 doc reduction | P95 latency | speedup vs dense fp16 | recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense fp16 CUDA | 2.00x | 77.17s | 1.00x | 0.602 | 0.510 |
| fast-plaid CUDA | 3.45x | 24.84s | 3.26x | 0.609 | 0.512 |
| bitmax binary CUDA | 32.00x | 1.29s | 59.59x | 0.602 | 0.491 |
| bitmax binary_q40 CUDA | 32.00x | 1.31s | 58.76x | 0.582 | 0.475 |
| bitmax int4 CUDA | 8.00x | 4.86s | 15.79x | 0.617 | 0.503 |

The current large fixed-query doc-scale stress run uses 256 fixed queries on a
VAST RTX 4090. It shows binary scoring remains around 56x faster than dense
fp16 from 5k to 25k docs, with small NDCG@10 loss.

| docs | dense P95 | binary P95 | binary speedup | dense NDCG@10 | binary NDCG@10 | binary_q40 NDCG@10 | int4 NDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 35.62s | 645ms | 55.66x | 0.377 | 0.372 | 0.372 | 0.379 |
| 10,000 | 71.75s | 1.26s | 57.06x | 0.376 | 0.369 | 0.371 | 0.378 |
| 25,000 | 174.69s | 3.14s | 55.86x | 0.374 | 0.363 | 0.371 | 0.376 |

See `docs/benchmarks.md` for the full benchmark ladder, command examples, and
interpretation notes.
