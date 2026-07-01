# Production Baseline Comparisons

`bitmax` should be compared as a compressed late-interaction reranking/scoring
layer, not as a full database. Baseline labels in benchmark artifacts should
make that distinction explicit.

## Baseline Classes

| baseline | what it measures | fair interpretation |
| --- | --- | --- |
| dense fp16 MaxSim | exact late-interaction scoring over full token embeddings | quality/latency reference, large memory footprint |
| FAISS/cuVS mean-pool flat IP | single-vector search over pooled embeddings | very fast/tiny, but not late interaction |
| FAISS token candidates + dense rerank | token-vector candidate generation then dense MaxSim rerank | stores full vectors and can recover dense quality, but is slower in tracked slices |
| Qdrant multivector | production-style exact multivector API path | useful API comparison, not a custom CUDA kernel comparison |
| fast-plaid | compressed PLAID-style late-interaction search | closer algorithmic family; compare quality, index build, storage, and latency separately |
| bitmax binary/q40/int4 | compressed document-token scoring/reranking | kernel/SDK layer that another RAG/search system can call |

## Required Metrics

Each comparison should report:

- P50/P95/P99 latency and best latency.
- bytes/doc or compression vs fp32.
- recall@1, recall@5, recall@10.
- MRR@10, NDCG@5, NDCG@10.
- quality deltas versus dense fp16.
- query count, document count, document-token count.
- whether the corpus is unique-document, mixed, or distractor-expanded stress.

## Current Evidence

Tracked artifacts live in `docs/benchmark_results/raw/`.

- Multi-dataset ViDoRe limit64: `multidataset-limit64-rich-summary.json`
- DocVQA scaling 64/256: `multidataset-docvqa-scaling-rich-summary.json`
- Mixed real doc scaling 1k/2k/3k: `mixed-syntheticdocqa-docscale-rich-summary.json`
- Unique mixed 10,171-doc CUDA comparison:
  `unique-public-10k-rich/vidore-mixed-public-unique-colqwen2-limit10000-comparison.json`
- Historical unique mixed 4,882-doc CUDA comparison:
  `unique-mixed-syntheticdocqa-5k-rich/vidore-mixed-syntheticdocqa-colqwen2-limit5000-comparison.json`
- 5k/10k/25k stress scaling: `docscale-stress-5k-10k-25k-rich-summary.json`

On the unique 10,171-doc slice, `bitmax_binary` keeps 32x fp32 document
compression, measures 1.29 s P95 latency, and lands at 0.491 NDCG@10 versus
dense fp16 exact MaxSim at 77.17 s P95 and 0.510 NDCG@10. `bitmax_int4`
keeps 8x fp32 document compression, measures 4.86 s P95, and lands at 0.503
NDCG@10. Mean-pooled FAISS/cuVS are much faster in absolute latency, but drop
to 0.036 NDCG@10 because they are single-vector baselines rather than
late-interaction scorers.

## Caveats

The 25k result is still a fixed-query distractor stress test. It is useful for
scoring latency, memory pressure, and hard-negative sensitivity, but a future
publication-grade 25k claim should use a fully unique 25k corpus.
