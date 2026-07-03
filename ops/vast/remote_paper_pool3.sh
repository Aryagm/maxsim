#!/usr/bin/env bash
# Follow-up to remote_paper_suite.sh: pool_factor=3 rows on the same caches.
set -uo pipefail
cd /root/bitmax
for cache in benchmark-results/vidore-docvqa-test-colqwen2-limit500.npz \
             benchmark-results/vidore-infovqa-test-colqwen2-limit500.npz \
             benchmark-results/vidore-arxivqa-test-colqwen2-limit500.npz \
             benchmark-results/vidore-tabfquad-test-colqwen2-limit500.npz \
             benchmark-results/vidore-tatdqa-test-colqwen2-limit1663.npz \
             benchmark-results/vidore-syntheticdocqa-ai-colqwen2-limit1000.npz \
             benchmark-results/vidore-syntheticdocqa-energy-colqwen2-limit1000.npz \
             benchmark-results/vidore-syntheticdocqa-government-colqwen2-limit1000.npz \
             benchmark-results/vidore-syntheticdocqa-healthcare-colqwen2-limit1000.npz \
             benchmark-results/vidore-syntheticdocqa-shift-colqwen2-limit1000.npz; do
  [ -f "$cache" ] || { echo "MISSING $cache"; continue; }
  slug=$(basename "$cache" .npz)
  python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke --input "$cache" \
    --gate "benchmark-results/paper-gate-${slug}.json" \
    --output "benchmark-results/paper-pool3-${slug}-r3.json" --top-k 10 --repeat 3 --variants pool3_binary \
    || echo "POOL3_DATASET_FAILED $slug"
done
python -m benchmarks.compare_open_source \
  --input benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz \
  --output benchmark-results/paper-unique-10k-pool3.json \
  --device cuda --repeat 3 --metric-ks 1,5,10 --limit-queries 256 --allow-unavailable \
  --implementations bitmax_pooled_binary3 || echo "POOL3_10K_FAILED"
python -m benchmarks.compare_open_source \
  --input benchmark-results/paper-docscale-25k.npz \
  --output benchmark-results/paper-docscale-25k-pool3.json \
  --device cuda --repeat 3 --metric-ks 1,5,10 --limit-queries 64 --allow-unavailable \
  --implementations bitmax_pooled_binary3 || echo "POOL3_DOCSCALE_FAILED"
echo POOL3_DONE
