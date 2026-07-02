#!/usr/bin/env bash
# Phase 1 on the benchmark instance: embedding caches, same-hardware baseline,
# E1 token-scale rows, and the dim128-gate microbench evidence.
set -euo pipefail
cd /root/bitmax
mkdir -p caches benchmark-results

build() {
  local dataset="$1" limit="$2" out="$3"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  python benchmarks/build_vidore_embeddings.py \
    --dataset "$dataset" --split test --limit "$limit" \
    --model vidore/colqwen2-v1.0-hf --batch-size 1 --output "$out"
}

build vidore/docvqa_test_subsampled 256 caches/vidore-docvqa-colqwen2-limit256.npz
build vidore/docvqa_test_subsampled 64 caches/vidore-docvqa-colqwen2-limit64.npz
build vidore/infovqa_test_subsampled 64 caches/vidore-infovqa-colqwen2-limit64.npz
build vidore/arxivqa_test_subsampled 64 caches/vidore-arxivqa-colqwen2-limit64.npz
build vidore/tabfquad_test_subsampled 64 caches/vidore-tabfquad-colqwen2-limit64.npz

VARIANTS="binary,binary_doc_scale,binary_token_scale,binary_token_scale_cuda,binary_token_scale_fp16_cuda,int4_symmetric_per_tensor,binary_dim_centroid_zero,binary_dim_centroid_q40"

python benchmarks/run_retrieval.py --stage embeddings-smoke \
  --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --output benchmark-results/r256-smoke-gate.json --top-k 10 --repeat 1 --variants binary

python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke \
  --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --gate benchmark-results/r256-smoke-gate.json \
  --output benchmark-results/r256-pareto-v2-r5.json --top-k 10 --repeat 5 --variants "$VARIANTS"

for ds in docvqa infovqa arxivqa tabfquad; do
  python benchmarks/run_retrieval.py --stage embeddings-smoke \
    --input "caches/vidore-${ds}-colqwen2-limit64.npz" \
    --output "benchmark-results/r64-${ds}-smoke-gate.json" --top-k 10 --repeat 1 --variants binary
  python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke \
    --input "caches/vidore-${ds}-colqwen2-limit64.npz" \
    --gate "benchmark-results/r64-${ds}-smoke-gate.json" \
    --output "benchmark-results/r64-${ds}-pareto-v2-r5.json" --top-k 10 --repeat 5 --variants "$VARIANTS"
done

python benchmarks/run_cuda_topk.py --stage lut-sweep --output benchmark-results/cuda-lut-sweep-pareto-v2.json

python benchmarks/exp_int8q_int4_sim.py --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --output benchmark-results/int8q-int4-sim-limit256.json
python benchmarks/exp_int8q_int4_sim.py --input caches/vidore-docvqa-colqwen2-limit64.npz \
  --output benchmark-results/int8q-int4-sim-limit64.json

echo PHASE1_DONE
