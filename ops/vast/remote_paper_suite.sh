#!/usr/bin/env bash
# Paper benchmark suite: full ViDoRe caches, per-dataset tier tables,
# 10k unique-corpus comparison vs OSS baselines, 25k docscale, ColPali subset.
set -uo pipefail
cd /root/bitmax
mkdir -p benchmark-results
stage() { echo "=== STAGE:$1 $(date -u +%H:%M:%S) ==="; }

stage setup
pip install --quiet --upgrade pip
pip install --quiet scikit-build-core pybind11 numpy pytest scipy
# torchaudio in the image is built for torch 2.4 and poisons the transformers
# import chain after the torch upgrade; nothing here uses it. torchvision is
# upgraded in lockstep with torch instead (transformers image utils want it).
pip uninstall --quiet -y torchaudio 2>/dev/null || true
pip install --quiet -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --quiet -U nvidia-nccl-cu12
# transformers 5.13.0 force-downgrades torch; 5.12.1 is the validated combo
# and must install AFTER torch so the cu126 build is left untouched.
pip install --quiet "transformers==5.12.1" datasets accelerate pillow
MAXSIM_BUILD_CUDA=1 pip install -e . --no-build-isolation --config-settings build-dir=/root/bitmax-build --quiet 2>&1 | tail -1
pytest -m cuda -q || { echo SUITE_FAILED_TESTS; exit 1; }
python -c "import torch, transformers; transformers.ColQwen2ForRetrieval; print('encoder stack ok:', torch.__version__, transformers.__version__)"   || { echo SUITE_FAILED_ENCODER_STACK; exit 1; }
pip install --quiet faiss-gpu==1.14.3 fast-plaid 2>&1 | tail -1 || echo "oss deps partial (allow-unavailable)"

stage build-caches
python -m benchmarks.reproduce --suite build-unique-caches --execute || { echo SUITE_FAILED_CACHES; exit 1; }
[ -f benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz ] || { echo SUITE_FAILED_CACHES_MISSING; exit 1; }
python benchmarks/build_vidore_embeddings.py --dataset vidore/tabfquad_test_subsampled --split test --limit 500 \
  --model vidore/colqwen2-v1.0-hf --batch-size 1 --output benchmark-results/vidore-tabfquad-test-colqwen2-limit500.npz

stage per-dataset
VARIANTS="binary,binary_token_scale_fp16_cuda,binary_token_scale_u4_cuda,int4_int8q_dp4a,pool2_binary"
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
  python benchmarks/run_retrieval.py --stage embeddings-smoke --input "$cache" \
    --output "benchmark-results/paper-gate-${slug}.json" --top-k 10 --repeat 1 --variants binary && \
  python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke --input "$cache" \
    --gate "benchmark-results/paper-gate-${slug}.json" \
    --output "benchmark-results/paper-${slug}-r3.json" --top-k 10 --repeat 3 --variants "$VARIANTS" \
    || echo "DATASET_FAILED $slug"
done

stage 10k-comparison
python -m benchmarks.compare_open_source \
  --input benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz \
  --output benchmark-results/paper-unique-10k-comparison.json \
  --device cuda --repeat 3 --metric-ks 1,5,10 --limit-queries 256 --allow-unavailable \
  --implementations dense_fp16,faiss_pooled,fast_plaid,bitmax_binary,bitmax_binary_token_scale,bitmax_binary_token_scale_u4,bitmax_int4,bitmax_int4_dp4a,bitmax_pooled_binary \
  || echo "STAGE_10K_FAILED"

stage docscale-25k
python -m benchmarks.build_docscale_stress \
  --input benchmark-results/vidore-mixed-public-unique-colqwen2-limit10000.npz \
  --target-docs 25000 --query-limit 64 \
  --output benchmark-results/paper-docscale-25k.npz && \
python -m benchmarks.compare_open_source \
  --input benchmark-results/paper-docscale-25k.npz \
  --output benchmark-results/paper-docscale-25k-comparison.json \
  --device cuda --repeat 3 --metric-ks 1,5,10 --limit-queries 64 --allow-unavailable \
  --implementations dense_fp16,bitmax_binary,bitmax_binary_token_scale,bitmax_binary_token_scale_u4,bitmax_pooled_binary \
  || echo "STAGE_DOCSCALE_FAILED"

stage colpali
for ds in docvqa infovqa arxivqa tabfquad; do
  python benchmarks/build_vidore_embeddings.py --dataset "vidore/${ds}_test_subsampled" --split test --limit 256 \
    --model vidore/colpali-v1.3-hf --batch-size 1 --output "benchmark-results/vidore-${ds}-colpali-limit256.npz" \
  || python benchmarks/build_vidore_embeddings.py --dataset "vidore/${ds}_test_subsampled" --split test --limit 256 \
    --model vidore/colpali-v1.2-hf --batch-size 1 --output "benchmark-results/vidore-${ds}-colpali-limit256.npz" \
  || { echo "COLPALI_BUILD_FAILED $ds"; continue; }
  python benchmarks/run_retrieval.py --stage embeddings-smoke --input "benchmark-results/vidore-${ds}-colpali-limit256.npz" \
    --output "benchmark-results/paper-gate-colpali-${ds}.json" --top-k 10 --repeat 1 --variants binary && \
  python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke --input "benchmark-results/vidore-${ds}-colpali-limit256.npz" \
    --gate "benchmark-results/paper-gate-colpali-${ds}.json" \
    --output "benchmark-results/paper-colpali-${ds}-r3.json" --top-k 10 --repeat 3 --variants "$VARIANTS" \
    || echo "COLPALI_DATASET_FAILED $ds"
done

echo PAPER_SUITE_DONE
