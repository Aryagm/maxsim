#!/usr/bin/env bash
# Parallel cache-build worker. Usage: remote_worker_builds.sh {a|b|c|d|e}
set -uo pipefail
ROLE="${1:?role a-e}"
cd /root/bitmax
mkdir -p benchmark-results

pip install --quiet --upgrade pip
pip uninstall --quiet -y torchaudio 2>/dev/null || true
pip install --quiet -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --quiet -U nvidia-nccl-cu12
pip install --quiet "transformers==5.12.1" datasets accelerate pillow numpy
python -c "import torch, transformers; transformers.ColQwen2ForRetrieval; print('encoder stack ok:', torch.__version__, transformers.__version__)" \
  || { echo WORKER_FAILED_ENCODER_STACK; exit 1; }

b() {
  local dataset="$1" split="$2" limit="$3" out="$4" extra="${5:-}"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  python -m benchmarks.build_vidore_embeddings --dataset "$dataset" --split "$split" --limit "$limit" \
    --model vidore/colqwen2-v1.0-hf --output "$out" $extra && echo "BUILT $out" || echo "BUILD_FAILED $out"
}

bp() {
  local dataset="$1" out="$2"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  python -m benchmarks.build_vidore_embeddings --dataset "$dataset" --split test --limit 256 \
    --model vidore/colpali-v1.3-hf --output "$out" \
  || python -m benchmarks.build_vidore_embeddings --dataset "$dataset" --split test --limit 256 \
    --model vidore/colpali-v1.2-hf --output "$out" \
  && echo "BUILT $out" || echo "BUILD_FAILED $out"
}

case "$ROLE" in
  a)
    b vidore/shiftproject_test test 1000 benchmark-results/vidore-syntheticdocqa-shift-colqwen2-limit1000.npz
    b vidore/infovqa_train train 1200 benchmark-results/vidore-infovqa-train-colqwen2-limit1200.npz --streaming
    bp vidore/docvqa_test_subsampled benchmark-results/vidore-docvqa-colpali-limit256.npz
    bp vidore/infovqa_test_subsampled benchmark-results/vidore-infovqa-colpali-limit256.npz
    bp vidore/arxivqa_test_subsampled benchmark-results/vidore-arxivqa-colpali-limit256.npz
    bp vidore/tabfquad_test_subsampled benchmark-results/vidore-tabfquad-colpali-limit256.npz
    ;;
  b)
    b vidore/arxivqa_train train 1200 benchmark-results/vidore-arxivqa-train-colqwen2-limit1200.npz --streaming
    b vidore/tatdqa_train train 1200 benchmark-results/vidore-tatdqa-train-colqwen2-limit1200.npz --streaming
    b vidore/tabfquad_test_subsampled test 500 benchmark-results/vidore-tabfquad-test-colqwen2-limit500.npz
    ;;
  c)
    b vidore/docvqa_train train 3000 benchmark-results/vidore-docvqa-train-colqwen2-limit3000.npz --streaming
    ;;
  d)
    b vidore/syntheticDocQA_energy_train train 3000 benchmark-results/vidore-syntheticdocqa-energy-train-colqwen2-limit3000.npz --streaming
    ;;
  e)
    b vidore/syntheticDocQA_artificial_intelligence_test test 1000 benchmark-results/vidore-syntheticdocqa-ai-colqwen2-limit1000.npz
    b vidore/syntheticDocQA_energy_test test 1000 benchmark-results/vidore-syntheticdocqa-energy-colqwen2-limit1000.npz
    b vidore/syntheticDocQA_government_reports_test test 1000 benchmark-results/vidore-syntheticdocqa-government-colqwen2-limit1000.npz
    ;;
esac
echo "WORKER_${ROLE}_DONE"
