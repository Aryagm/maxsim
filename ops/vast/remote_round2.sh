#!/usr/bin/env bash
# Round 2: validate SDK token-scale mode, u8 scales, and dp4a binary kernels.
set -euo pipefail
cd /root/bitmax
mkdir -p caches benchmark-results

pip install --quiet --upgrade pip
pip install --quiet scikit-build-core pybind11 numpy pytest scipy
pip install --quiet -U torch --index-url https://download.pytorch.org/whl/cu126
MAXSIM_BUILD_CUDA=1 pip install -e . --no-build-isolation --config-settings build-dir=/root/bitmax-build --quiet 2>&1 | tail -1
python -c "from maxsim import _maxsim_cuda; print('ext ok, gate', _maxsim_cuda.get_dim128_unrolled_min_avg_tokens())"
pytest -m cuda -q

pip install --quiet transformers datasets accelerate pillow

build() {
  local dataset="$1" limit="$2" out="$3"
  if [ -f "$out" ]; then echo "SKIP $out"; return; fi
  python benchmarks/build_vidore_embeddings.py \
    --dataset "$dataset" --split test --limit "$limit" \
    --model vidore/colqwen2-v1.0-hf --batch-size 1 --output "$out"
}
build vidore/docvqa_test_subsampled 256 caches/vidore-docvqa-colqwen2-limit256.npz
build vidore/docvqa_test_subsampled 64 caches/vidore-docvqa-colqwen2-limit64.npz

python benchmarks/run_retrieval.py --stage embeddings-smoke \
  --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --output benchmark-results/r2-smoke-gate.json --top-k 10 --repeat 1 --variants binary

python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke \
  --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --gate benchmark-results/r2-smoke-gate.json \
  --output benchmark-results/r2-round2-limit256-r5.json --top-k 10 --repeat 5 \
  --variants binary,binary_token_scale_fp16_cuda,binary_token_scale_u8_cuda,binary_int8q_dp4a,binary_token_scale_fp16_int8q_dp4a,int4_int8q_dp4a

python benchmarks/run_retrieval.py --stage embeddings-smoke \
  --input caches/vidore-docvqa-colqwen2-limit64.npz \
  --output benchmark-results/r2-smoke-gate-64.json --top-k 10 --repeat 1 --variants binary
python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke \
  --input caches/vidore-docvqa-colqwen2-limit64.npz \
  --gate benchmark-results/r2-smoke-gate-64.json \
  --output benchmark-results/r2-round2-limit64-r5.json --top-k 10 --repeat 5 \
  --variants binary,binary_token_scale_fp16_cuda,binary_token_scale_u8_cuda,binary_int8q_dp4a,binary_token_scale_fp16_int8q_dp4a,int4_int8q_dp4a

echo ROUND2_DONE
