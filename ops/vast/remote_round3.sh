#!/usr/bin/env bash
set -euo pipefail
cd /root/bitmax
mkdir -p benchmark-results
pip install --quiet --upgrade pip
pip install --quiet scikit-build-core pybind11 numpy pytest scipy
pip install --quiet -U torch --index-url https://download.pytorch.org/whl/cu126
BITMAX_BUILD_CUDA=1 pip install -e . --no-build-isolation --config-settings build-dir=/root/bitmax-build --quiet 2>&1 | tail -1
pytest -m cuda -q
python benchmarks/exp_qtile_sweep.py --output benchmark-results/qtile-sweep-plain.json
python benchmarks/exp_qtile_sweep.py --output benchmark-results/qtile-sweep-token-scale.json --token-scale
python benchmarks/run_retrieval.py --stage embeddings-smoke \
  --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --output benchmark-results/r3-smoke-gate.json --top-k 10 --repeat 1 --variants binary
python benchmarks/run_retrieval.py --stage embeddings-cuda-smoke \
  --input caches/vidore-docvqa-colqwen2-limit256.npz \
  --gate benchmark-results/r3-smoke-gate.json \
  --output benchmark-results/r3-headline-confirm-r5.json --top-k 10 --repeat 5 \
  --variants binary,binary_token_scale_fp16_cuda,binary_token_scale_u8_cuda,int4_int8q_dp4a
echo ROUND3_DONE
