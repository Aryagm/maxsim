#!/usr/bin/env bash
# One-time setup for a bitmax benchmark instance (pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel).
set -euo pipefail

cd /root/bitmax

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

pip install --quiet --upgrade pip
pip install --quiet scikit-build-core pybind11 numpy pytest
BITMAX_BUILD_CUDA=1 pip install -e . --no-build-isolation --config-settings build-dir=/root/bitmax-build -v 2>&1 | tail -5

python -c "from bitmax import _bitmax_cuda; print('cuda ext OK:', _bitmax_cuda.__doc__)"
pytest -m cuda -q

pip install --quiet transformers datasets accelerate pillow
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0))
EOF
