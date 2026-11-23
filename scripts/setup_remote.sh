#!/usr/bin/env bash
# One-shot setup on a rented GPU box (Vast.ai, RTX 5090 / Blackwell sm_120).
#   git clone https://github.com/7oh0research/CUAI-CV2.git && cd CUAI-CV2 && bash scripts/setup_remote.sh
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# Blackwell needs a CUDA 12.8+ torch build; skip if the image already ships one (preflight below verifies).
if ! python -c "import torch,sys; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
  pip install torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install -e ".[infer,dev]"
python scripts/profile_efficiency.py --preflight
pytest -q tests/test_efficiency.py
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv
echo "ready: tmux new -s h1 'source .venv/bin/activate && python scripts/profile_efficiency.py --config configs/h1.yaml'"
