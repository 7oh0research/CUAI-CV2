#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${CUAI_ROOT:-/workspace/CUAI-CV2}"
CONDA="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
[[ -x "$CONDA" ]] || { echo "[ERROR] Conda not found: $CONDA" >&2; exit 1; }
cd "$ROOT"
nvidia-smi
for env in h1_llava h1_fastv h1_sparsevlm h1_visionzip; do
  echo "===== $env ====="
  "$CONDA" run -n "$env" python -c "import torch,torchvision,numpy,transformers; print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('numpy',numpy.__version__); print('transformers',transformers.__version__); print('cuda',torch.version.cuda); print('gpu',torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert 'sm_120' in torch.cuda.get_arch_list()"
done
