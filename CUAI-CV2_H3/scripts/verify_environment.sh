#!/usr/bin/env bash
set -Eeuo pipefail
if ! command -v conda >/dev/null 2>&1; then export PATH="$HOME/miniforge3/bin:$PATH"; fi
for ENV in h3_vanilla h3_fastv h3_sparsevlm h3_visionzip; do
  echo "===== $ENV ====="
  conda run -n "$ENV" python - <<'PY'
import torch, torchvision, numpy
print('Python packages:', torch.__version__, torchvision.__version__, numpy.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit('CUDA is unavailable')
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
x=torch.randn((1024,1024),device='cuda'); y=x@x
print('CUDA matmul:', tuple(y.shape), 'OK')
PY
done
