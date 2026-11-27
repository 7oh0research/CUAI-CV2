#!/usr/bin/env bash
set -euo pipefail
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$drv" -lt 570 ]; then echo "[STOP] driver $drv < 570: RTX 5090 needs CUDA 12.8+"; exit 1; fi
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python - <<'EOF'
import torch, transformers
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16); print("matmul ok", (x @ x).float().abs().mean().item() > 0)
EOF
