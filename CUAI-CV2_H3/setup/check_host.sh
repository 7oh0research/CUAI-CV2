#!/usr/bin/env bash
set -Eeuo pipefail
command -v nvidia-smi >/dev/null || { echo '[ERROR] nvidia-smi not found. Use a Vast.ai GPU instance with NVIDIA driver passthrough.'; exit 1; }
nvidia-smi
python3 - <<'PY'
import platform, shutil
print('Architecture:', platform.machine())
print('Free disk (bytes):', shutil.disk_usage('/workspace' if __import__('os').path.exists('/workspace') else '/').free)
if platform.machine() not in ('x86_64','amd64'):
    raise SystemExit('This setup targets Linux x86_64.')
PY
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$GPU_NAME" == *"5090"* ]] || echo "[WARN] Detected GPU: $GPU_NAME (target hardware is RTX 5090; continuing)."
