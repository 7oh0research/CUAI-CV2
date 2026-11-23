#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for cmd in git curl; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $cmd" >&2; exit 1; }
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. Install a working NVIDIA driver before benchmarking." >&2
  echo "        CUDA toolkit is not the same as the NVIDIA driver." >&2
  exit 1
fi
nvidia-smi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
if [[ -z "$DRIVER" ]]; then echo "[ERROR] NVIDIA GPU/driver was not detected." >&2; exit 1; fi
if [[ "$(printf '%s\n' "525.60.13" "$DRIVER" | sort -V | head -n1)" != "525.60.13" ]]; then
  echo "[ERROR] NVIDIA Driver $DRIVER is older than the CUDA 12.1 minimum (525.60.13). Update the driver first." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[WARN] nvcc (CUDA compiler) not found. PyTorch CUDA can still work, but FlashAttention source build will be skipped."
else
  nvcc --version | tail -n 4
fi
python3 --version 2>/dev/null || true
if [[ -n "${MME_ROOT:-}" ]]; then
  [[ -d "$MME_ROOT" ]] || { echo "[ERROR] MME_ROOT does not exist: $MME_ROOT" >&2; exit 1; }
  echo "[OK] MME_ROOT=$MME_ROOT"
else
  echo "[WARN] MME_ROOT is not set. Set it before running step2_smoke_test.sh or later."
fi
echo "[OK] Host preflight passed (driver=$DRIVER)."
