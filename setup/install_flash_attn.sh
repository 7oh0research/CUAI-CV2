#!/usr/bin/env bash
# Optional extension. Returns 0 when installed, 2 when safely skipped, 1 on install failure.
set -Eeuo pipefail
E="${1:?Usage: install_flash_attn.sh ENV_NAME}"
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$p" ]] && CONDA="$p" && break
  done
fi
[[ -n "$CONDA" ]] || { echo "[WARN] conda not found; skipping FlashAttention."; exit 2; }

if "$CONDA" run -n "$E" python -c 'import flash_attn' >/dev/null 2>&1; then
  echo "[OK] flash-attn already installed in $E"; exit 0
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[WARN] nvcc is missing; FlashAttention requires a CUDA compiler for this pinned stack. Skipping $E."; exit 2
fi

TORCH_CUDA="$("$CONDA" run -n "$E" python -c 'import torch; print(torch.version.cuda or "")' | tail -n1)"
if [[ "$TORCH_CUDA" != 12.1* ]]; then
  echo "[WARN] $E uses CUDA runtime '$TORCH_CUDA' (expected 12.1); skipping FlashAttention build."; exit 2
fi
CAP="$("$CONDA" run -n "$E" python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))' 2>/dev/null | tail -n1 || true)"
if [[ -z "$CAP" ]]; then echo "[WARN] No visible CUDA GPU; skipping FlashAttention for $E."; exit 2; fi
export TORCH_CUDA_ARCH_LIST="$CAP"
export MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
echo "[INFO] Building flash-attn==2.5.8 for env=$E, CUDA=$TORCH_CUDA, GPU=$CAP, jobs=$MAX_JOBS"
"$CONDA" run --no-capture-output -n "$E" python -m pip install 'flash-attn==2.5.8' --no-build-isolation
"$CONDA" run -n "$E" python -c 'import flash_attn; print("flash-attn", flash_attn.__version__)'
