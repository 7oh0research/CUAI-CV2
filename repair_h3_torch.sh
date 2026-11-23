#!/usr/bin/env bash
# Rebuild/repair all H3 environments using the pinned Python 3.10 stack.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$p" ]] && CONDA="$p" && break
  done
fi
[[ -n "$CONDA" ]] || { echo "[ERROR] Conda not found. Install Miniforge first." >&2; exit 1; }
export CONDA_EXE="$CONDA"

# Reuse the canonical setup path. It removes environments with the wrong
# Python version, ignores external Conda pins, installs CUDA PyTorch via pip,
# and reinstalls the remaining benchmark dependencies.
H3_ONLY=1 INSTALL_FLASH_ATTN=0 bash setup/make_envs.sh

for env_name in h1_llava h1_fastv h1_sparsevlm h1_visionzip; do
  echo
  echo "========== Validating $env_name =========="
  "$CONDA" run -n "$env_name" python - <<'PY'
import sys, torch, torchvision, transformers
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
assert sys.version_info[:2] == (3, 10), "Expected Python 3.10"
assert torch.__version__.split("+")[0] == "2.1.2"
assert torchvision.__version__.split("+")[0] == "0.16.2"
assert torch.version.cuda is not None, "CPU-only PyTorch was installed"
PY
done

echo
echo "[OK] All H3 environments were repaired and validated."
