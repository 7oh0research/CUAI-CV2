#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash setup/check_host.sh
FREE_GB=$(df -BG "${WORKSPACE:-/workspace}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}' || df -BG . | awk 'NR==2 {gsub("G","",$4); print $4}')
if [[ -n "${FREE_GB:-}" && "$FREE_GB" -lt 40 ]]; then
  echo "[ERROR] Only ${FREE_GB}GB free. H3 full setup needs at least 40GB free; use a 60GB disk (70GB+ with full datasets). No installation was started."
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ ! -x "$HOME/miniforge3/bin/conda" ]]; then
    echo '[INFO] Installing Miniforge for Linux x86_64...'
    curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
  fi
  export PATH="$HOME/miniforge3/bin:$PATH"
fi
CONDA="$(command -v conda)"

# Four isolated envs are intentional: the upstream implementations have different dependency stacks.
for ENV in h3_vanilla h3_fastv h3_sparsevlm h3_visionzip; do
  if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV"; then
    "$CONDA" create -y -n "$ENV" python=3.10 pip
  fi
  "$CONDA" run -n "$ENV" python -m pip install --upgrade pip setuptools wheel
  "$CONDA" run -n "$ENV" python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
  "$CONDA" run -n "$ENV" python -m pip install -r requirements.txt
 done

 echo '[OK] Four H3 Python environments created.'
 echo 'Next: bash setup/clone_methods.sh'
