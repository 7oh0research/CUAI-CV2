#!/usr/bin/env bash
# Minimal setup for H3 only: LLaVA + FastV + SparseVLM + VisionZip environments.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
bash setup/install_system_deps.sh
bash setup/install_miniforge.sh
export PATH="$HOME/miniforge3/bin:$PATH"; export CONDA_EXE="$HOME/miniforge3/bin/conda"
bash setup/check_host.sh
H3_ONLY=1 bash setup/clone_methods.sh
H3_ONLY=1 bash setup/make_envs.sh
"$CONDA_EXE" run -n h1_llava python -m pip install -e . --no-deps
"$CONDA_EXE" run -n h1_fastv python -m pip install -e . --no-deps
"$CONDA_EXE" run -n h1_sparsevlm python -m pip install -e . --no-deps
"$CONDA_EXE" run -n h1_visionzip python -m pip install -e . --no-deps
printf '\n[OK] H3 environments installed. Configure MME_ROOT, POPE_ROOT, POPE_FILE in .env.\n'
