#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

bash setup/install_system_deps.sh
bash setup/install_miniforge.sh
export PATH="${CONDA_PREFIX_ROOT:-$HOME/miniforge3}/bin:$PATH"
export CONDA_EXE="${CONDA_PREFIX_ROOT:-$HOME/miniforge3}/bin/conda"
bash setup/check_host.sh
bash setup/clone_methods.sh
bash setup/make_envs.sh
bash step1_env_check.sh

echo
echo "[DONE] Initial setup finished."
echo "Next: configure MME_ROOT in .env, then run bash step2_smoke_test.sh"
