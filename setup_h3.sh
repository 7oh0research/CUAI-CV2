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

# Fail early if any H3 environment is incomplete; never report setup success
# when torch is absent or the wrong CPU-only build is installed.
for env_name in h1_llava h1_fastv h1_sparsevlm h1_visionzip; do
  "$CONDA_EXE" run -n "$env_name" python -c \
    'import torch, torchvision, transformers; assert torch.__version__.split("+")[0] == "2.1.2"; assert torchvision.__version__.split("+")[0] == "0.16.2"; assert torch.version.cuda is not None; print("[OK]", torch.__version__, torchvision.__version__, "CUDA", torch.version.cuda, "GPU available:", torch.cuda.is_available())' \
    || { echo "[ERROR] Incomplete H3 environment: $env_name" >&2; exit 1; }
done
printf '\n[OK] H3 environments installed and validated. Configure MME_ROOT, POPE_ROOT, POPE_FILE in .env.\n'
