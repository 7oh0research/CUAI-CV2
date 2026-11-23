#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$p" ]] && CONDA="$p" && break
  done
fi
[[ -n "$CONDA" ]] || { echo "[ERROR] Conda not found. Run bash setup/install_miniforge.sh first." >&2; exit 1; }
export CONDA_EXE="$CONDA"

# LLaVA-1.5 compatibility baseline. Keep these versions aligned with upstream LLaVA.
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
CUDA_RUNTIME="${CUDA_RUNTIME:-12.1}"

create_env() {
  local name="$1" repo="$2"
  if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$name"; then
    "$CONDA" create -y -n "$name" "python=$PYTHON_VERSION" pip
  fi
  "$CONDA" install -y -n "$name" -c pytorch -c nvidia \
    "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" "pytorch-cuda=$CUDA_RUNTIME"
  "$CONDA" run --no-capture-output -n "$name" python -m pip install --upgrade 'pip<25' 'setuptools<76' wheel packaging ninja
  "$CONDA" run --no-capture-output -n "$name" python -m pip install \
    'transformers==4.37.2' 'tokenizers==0.15.1' 'sentencepiece==0.1.99' \
    'accelerate==0.21.0' 'numpy==1.26.4' 'scikit-learn==1.2.2' \
    'einops==0.6.1' 'einops-exts==0.0.4' 'timm==0.6.13' \
    'httpx==0.24.0' 'gradio==4.16.0' 'gradio_client==0.8.1' \
    'bitsandbytes==0.43.1' 'peft==0.10.0' 'pydantic<3' \
    pillow pyyaml pandas matplotlib pynvml requests shortuuid scipy
  if [[ -d "$repo" ]]; then
    if [[ -f "$repo/pyproject.toml" || -f "$repo/setup.py" ]]; then
      "$CONDA" run --no-capture-output -n "$name" python -m pip install -e "$repo" --no-deps
    else
      echo "[WARN] $repo has no pyproject.toml/setup.py; skipping editable install."
    fi
  else
    echo "[WARN] Method source not found: $repo"
  fi
  # Install this benchmark package without allowing pip to replace the pinned stack.
  "$CONDA" run --no-capture-output -n "$name" python -m pip install -e . --no-deps
  "$CONDA" run -n "$name" python -m pip check || true
  echo "[OK] Environment ready: $name"
}

if [[ "${H3_ONLY:-0}" == "1" ]]; then
  create_env h1_llava third_party/baseline
  create_env h1_fastv third_party/baseline
  if [[ -d third_party/fastv/src/transformers ]]; then
    "$CONDA" run --no-capture-output -n h1_fastv python -m pip install -e third_party/fastv/src/transformers --no-deps
  else
    echo "[WARN] FastV transformers fork not found at third_party/fastv/src/transformers."
  fi
  create_env h1_sparsevlm third_party/sparsevlm
  create_env h1_visionzip third_party/baseline
  if [[ -d third_party/visionzip ]]; then
    "$CONDA" run --no-capture-output -n h1_visionzip python -m pip install -e third_party/visionzip --no-deps || echo "[WARN] VisionZip editable install failed."
  fi
else
  create_env h1_llava third_party/baseline
  create_env h1_fastv third_party/baseline
  if [[ -d third_party/fastv/src/transformers ]]; then
    "$CONDA" run --no-capture-output -n h1_fastv python -m pip install -e third_party/fastv/src/transformers --no-deps
  else
    echo "[WARN] FastV transformers fork not found at third_party/fastv/src/transformers."
  fi
  create_env h1_pdrop third_party/pdrop
  create_env h1_sparsevlm third_party/sparsevlm
  create_env h1_visionzip third_party/baseline
  if [[ -d third_party/visionzip ]]; then
    "$CONDA" run --no-capture-output -n h1_visionzip python -m pip install -e third_party/visionzip --no-deps || echo "[WARN] VisionZip editable install failed."
  fi
  create_env h1_divprune third_party/divprune
  create_env h1_cdpruner third_party/cdpruner
  if [[ -d third_party/anchorprune ]]; then create_env h1_anchorprune third_party/anchorprune; fi
fi

if [[ "${INSTALL_FLASH_ATTN:-auto}" != "0" ]]; then
  for e in h1_llava h1_fastv h1_pdrop h1_sparsevlm h1_visionzip h1_divprune h1_cdpruner; do
    if [[ "${H3_ONLY:-0}" == "1" && ! "$e" =~ ^(h1_llava|h1_fastv|h1_sparsevlm|h1_visionzip)$ ]]; then continue; fi
    bash setup/install_flash_attn.sh "$e" || {
      rc=$?
      echo "[WARN] FlashAttention unavailable for $e (exit=$rc). Other attention modes remain available."
    }
  done
  if [[ -d third_party/anchorprune ]]; then bash setup/install_flash_attn.sh h1_anchorprune || true; fi
fi

echo "[OK] All requested environments have been processed."
