#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi unavailable'; exit 1; }
nvidia-smi
FREE=$(df -BG "$ROOT" | awk 'NR==2 {gsub("G","",$4);print $4}')
(( FREE >= 40 )) || { echo "Need >=40GB free; recommend 60-70GB (now ${FREE}GB)"; exit 2; }
if ! command -v conda >/dev/null; then
  if [[ ! -x "$HOME/miniforge3/bin/conda" ]]; then curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh; bash /tmp/mf.sh -b -p "$HOME/miniforge3"; fi
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else eval "$(conda shell.bash hook)"; fi
export PIP_NO_CACHE_DIR=1
bash setup/clone_methods.sh
mk_env(){ local e=$1; conda env list | awk '{print $1}' | grep -qx "$e" || conda create -y -n "$e" python=3.10 pip; conda run -n "$e" pip install -U pip; }
mk_env h3_vanilla
conda run -n h3_vanilla pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n h3_vanilla pip install transformers==4.46.3 accelerate==1.2.1 sentencepiece==0.2.0 protobuf==5.29.3 pillow==11.1.0 numpy==1.26.4 pyyaml==6.0.2
mk_env h3_fastv
conda run -n h3_fastv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n h3_fastv pip install -e third_party/fastv/src/FastV/llava-hf/transformers
conda run -n h3_fastv pip install accelerate==1.2.1 sentencepiece==0.2.0 protobuf==5.29.3 pillow==11.1.0 numpy==1.26.4
mk_env h3_sparsevlm
conda run -n h3_sparsevlm pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n h3_sparsevlm pip install -e third_party/sparsevlm
conda run -n h3_sparsevlm pip install transformers==4.37.0 accelerate sentencepiece protobuf pillow numpy==1.26.4
mk_env h3_visionzip
conda run -n h3_visionzip pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n h3_visionzip pip install -e third_party/llava
conda run -n h3_visionzip pip install -e third_party/visionzip
conda run -n h3_visionzip pip install transformers==4.37.2 accelerate sentencepiece protobuf pillow numpy==1.26.4
for e in h3_vanilla h3_fastv h3_sparsevlm h3_visionzip; do conda run -n "$e" pip install pyyaml==6.0.2 tqdm==4.67.1; done
bash scripts/verify_environment.sh
