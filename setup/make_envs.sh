#!/usr/bin/env bash
# 방법별 env. 각 저장소의 설치 방식(pip install -e .)을 따른다.
set -e
mk(){ name=$1; dir=$2
  conda env list | grep -q "^$name " || conda create -y -n $name python=3.10
  conda run -n $name pip install --upgrade pip
  [ -d "$dir" ] && conda run -n $name pip install -e "$dir" || echo "$name: $dir 없음" >> ISSUES.md
  conda run -n $name pip install -e . pyyaml pandas matplotlib pynvml
  bash setup/install_flash_attn.sh $name || echo "$name: flash-attn 설치 실패 -> unsupported_env" >> ISSUES.md
}
mk h1_llava third_party/baseline
mk h1_fastv third_party/baseline
[ -d third_party/fastv/src/transformers ] && conda run -n h1_fastv pip install -e third_party/fastv/src/transformers
mk h1_pdrop third_party/pdrop
mk h1_sparsevlm third_party/sparsevlm
mk h1_visionzip third_party/baseline
conda run -n h1_visionzip pip install -e third_party/visionzip
mk h1_divprune third_party/divprune
mk h1_cdpruner third_party/cdpruner
[ -d third_party/anchorprune ] && mk h1_anchorprune third_party/anchorprune
