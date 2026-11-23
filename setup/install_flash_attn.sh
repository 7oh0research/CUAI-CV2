#!/usr/bin/env bash
# pip wheel -> 버전 변경 -> 소스 빌드 순서로 3회 시도
E=$1
conda run -n $E pip install flash-attn --no-build-isolation && exit 0
conda run -n $E pip install flash-attn==2.5.8 --no-build-isolation && exit 0
TORCH_CUDA_ARCH_LIST=8.9 conda run -n $E pip install git+https://github.com/Dao-AILab/flash-attention --no-build-isolation && exit 0
exit 1
