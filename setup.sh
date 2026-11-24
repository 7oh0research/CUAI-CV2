#!/usr/bin/env bash
# RTX 5090 (Blackwell, sm_120) 환경 설정. Ubuntu 계열 Vast.ai 서버 기준.
set -euo pipefail
cd "$(dirname "$0")"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # sm_120은 CUDA 12.8 이상 빌드 필요

echo "== [1/5] GPU 드라이버 확인"
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi 없음: GPU 인스턴스가 아님" ; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -lt 570 ]; then
  echo "드라이버 $DRV < 570: RTX 5090(CUDA 12.8)을 지원하지 않음. 다른 머신을 빌리세요."; exit 1
fi

echo "== [2/5] 시스템 패키지"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends python3 python3-venv python3-pip git tmux wget ca-certificates libgl1 libglib2.0-0

echo "== [3/5] 가상환경"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "== [4/5] PyTorch (CUDA 12.8 빌드) 및 의존성"
pip install "torch>=2.7" torchvision --index-url "$TORCH_INDEX"
pip install -r requirements.txt
pip freeze > env_freeze.txt

echo "== [5/5] GPU 실행 검증"
python scripts/check_env.py | tee env_check.txt
echo "설치 완료. 'source .venv/bin/activate' 후 실행하세요."
