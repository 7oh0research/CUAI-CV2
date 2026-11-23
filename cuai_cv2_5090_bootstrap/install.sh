#!/usr/bin/env bash
# Fresh Vast.ai bootstrap for CUAI-CV2 H3 on RTX 5090.
set -Eeuo pipefail

REPO_URL="https://github.com/7oh0research/CUAI-CV2.git"
BRANCH="main"
ROOT="${CUAI_ROOT:-/workspace/CUAI-CV2}"
CONDA_PREFIX_ROOT="${CONDA_PREFIX_ROOT:-$HOME/miniforge3}"
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
CUDA_RUNTIME="12.8"

log(){ printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die(){ echo "[ERROR] $*" >&2; exit 1; }

[[ "$(uname -m)" == x86_64 ]] || die "This package currently targets x86_64 Vast.ai instances."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Choose a Vast.ai NVIDIA CUDA image with driver passthrough."
nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$GPU_NAME" == *"5090"* ]] || die "Expected RTX 5090, detected: $GPU_NAME"

mkdir -p "$(dirname "$ROOT")"
AVAIL_KB="$(df -Pk "$(dirname "$ROOT")" | awk 'NR==2{print $4}')"
AVAIL_GB="$((AVAIL_KB / 1024 / 1024))"
log "Free disk before install: ${AVAIL_GB} GiB"
if (( AVAIL_GB < 40 )); then
  die "Only ${AVAIL_GB} GiB free. This repository creates four isolated method environments and downloads a 7B model plus datasets. A 30GB disk is not a safe target. Increase Vast.ai disk to 60GB (preferably 70GB if storing full datasets) before proceeding. No project installation was started."
fi

# A truly minimal Vast.ai image may not contain git/curl yet. Install only the
# bootstrap tools before cloning; the repository installer adds the rest later.
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$EUID" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO apt-get update
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends git curl ca-certificates
  else
    die "git/curl missing and apt-get unavailable. Install git, curl, ca-certificates, then rerun."
  fi
fi

if [[ ! -d "$ROOT/.git" ]]; then
  [[ ! -e "$ROOT" ]] || die "$ROOT exists but is not a Git repository. Move/remove it first."
  log "Cloning $REPO_URL ($BRANCH)"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
else
  log "Using existing repository at $ROOT"
  git -C "$ROOT" fetch --depth 1 origin "$BRANCH"
  git -C "$ROOT" checkout "$BRANCH"
  git -C "$ROOT" reset --hard "origin/$BRANCH"
fi
cd "$ROOT"

log "Installing OS build/runtime dependencies using repository installer"
bash setup/install_system_deps.sh

if [[ ! -x "$CONDA_PREFIX_ROOT/bin/conda" ]]; then
  log "Installing isolated Miniforge at $CONDA_PREFIX_ROOT"
  CONDA_PREFIX_ROOT="$CONDA_PREFIX_ROOT" bash setup/install_miniforge.sh
fi
CONDA="$CONDA_PREFIX_ROOT/bin/conda"
[[ -x "$CONDA" ]] || die "Conda executable missing: $CONDA"
export CONDA_EXE="$CONDA"
export PATH="$CONDA_PREFIX_ROOT/bin:$PATH"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

log "Patching repository's stale PyTorch pins to the official RTX 5090 CUDA 12.8 wheel pair"
python3 - <<'PY'
from pathlib import Path
files = [Path('setup/make_envs.sh'), Path('setup_h3.sh'), Path('start.sh'), Path('run_h3.sh')]
replacements = {
    '2.7.1': '2.9.1',
    '0.22.1': '0.24.1',
}
for path in files:
    if not path.exists():
        raise SystemExit(f'[ERROR] Required repository file missing: {path}')
    text = path.read_text()
    old = text
    for a, b in replacements.items():
        text = text.replace(a, b)
    if text != old:
        path.write_text(text)
        print(f'[PATCHED] {path}')
# The repo's environment builder defaults to CUDA 12.8 already; enforce the single intended value.
p = Path('setup/make_envs.sh')
s = p.read_text().replace('CUDA_RUNTIME="${CUDA_RUNTIME:-12.8}"', 'CUDA_RUNTIME="${CUDA_RUNTIME:-12.8}"')
p.write_text(s)
PY

log "Installing H3 dependencies (repository currently uses four isolated method environments)"
H3_ONLY=1 INSTALL_FLASH_ATTN=0 bash setup_h3.sh

log "Removing package caches to recover disk space"
"$CONDA" clean --all -y || true
rm -rf "$HOME/.cache/pip" || true

log "Validating all four H3 environments against RTX 5090"
for env in h1_llava h1_fastv h1_sparsevlm h1_visionzip; do
  "$CONDA" run -n "$env" python - <<'PY'
import sys, numpy, torch, torchvision, transformers
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split('+')[0] == '2.9.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.24.1', torchvision.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert numpy.__version__ == '1.26.4', numpy.__version__
assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
assert torch.cuda.is_available()
x = torch.randn((512,512), device='cuda')
y = torch.randn((512,512), device='cuda')
_ = x @ y
print('[OK]', sys.version.split()[0], torch.__version__, torchvision.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0), 'transformers', transformers.__version__)
PY
done

log "Setup complete. Repository: $ROOT"
echo "Next: configure datasets in $ROOT/.env, then run bash run_h3.sh after verifying data paths."
