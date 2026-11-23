#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
if [[ -n "$SUDO" ]] && ! command -v sudo >/dev/null 2>&1; then
  echo "[ERROR] sudo is required. Run as root or install sudo first." >&2; exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    ca-certificates curl wget git git-lfs unzip zip bzip2 xz-utils python3 \
    build-essential gcc g++ make cmake ninja-build pkg-config \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libgomp1 \
    pciutils htop tmux
elif command -v dnf >/dev/null 2>&1; then
  $SUDO dnf install -y \
    ca-certificates curl wget git git-lfs unzip zip bzip2 xz python3 \
    gcc gcc-c++ make cmake ninja-build pkgconf-pkg-config \
    mesa-libGL libSM libXext libXrender libgomp pciutils htop tmux
elif command -v yum >/dev/null 2>&1; then
  $SUDO yum install -y \
    ca-certificates curl wget git git-lfs unzip zip bzip2 xz python3 \
    gcc gcc-c++ make cmake ninja-build pkgconfig \
    mesa-libGL libSM libXext libXrender libgomp pciutils htop tmux
elif command -v pacman >/dev/null 2>&1; then
  $SUDO pacman -Sy --needed --noconfirm \
    ca-certificates curl wget git git-lfs unzip zip bzip2 xz python \
    base-devel cmake ninja pkgconf libglvnd libsm libxext libxrender \
    pciutils htop tmux
else
  echo "[ERROR] Unsupported package manager. Install git, curl, wget, unzip, Python 3, build tools, and OpenGL runtime libraries manually." >&2
  exit 1
fi

git lfs install --skip-repo 2>/dev/null || true
echo "[OK] Linux system dependencies installed."
