#!/usr/bin/env bash
# Install OS packages without depending on the NVIDIA CUDA APT mirror.
# Vast.ai images normally expose the host NVIDIA driver already; PyTorch's
# CUDA runtime is installed in Conda, so the CUDA APT repository is not needed
# for this setup step.
set -Eeuo pipefail

if [[ "${EUID}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
if [[ -n "$SUDO" ]] && ! command -v sudo >/dev/null 2>&1; then
  echo "[ERROR] sudo is required. Run as root or install sudo first." >&2
  exit 1
fi

apt_update_with_cuda_fallback() {
  echo "[INFO] Updating APT package indexes..."
  if $SUDO apt-get update; then
    return 0
  fi

  echo "[WARN] APT update failed. Checking for a temporarily inconsistent NVIDIA CUDA repository..."
  local backup_dir
  backup_dir="$(mktemp -d /tmp/h1bench-apt-sources.XXXXXX)"
  local -a disabled_sources=()

  restore_sources() {
    local f
    for f in "${disabled_sources[@]:-}"; do
      if [[ -f "$backup_dir/$(basename "$f")" ]]; then
        $SUDO mv "$backup_dir/$(basename "$f")" "$f"
      fi
    done
    rm -rf "$backup_dir"
  }
  trap restore_sources EXIT

  # Disable only source files that point to NVIDIA's CUDA APT repository.
  # Keep the normal Ubuntu repositories enabled. Restore the files on exit.
  while IFS= read -r -d '' src; do
    if $SUDO grep -Eqi 'developer\.download\.nvidia\.com/compute/cuda' "$src"; then
      local base
      base="$(basename "$src")"
      echo "[WARN] Temporarily disabling inconsistent CUDA APT source: $src"
      $SUDO mv "$src" "$backup_dir/$base"
      disabled_sources+=("$src")
    fi
  done < <(find /etc/apt -maxdepth 3 -type f \( -name '*.list' -o -name '*.sources' \) -print0 2>/dev/null)

  if [[ ${#disabled_sources[@]} -eq 0 ]]; then
    echo "[ERROR] APT update failed, and no NVIDIA CUDA source could be isolated." >&2
    return 1
  fi

  # Clear stale partial indexes and update using Ubuntu/other non-CUDA sources.
  $SUDO rm -rf /var/lib/apt/lists/*
  $SUDO apt-get clean
  $SUDO apt-get update
  echo "[INFO] APT indexes updated without the NVIDIA CUDA repository."
}

if command -v apt-get >/dev/null 2>&1; then
  apt_update_with_cuda_fallback
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
