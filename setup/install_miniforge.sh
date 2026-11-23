#!/usr/bin/env bash
set -Eeuo pipefail

PREFIX="${CONDA_PREFIX_ROOT:-$HOME/miniforge3}"
if [[ -x "$PREFIX/bin/conda" ]]; then
  echo "[OK] Conda already exists: $PREFIX"
  exit 0
fi
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ASSET="Miniforge3-Linux-x86_64.sh" ;;
  aarch64|arm64) ASSET="Miniforge3-Linux-aarch64.sh" ;;
  *) echo "[ERROR] Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://github.com/conda-forge/miniforge/releases/latest/download/$ASSET"
echo "[INFO] Downloading Miniforge: $URL"
curl -fL "$URL" -o "$TMP/miniforge.sh"
bash "$TMP/miniforge.sh" -b -p "$PREFIX"
"$PREFIX/bin/conda" config --set auto_activate_base false
"$PREFIX/bin/conda" config --set channel_priority strict
printf '\n# h1_bench Miniforge\nexport PATH="%s/bin:$PATH"\n' "$PREFIX" >> "$HOME/.bashrc"
echo "[OK] Miniforge installed at $PREFIX"
echo "[INFO] This installer uses 'conda run'; opening a new shell is not required."
