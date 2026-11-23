#!/usr/bin/env bash
# One-command bootstrap for an already-extracted h1_bench repository on Vast.ai/Linux.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f setup/setup_all.sh ]]; then
  echo "[ERROR] setup/setup_all.sh not found. Run this script from the extracted h1_bench repository." >&2
  exit 1
fi

# Create a local config file only if the user has not made one.
# MME_ROOT is intentionally left unset until the dataset path is known.
if [[ ! -f .env ]]; then
  cat > .env <<'ENVEOF'
# Optional: set this to the absolute path of the extracted MME_Benchmark_release_version folder.
# Example: MME_ROOT=/workspace/h1_bench/data/MME_Benchmark_release_version
MME_ROOT=

# Optional: local AnchorPrune checkout. Leave empty if not available.
ANCHORPRUNE_PATH=

# Optional: set INSTALL_FLASH_ATTN=0 to skip FlashAttention compilation.
# INSTALL_FLASH_ATTN=auto
# FLASH_ATTN_MAX_JOBS=4
ENVEOF
  echo "[INFO] Created .env (MME_ROOT can be configured after setup)."
fi

# Preserve the current shell's project path and let setup_all handle all installation steps.
exec bash setup/setup_all.sh
