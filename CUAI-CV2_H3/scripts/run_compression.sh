#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_run_guard.sh"
echo '[STOP] FastV/SparseVLM/VisionZip adapters require upstream API/commit validation before execution. See docs/ADAPTER_CONTRACT.md.'
exit 3
