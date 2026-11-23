#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_run_guard.sh"
echo '[STOP] Original-image inference adapter is not yet bound to the pinned LLaVA checkout. See docs/ADAPTER_CONTRACT.md.'
exit 3
