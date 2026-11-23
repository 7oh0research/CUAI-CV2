#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_run_guard.sh"
echo '[STOP] Blind inference adapter is not yet bound to a pinned LLaVA checkout.'
echo 'See docs/ADAPTER_CONTRACT.md. Do not substitute a text-only model; it must be the same LLaVA-1.5-7B language backbone with image input removed.'
exit 3
