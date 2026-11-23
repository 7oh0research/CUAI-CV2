#!/usr/bin/env bash
# Backward-compatible one-command entry point for the full experiment pipeline.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/run_pipeline.sh" "$@"
