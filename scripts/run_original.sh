#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; [[ -f .env ]] && set -a && source .env && set +a
METHOD=vanilla STAGE=original bash scripts/run_method.sh
