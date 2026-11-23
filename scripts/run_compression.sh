#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; [[ -f .env ]] && set -a && source .env && set +a
: "${METHOD:?METHOD=fastv|sparsevlm|visionzip}"; METHOD="$METHOD" STAGE=compressed bash scripts/run_method.sh
