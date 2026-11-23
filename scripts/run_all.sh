#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; [[ -f .env ]] && set -a && source .env && set +a
bash scripts/verify_environment.sh
bash scripts/run_blind.sh
bash scripts/run_original.sh
for METHOD in fastv sparsevlm visionzip; do METHOD="$METHOD" bash scripts/run_compression.sh; done
bash scripts/analyze.sh
