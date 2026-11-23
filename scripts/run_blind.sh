#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; [[ -f .env ]] && set -a && source .env && set +a
METHOD=vanilla STAGE=blind bash scripts/run_method.sh
python scripts/freeze_blind_set.py --blind results/raw/vanilla_blind.jsonl --out results/processed/blind_solvable_ids.json
