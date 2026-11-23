#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/build_analysis.py --blind results/raw/blind.jsonl --predictions results/raw/all_predictions.jsonl --out results/processed
