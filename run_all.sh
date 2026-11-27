#!/usr/bin/env bash
# usage: bash run_all.sh quick|full
set -euo pipefail
MODE=${1:-quick}
source .venv/bin/activate
if [ "$MODE" = full ] && [ ! -f results/quick/SANITY_OK ]; then
  echo "[STOP] run 'bash run_all.sh quick' first (sanity check must pass)"; exit 1; fi
if [ "$MODE" = quick ]; then
  python scripts/sanity_check.py && mkdir -p results/quick && touch results/quick/SANITY_OK
fi
python scripts/run.py "$MODE" 2>&1 | tee -a "results_${MODE}.log"
python scripts/analyze.py "$MODE"
echo "report: results/$MODE/report.md"
