#!/usr/bin/env bash
# 사용법: bash run_all.sh quick | full
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
MODE="${1:-quick}"
if [ "$MODE" = "quick" ]; then N="--max-samples 200"; RUN="quick"; else N=""; RUN="full"; fi
python -m h3.run --config configs/default.yaml --run-name "$RUN" $N 2>&1 | tee -a "logs_${RUN}.txt"
python -m h3.analyze --config configs/default.yaml --run-name "$RUN" 2>&1 | tee -a "logs_${RUN}.txt"
echo "보고서: results/$RUN/report.md"
