#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source scripts/common.sh
for bs in 4 8 16; do for m in $METHODS; do for at in $ATTNS; do R=32; [ $m = baseline ] && R=576; run $m $R $at $bs 3 results/raw; done; done; done
echo "다음: bash step5_R_sweep.sh"
