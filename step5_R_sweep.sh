#!/usr/bin/env bash
source scripts/common.sh
for R in 64 128 192; do for bs in 1 4 8 16; do for m in $METHODS; do [ $m = baseline ] && continue
  for at in $ATTNS; do run $m $R $at $bs 3 results/raw; done; done; done; done
echo "다음: bash step6_aggregate.sh"
