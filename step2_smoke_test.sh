#!/usr/bin/env bash
source scripts/common.sh
mkdir -p results/smoke
for m in $METHODS; do for at in sdpa eager flash_attention_2; do run $m 32 $at 1 1 results/smoke --n 4 --smoke; done; done
python -m h1bench.smoke_report
echo "다음: bash step3_main_R32_bs1.sh   (확인 파일: results/smoke/smoke_report.txt)"
