#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source scripts/common.sh
for m in $METHODS; do for at in $ATTNS; do R=32; [ $m = baseline ] && R=576; run $m $R $at 1 3 results/raw; done; done
# MME 정상 동작 확인 (R=32, bs=1, sdpa -> eager 순으로 ok 인 첫 구현)
for m in $METHODS; do
  [ -d results/mme/$m ] && continue
  for at in sdpa eager; do R=32; [ $m = baseline ] && R=576
    f=results/raw/${m}_${R}_${at}_1_0.jsonl
    if [ -f $f ] && grep -q '"status": "ok"' $f; then
      "$CONDA" run --no-capture-output -n "$(env_of "$m")" python -m h1bench.mme_eval --method $m --attn $at --R 32 >> logs/mme_$m.log 2>&1
      echo "$m,$at" >> results/mme_attn_used.csv; break; fi
  done
done
echo "MME 채점: python \$MME_ROOT/../eval_tool/calculation.py --results_dir results/mme/<method> > results/mme/<method>_score.txt"
echo "다음: bash step4_batch_sweep.sh"
