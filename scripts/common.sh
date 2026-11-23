#!/usr/bin/env bash
# 공통: run METHOD R ATTN BS REPS OUTDIR [extra]
set -u
: "${MME_ROOT:?MME_ROOT 를 설정하세요}"
mkdir -p logs results/raw
env_of(){ python - "$1" <<'EOF'
import yaml,sys;print(yaml.safe_load(open("configs/methods.yaml"))["methods"][sys.argv[1]]["env"])
EOF
}
skip_oom(){ # 같은 method,R,attn 에서 더 작은 bs 가 oom 이면 건너뜀
  for f in results/raw/$1_$2_$3_*_0.jsonl; do [ -f "$f" ] && grep -q '"status": "oom"' "$f" && return 0; done; return 1; }
run(){ m=$1; R=$2; at=$3; bs=$4; reps=$5; dir=$6; shift 6
  if [ "$m" = anchorprune ] && [ ! -d third_party/anchorprune ]; then echo "anchorprune blocked"; return; fi
  if grep -q '"unsupported_env_flash": true' env/env_flags.json 2>/dev/null && [ "$at" = flash_attention_2 ]; then
     echo "{\"type\":\"summary\",\"method\":\"$m\",\"R\":$R,\"attn_requested\":\"$at\",\"batch_size\":$bs,\"status\":\"unsupported_env\"}" > $dir/${m}_${R}_${at}_${bs}_0.jsonl; return; fi
  [ "$dir" = results/raw ] && skip_oom $m $R $at && { echo "skip (oom at smaller bs): $m $R $at $bs"; return; }
  E=$(env_of $m)
  for ((r=0;r<reps;r++)); do
    o=$dir/${m}_${R}_${at}_${bs}_${r}.jsonl
    [ -s "$o" ] && continue   # resume
    echo "[run] $m R=$R attn=$at bs=$bs rep=$r"
    conda run --no-capture-output -n $E python -m h1bench.run_one --method $m --R $R --attn $at --bs $bs --rep $r --out $o "$@" \
       >> logs/${m}_${R}_${at}_${bs}.log 2>&1
    grep -q '"status": "oom"' "$o" 2>/dev/null && break
  done; }
METHODS="baseline fastv pdrop sparsevlm visionzip divprune cdpruner anchorprune"
ATTNS="eager sdpa flash_attention_2"
