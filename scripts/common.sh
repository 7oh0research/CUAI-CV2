#!/usr/bin/env bash
# Common runner: run METHOD R ATTN BS REPS OUTDIR [extra args]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${MME_ROOT:?Set MME_ROOT in .env (see .env.example) before running benchmark steps.}"
[[ -d "$MME_ROOT" ]] || { echo "[ERROR] MME_ROOT does not exist: $MME_ROOT" >&2; exit 1; }
mkdir -p logs results/raw env

CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$p" ]] && CONDA="$p" && break
  done
fi
[[ -n "$CONDA" ]] || { echo "[ERROR] Conda not found." >&2; exit 1; }

env_of(){ "$CONDA" run -n h1_llava python - "$1" <<'PY'
import yaml,sys
print(yaml.safe_load(open("configs/methods.yaml"))["methods"][sys.argv[1]]["env"])
PY
}
skip_oom(){
  for f in results/raw/"$1"_"$2"_"$3"_*_0.jsonl; do
    [[ -f "$f" ]] && grep -q '"status": "oom"' "$f" && return 0
  done
  return 1
}
run(){
  local m="$1" R="$2" at="$3" bs="$4" reps="$5" dir="$6"; shift 6
  if [[ "$m" == anchorprune && ! -d third_party/anchorprune ]]; then echo "[BLOCKED] anchorprune: set ANCHORPRUNE_PATH in .env"; return 0; fi
  local E; E="$(env_of "$m")"
  if [[ "$at" == flash_attention_2 ]] && ! grep -q '"flash_attn": true' "env/${E}.json" 2>/dev/null; then
    echo "{\"type\":\"summary\",\"method\":\"$m\",\"R\":$R,\"attn_requested\":\"$at\",\"batch_size\":$bs,\"status\":\"unsupported_env\",\"error\":\"flash-attn is not installed in $E\"}" > "$dir/${m}_${R}_${at}_${bs}_0.jsonl"
    return 0
  fi
  if [[ "$dir" == results/raw ]] && skip_oom "$m" "$R" "$at"; then echo "[SKIP] smaller batch already OOM: $m R=$R $at bs=$bs"; return 0; fi
  for ((r=0; r<reps; r++)); do
    local o="$dir/${m}_${R}_${at}_${bs}_${r}.jsonl"
    [[ -s "$o" ]] && continue
    echo "[RUN] $m R=$R attn=$at bs=$bs rep=$r"
    "$CONDA" run --no-capture-output -n "$E" python -m h1bench.run_one \
      --method "$m" --R "$R" --attn "$at" --bs "$bs" --rep "$r" --out "$o" "$@" \
      >> "logs/${m}_${R}_${at}_${bs}.log" 2>&1 || true
    grep -q '"status": "oom"' "$o" 2>/dev/null && break || true
    [[ -s "$o" ]] || { echo "[ERROR] No result file produced. See logs/${m}_${R}_${at}_${bs}.log" >&2; return 1; }
  done
}
METHODS="baseline fastv pdrop sparsevlm visionzip divprune cdpruner anchorprune"
ATTNS="eager sdpa flash_attention_2"
