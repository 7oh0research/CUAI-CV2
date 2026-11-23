#!/usr/bin/env bash
# H3-only pipeline. H1 is covered by prior-work analysis; no H1 grid is run.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do [[ -x "$p" ]] && CONDA="$p" && break; done; fi
[[ -n "$CONDA" ]] || { echo '[ERROR] Conda not found. Run bash setup_h3.sh first.' >&2; exit 1; }
[[ -d "${MME_ROOT:-}" ]] || { echo '[ERROR] Set MME_ROOT in .env to MME_Benchmark_release_version.' >&2; exit 1; }
[[ -n "${POPE_FILE:-}" && -f "$POPE_FILE" ]] || { echo '[ERROR] Set POPE_FILE to the POPE JSONL/CSV annotation file.' >&2; exit 1; }
[[ -d "${POPE_ROOT:-}" ]] || { echo '[ERROR] Set POPE_ROOT to the directory containing POPE images.' >&2; exit 1; }
for env_name in h1_llava h1_fastv h1_sparsevlm h1_visionzip; do
  "$CONDA" env list | awk '{print $1}' | grep -qx "$env_name" || { echo "[ERROR] Missing Conda env: $env_name. Run bash setup_h3.sh" >&2; exit 1; }
done
mkdir -p results/h3 logs/h3
if [[ -e .h3_pipeline.lock ]]; then echo '[ERROR] H3 pipeline lock exists.' >&2; exit 1; fi
: > .h3_pipeline.lock; trap 'rm -f .h3_pipeline.lock' EXIT

run_env() { local env="$1"; shift; "$CONDA" run --no-capture-output -n "$env" python "$@"; }
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/h3/pipeline.log; }

# Print exact dataset counts before downloading/loading model weights.
run_env h1_llava - <<'PY' | tee -a logs/h3/pipeline.log
import os
from h1bench.h3_data import load_h3_data
x=load_h3_data(os.environ['MME_ROOT'],os.environ['POPE_ROOT'],os.environ['POPE_FILE'])
from collections import Counter
print('H3 dataset counts:',dict(Counter(i['dataset'] for i in x)),'total=',len(x))
PY

BLIND=results/h3/blind_baseline.jsonl
if [[ ! -s "$BLIND" ]]; then
  log 'Stage 1/3: Vanilla blind inference (no image)'
  run_env h1_llava -m h1bench.h3_eval --method baseline --mode blind --output "$BLIND" 2>&1 | tee logs/h3/blind.log
else log 'Resume: blind predictions already exist'; fi

for method in baseline fastv sparsevlm visionzip; do
  env_name="h1_llava"; [[ "$method" == fastv ]] && env_name=h1_fastv; [[ "$method" == sparsevlm ]] && env_name=h1_sparsevlm; [[ "$method" == visionzip ]] && env_name=h1_visionzip
  out="results/h3/${method}_visual.jsonl"
  if [[ -s "$out" ]]; then log "Resume: $method visual predictions already exist"; continue; fi
  log "Stage 2/3: visual inference method=$method env=$env_name"
  if ! run_env "$env_name" -m h1bench.h3_eval --method "$method" --mode visual --output "$out" 2>&1 | tee "logs/h3/${method}.log"; then
    if [[ "$method" == "baseline" ]]; then
      log "FATAL: Vanilla visual inference failed; cannot establish the control condition."
      exit 4
    fi
    echo "unsupported_or_failed: inspect logs/h3/${method}.log" > "results/h3/${method}.unsupported"
    log "WARNING: $method could not run in its official environment; recorded as unsupported/failed and continuing."
  fi
done

log 'Stage 3/3: calculate Score A / Score B'
run_env h1_llava -m h1bench.h3_report --blind "$BLIND" --results results/h3 --out results/h3 | tee -a logs/h3/pipeline.log
log 'H3 pipeline completed. See results/h3/h3_scores.csv and h3_report.json.'
