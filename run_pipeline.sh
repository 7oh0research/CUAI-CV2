#!/usr/bin/env bash
# End-to-end orchestration: preflight -> smoke -> estimate -> benchmark/MME -> aggregate.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

YES=0
SKIP_SMOKE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Usage: bash run_pipeline.sh [--yes] [--skip-smoke] [--dry-run]" >&2; exit 2 ;;
  esac
done

if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${MME_ROOT:?Set MME_ROOT in .env before running the full pipeline.}"
[[ -d "$MME_ROOT" ]] || { echo "[ERROR] MME_ROOT does not exist: $MME_ROOT" >&2; exit 1; }
mkdir -p logs results/raw results/smoke results/mme env

if [[ -e .pipeline.lock ]]; then
  echo "[ERROR] .pipeline.lock exists. Another pipeline may be running. If not, remove it manually." >&2
  exit 1
fi
: > .pipeline.lock
cleanup() { rm -f .pipeline.lock; }
trap cleanup EXIT INT TERM

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a logs/pipeline.log; }
run_stage() {
  local name="$1"; shift
  log "START: $name"
  "$@" 2>&1 | tee -a logs/pipeline.log
  log "DONE: $name"
}

log "Pipeline started in $ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "Would run: step1_env_check -> step2_smoke_test -> runtime estimate -> step3 -> step4 -> step5 -> step6"
  exit 0
fi

run_stage "Environment validation" bash step1_env_check.sh
if [[ "$SKIP_SMOKE" == 0 ]]; then
  if ! run_stage "Smoke test" bash step2_smoke_test.sh; then
    echo "[STOP] Smoke Test failed. Inspect results/smoke/smoke_report.txt and logs/ before continuing." | tee -a logs/pipeline.log
    exit 3
  fi
  if grep -Eq 'FAIL|MISMATCH' results/smoke/smoke_report.txt; then
    echo "[STOP] Smoke Test found a failure or token-count mismatch." | tee -a logs/pipeline.log
    echo "Inspect results/smoke/smoke_report.txt and logs/ before running the full grid." | tee -a logs/pipeline.log
    exit 3
  fi
else
  log "WARNING: Smoke Test skipped by user request."
fi

log "GPU-specific runtime estimate"
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do [[ -x "$p" ]] && conda_exe="$p" && break; done
fi
[[ -n "$conda_exe" ]] || { echo "[ERROR] Conda not found." >&2; exit 1; }
"$conda_exe" run --no-capture-output -n h1_llava python -m h1bench.runtime_estimate | tee -a logs/pipeline.log

if [[ "$YES" == 0 ]]; then
  echo
  echo "The full benchmark can take many hours and may incur GPU rental costs."
  read -r -p "Start full benchmark now? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { log "User cancelled before full benchmark."; exit 0; }
fi

run_stage "Main R=32 / bs=1 benchmark + MME prediction generation" bash step3_main_R32_bs1.sh
run_stage "Batch-size sweep" bash step4_batch_sweep.sh
run_stage "Visual-token R sweep" bash step5_R_sweep.sh
run_stage "Aggregate CSV and figures" bash step6_aggregate.sh

# Optional official MME scorer. Do not guess or silently invent a scorer path/CLI.
MME_EVAL_TOOL="${MME_EVAL_TOOL:-}"
if [[ -z "$MME_EVAL_TOOL" && -f "$MME_ROOT/../eval_tool/calculation.py" ]]; then
  MME_EVAL_TOOL="$MME_ROOT/../eval_tool/calculation.py"
fi
if [[ -n "$MME_EVAL_TOOL" && -f "$MME_EVAL_TOOL" ]]; then
  log "Official MME evaluation tool detected: $MME_EVAL_TOOL"
  for d in results/mme/*; do
    [[ -d "$d" && -f "$d/.complete" ]] || continue
    name="$(basename "$d")"
    log "Official MME scoring: $name"
    "$conda_exe" run --no-capture-output -n h1_llava python "$MME_EVAL_TOOL" --results_dir "$d" > "results/mme/${name}_score.txt" 2>&1 || {
      echo "[WARN] Official MME scorer failed for $name. Check results/mme/${name}_score.txt" | tee -a logs/pipeline.log
    }
  done
else
  log "Official MME scorer not found. Prediction files are generated; scoring remains a separate step."
fi

python3 - <<'PY' | tee -a logs/pipeline.log
from pathlib import Path
import csv, json
p = Path("results/summary.csv")
rows = list(csv.DictReader(p.open())) if p.exists() else []
counts = {}
for r in rows:
    counts[r.get("status", "unknown")] = counts.get(r.get("status", "unknown"), 0) + 1
report = {
    "summary_rows": len(rows),
    "status_counts": counts,
    "mme_completed_methods": sorted(x.parent.name for x in Path("results/mme").glob("*/.complete")),
    "summary_csv": str(p),
    "ratios_csv": "results/ratios.csv",
    "figures": sorted(str(x) for x in Path("results/figs").glob("*.png")),
}
Path("results/final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
log "PIPELINE COMPLETE. See results/final_report.json and logs/pipeline.log"
