#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$p" ]] && CONDA="$p" && break
  done
fi
[[ -n "$CONDA" ]] || { echo "[ERROR] Conda not found." >&2; exit 1; }
"$CONDA" run --no-capture-output -n h1_llava python -m h1bench.aggregate
echo "완료: results/summary.csv, results/ratios.csv, results/figs/"
