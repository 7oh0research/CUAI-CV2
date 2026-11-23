#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PROFILE="full"
if [[ "${1:-}" == "--profile" ]]; then PROFILE="${2:-}"; shift 2; fi
case "$PROFILE" in
  h3)
    CONDA="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "$CONDA" ]]; then for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do [[ -x "$p" ]] && CONDA="$p" && break; done; fi
    NEED=0
    [[ -z "$CONDA" ]] && NEED=1
    if [[ "$NEED" == 0 ]]; then
      for e in h1_llava h1_fastv h1_sparsevlm h1_visionzip; do
        "$CONDA" env list | awk '{print $1}' | grep -qx "$e" || { NEED=1; break; }
        pyver="$("$CONDA" run -n "$e" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null | tail -n1 || true)"
        [[ "$pyver" == "3.10" ]] || { echo "[WARN] $e uses Python ${pyver:-unknown}; setup will repair it."; NEED=1; break; }
      done
    fi
    if [[ "$NEED" == 1 ]]; then bash setup_h3.sh; CONDA="$HOME/miniforge3/bin/conda"; fi
    if [[ ! -f .env ]]; then cp .env.example .env; fi
    if [[ -f .env ]]; then set -a; source .env; set +a; fi
    if [[ -z "${MME_ROOT:-}" || ! -d "$MME_ROOT" ]]; then
      read -r -p 'Absolute path to MME_Benchmark_release_version: ' v
      [[ -d "$v" ]] || { echo '[ERROR] MME directory not found.' >&2; exit 1; }
      printf '\nMME_ROOT=%q\n' "$(cd "$v" && pwd)" >> .env
    fi
    if [[ -z "${POPE_FILE:-}" || ! -f "$POPE_FILE" ]]; then
      read -r -p 'Absolute path to POPE JSONL/CSV annotation file: ' v
      [[ -f "$v" ]] || { echo '[ERROR] POPE annotation file not found.' >&2; exit 1; }
      printf '\nPOPE_FILE=%q\n' "$(cd "$(dirname "$v")" && pwd)/$(basename "$v")" >> .env
    fi
    if [[ -z "${POPE_ROOT:-}" || ! -d "$POPE_ROOT" ]]; then
      read -r -p 'Absolute path to POPE image root directory: ' v
      [[ -d "$v" ]] || { echo '[ERROR] POPE image directory not found.' >&2; exit 1; }
      printf '\nPOPE_ROOT=%q\n' "$(cd "$v" && pwd)" >> .env
    fi
    export CONDA_EXE="$CONDA"
    exec bash run_h3.sh "$@"
    ;;
  full)
    CONDA="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "$CONDA" ]]; then for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do [[ -x "$p" ]] && CONDA="$p" && break; done; fi
    if [[ -z "$CONDA" ]] || ! "$CONDA" env list 2>/dev/null | awk '{print $1}' | grep -qx h1_llava; then bash setup.sh; CONDA="$HOME/miniforge3/bin/conda"; fi
    [[ -f .env ]] || cp .env.example .env
    if [[ -f .env ]]; then set -a; source .env; set +a; fi
    if [[ -z "${MME_ROOT:-}" || ! -d "$MME_ROOT" ]]; then
      read -r -p 'Absolute path to MME_Benchmark_release_version: ' v
      [[ -d "$v" ]] || { echo '[ERROR] MME directory not found.' >&2; exit 1; }
      printf '\nMME_ROOT=%q\n' "$(cd "$v" && pwd)" >> .env
    fi
    export CONDA_EXE="$CONDA"
    exec bash run_pipeline.sh "$@"
    ;;
  *) echo 'Usage: bash start.sh [--profile h3|full]' >&2; exit 2 ;;
esac
