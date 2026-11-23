#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f .env ]] || { echo '[ERROR] .env missing. Run: cp .env.example .env and edit paths.'; exit 2; }
set -a; source .env; set +a
: "${MODEL_PATH:?Set MODEL_PATH in .env}"
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"; exit 2; }
