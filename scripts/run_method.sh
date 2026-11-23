#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; [[ -f .env ]] && set -a && source .env && set +a
: "${H3_MANIFEST:?Set H3_MANIFEST}"; : "${METHOD:?Set METHOD}"; STAGE=${STAGE:-compressed}; LIMIT_ARGS=(); [[ -n ${LIMIT:-} ]] && LIMIT_ARGS=(--limit "$LIMIT")
case "$METHOD:$STAGE" in
 vanilla:blind|vanilla:original) conda run --no-capture-output -n h3_vanilla python src/h3bench/runner.py --manifest "$H3_MANIFEST" --model "${HF_MODEL_PATH:?}" --revision "${HF_MODEL_REVISION:-a272c74}" --condition "$STAGE" --method vanilla --out "results/raw/vanilla_${STAGE}.jsonl" "${LIMIT_ARGS[@]}" ;;
 fastv:*) conda run --no-capture-output -n h3_fastv python src/h3bench/fastv_runner.py --manifest "$H3_MANIFEST" --model "${HF_MODEL_PATH:?}" --condition "$STAGE" --out "results/raw/fastv_${STAGE}.jsonl" "${LIMIT_ARGS[@]}" ;;
 sparsevlm:*) RETAIN_TOKN="${TOKEN_BUDGET:-64}" conda run --no-capture-output -n h3_sparsevlm python src/h3bench/legacy_runner.py --method sparsevlm --manifest "$H3_MANIFEST" --model "${LLAVA_MODEL_PATH:?}" --condition "$STAGE" --out "results/raw/sparsevlm_${STAGE}.jsonl" "${LIMIT_ARGS[@]}" ;;
 visionzip:*) conda run --no-capture-output -n h3_visionzip python src/h3bench/legacy_runner.py --method visionzip --manifest "$H3_MANIFEST" --model "${LLAVA_MODEL_PATH:?}" --condition "$STAGE" --out "results/raw/visionzip_${STAGE}.jsonl" "${LIMIT_ARGS[@]}" ;;
 *) echo "Unknown method/stage $METHOD/$STAGE"; exit 2;; esac
