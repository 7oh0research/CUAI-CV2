#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
set -e
for s in step1_env_check step2_smoke_test step3_main_R32_bs1 step4_batch_sweep step5_R_sweep step6_aggregate; do bash $s.sh; done
