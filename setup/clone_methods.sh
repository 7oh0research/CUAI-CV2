#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; mkdir -p third_party results/reports
clone(){ local n=$1 u=$2 b=${3:-}; if [[ ! -d third_party/$n/.git ]]; then if [[ -n $b ]]; then git clone --depth 1 -b "$b" "$u" third_party/$n; else git clone --depth 1 "$u" third_party/$n; fi; fi; }
clone llava https://github.com/haotian-liu/LLaVA.git
clone fastv https://github.com/pkunlp-icler/FastV.git
clone sparsevlm https://github.com/Gumpest/SparseVLMs.git v1.5
clone visionzip https://github.com/JIA-Lab-research/VisionZip.git
{ date -Is; for d in third_party/*; do [[ -d $d/.git ]] && echo "$(basename $d) $(git -C "$d" rev-parse HEAD)"; done; } | tee results/reports/source_revisions.txt
