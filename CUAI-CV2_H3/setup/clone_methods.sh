#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party
clone_or_update() {
  local name="$1" url="$2" dest="third_party/$1"
  if [[ -d "$dest/.git" ]]; then
    echo "[INFO] $name already exists; leaving working tree untouched."
  else
    git clone --depth 1 "$url" "$dest"
  fi
}
clone_or_update llava https://github.com/haotian-liu/LLaVA.git
clone_or_update fastv https://github.com/pkunlp-icler/fastv.git
clone_or_update sparsevlm https://github.com/Gumpest/SparseVLMs.git
clone_or_update visionzip https://github.com/JIA-Lab-research/VisionZip.git
{
  echo "Generated: $(date -Is)"
  for d in third_party/*; do
    [[ -d "$d/.git" ]] || continue
    echo "$(basename "$d") $(git -C "$d" rev-parse HEAD)"
  done
} | tee results/reports/source_revisions.txt
