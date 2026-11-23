#!/usr/bin/env bash
set -e
mkdir -p third_party env
declare -A REPOS=(
 [baseline]=https://github.com/haotian-liu/LLaVA
 [fastv]=https://github.com/pkunlp-icler/FastV
 [pdrop]=https://github.com/Cooperx521/PyramidDrop
 [sparsevlm]=https://github.com/Gumpest/SparseVLMs
 [visionzip]=https://github.com/dvlab-research/VisionZip
 [divprune]=https://github.com/vbdi/divprune
 [cdpruner]=https://github.com/Theia-4869/CDPruner )
echo "{" > env/methods.json
for m in "${!REPOS[@]}"; do
  [ -d third_party/$m ] || git clone --depth 1 ${REPOS[$m]} third_party/$m || { echo "$m clone failed" >> ISSUES.md; continue; }
  h=$(git -C third_party/$m rev-parse HEAD)
  echo "  \"$m\": {\"url\": \"${REPOS[$m]}\", \"commit\": \"$h\"}," >> env/methods.json
done
if [ -n "$ANCHORPRUNE_PATH" ] && [ -d "$ANCHORPRUNE_PATH" ]; then
  ln -sfn "$ANCHORPRUNE_PATH" third_party/anchorprune
  h=$(git -C "$ANCHORPRUNE_PATH" rev-parse HEAD 2>/dev/null || echo none)
  echo "  \"anchorprune\": {\"url\": \"$ANCHORPRUNE_PATH\", \"commit\": \"$h\"}" >> env/methods.json
else
  echo "  \"anchorprune\": {\"status\": \"blocked\"}" >> env/methods.json
fi
echo "}" >> env/methods.json
