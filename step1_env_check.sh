#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" ]]; then
  for p in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$p" ]] && CONDA="$p" && break
  done
fi
[[ -n "$CONDA" ]] || { echo "[ERROR] Conda not found. Run bash setup/setup_all.sh" >&2; exit 1; }
"$CONDA" run --no-capture-output -n h1_llava python -m h1bench.env_check

FLASH_COUNT=0
for e in $("$CONDA" env list | awk '$1 ~ /^h1_/ {print $1}'); do
  "$CONDA" run -n "$e" python - <<'PY' > "env/${e}.json"
import json
import torch
try:
    import transformers
    tv = transformers.__version__
except Exception:
    tv = None
try:
    import flash_attn
    fa, fav = True, getattr(flash_attn, "__version__", "unknown")
except Exception:
    fa, fav = False, None
print(json.dumps({"torch": torch.__version__, "cuda_torch": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "transformers": tv,
                  "flash_attn": fa, "flash_attn_version": fav}))
PY
  if grep -q '"flash_attn": true' "env/${e}.json"; then FLASH_COUNT=$((FLASH_COUNT + 1)); fi
done
python3 - "$FLASH_COUNT" <<'PY'
import json, sys
json.dump({"flash_env_count": int(sys.argv[1])}, open("env/env_flags.json", "w"), indent=2)
PY
echo "[OK] Environment check complete. FlashAttention available in $FLASH_COUNT env(s)."
echo "Next: configure MME_ROOT in .env, then bash step2_smoke_test.sh"
