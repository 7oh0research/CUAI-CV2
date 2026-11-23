#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party env

python3 - <<'PY'
import json, os, subprocess
repos = {
    "baseline": "https://github.com/haotian-liu/LLaVA",
    "fastv": "https://github.com/pkunlp-icler/FastV",
    "pdrop": "https://github.com/Cooperx521/PyramidDrop",
    "sparsevlm": "https://github.com/Gumpest/SparseVLMs",
    "visionzip": "https://github.com/dvlab-research/VisionZip",
    "divprune": "https://github.com/vbdi/divprune",
    "cdpruner": "https://github.com/Theia-4869/CDPruner",
}
records = {}
for name, url in repos.items():
    path = f"third_party/{name}"
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(["git", "clone", "--depth", "1", url, path], check=True)
    commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    records[name] = {"url": url, "commit": commit}
anchor = os.environ.get("ANCHORPRUNE_PATH", "")
if anchor and os.path.isdir(anchor):
    anchor = os.path.abspath(anchor)
    os.symlink(anchor, "third_party/anchorprune") if not os.path.lexists("third_party/anchorprune") else None
    try: commit = subprocess.check_output(["git", "-C", anchor, "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = "unknown"
    records["anchorprune"] = {"path": anchor, "commit": commit}
else:
    records["anchorprune"] = {"status": "blocked", "reason": "ANCHORPRUNE_PATH not provided"}
with open("env/methods.json", "w") as f: json.dump(records, f, indent=2)
print(json.dumps(records, indent=2))
PY
