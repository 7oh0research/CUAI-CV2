"""Run all (benchmark x condition) cells. Resumable: finished ids are skipped.

Conditions:
  controls : vanilla/image, vanilla/blank, vanilla/noimage      (H3 filter)
  methods  : <method>/<budget> for every method x budget        (comparison)
Output: results/<mode>/raw/<bench>/<condition>.jsonl
"""
import argparse, json, os, sys, time, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from vtcb.data import iter_items
from vtcb.scoring import score

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["quick", "full"])
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--benches", nargs="*")
ap.add_argument("--methods", nargs="*")
ap.add_argument("--budgets", nargs="*", type=int)
ap.add_argument("--skip-controls", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))

from vtcb.model import LlavaPruned
M = LlavaPruned(cfg)

conds = [] if a.skip_controls else [("vanilla", 576, c) for c in cfg["controls"]]
for m in (a.methods or cfg["methods"]):
    for b in (a.budgets or cfg["budgets"]):
        conds.append((m, b, "image"))

def cname(m, b, c):
    return f"vanilla_{c}" if m == "vanilla" else f"{m}_{b}"

for bench in (a.benches or cfg["benchmarks"]):
    n = cfg["quick_samples"] if a.mode == "quick" else cfg["benchmarks"][bench]["max_samples"]
    d = f"results/{a.mode}/raw/{bench}"; os.makedirs(d, exist_ok=True)
    done = {}
    for m, b, c in conds:
        p = f"{d}/{cname(m, b, c)}.jsonl"
        done[p] = set(json.loads(l)["id"] for l in open(p)) if os.path.exists(p) else set()
    # outer loop over items -> each image is decoded once for all conditions
    items = iter_items(bench, n, cfg["seed"])
    for it in tqdm(items, desc=bench, total=n):
        for m, b, c in conds:
            p = f"{d}/{cname(m, b, c)}.jsonl"
            if it["id"] in done[p]:
                continue
            t = time.time()
            raw, info = M.generate(it, m, b, c)
            s, parsed, ok = score(it, raw)
            rec = dict(id=it["id"], bench=bench, method=m, budget=b, control=c,
                       category=it["category"], kind=it["kind"], n_choices=it["n_choices"],
                       answer=it["answer"], raw=raw, parsed=parsed, parse_ok=ok, score=s,
                       sec=round(time.time() - t, 3), **info)
            with open(p, "a") as f:
                f.write(json.dumps(rec) + "\n")
print("done")
