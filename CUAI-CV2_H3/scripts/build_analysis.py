#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from h3bench.data.io import read_jsonl
from h3bench.evaluation.analysis import build
p=argparse.ArgumentParser()
p.add_argument('--blind',required=True); p.add_argument('--predictions',required=True); p.add_argument('--out',required=True)
a=p.parse_args()
blind=read_jsonl(a.blind); preds=read_jsonl(a.predictions)
rows=build(blind,preds)
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
if rows:
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'Wrote {len(rows)} summary rows to {out}/metrics.csv')
print(f'Blind-solvable: {sum(bool(r.get("is_correct")) for r in blind)}/{len(blind)}')
