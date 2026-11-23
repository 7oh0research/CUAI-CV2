#!/usr/bin/env python3
"""Validate a normalized H3 JSONL manifest and emit a stable question list."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
seen=set(); rows=[]
for i,line in enumerate(Path(a.input).open(encoding='utf-8'),1):
 if not line.strip(): continue
 r=json.loads(line)
 for k in ('question_id','dataset','question','ground_truth'):
  if k not in r: raise SystemExit(f'Line {i}: missing {k}')
 q=str(r['question_id'])
 if q in seen: raise SystemExit(f'Duplicate question_id: {q}')
 seen.add(q)
 if r.get('image') and not Path(r['image']).is_file(): raise SystemExit(f'Line {i}: image not found: {r["image"]}')
 rows.append(r)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with Path(a.output).open('w',encoding='utf-8') as f:
 for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'Validated {len(rows)} questions -> {a.output}')
