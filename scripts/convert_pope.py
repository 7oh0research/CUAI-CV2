#!/usr/bin/env python3
"""Convert a standard POPE JSON/JSONL annotation file to the H3 manifest."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--image-root',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='random');a=p.parse_args()
src=Path(a.input); raw=src.read_text(encoding='utf-8').strip()
try: data=json.loads(raw); data=data if isinstance(data,list) else [data]
except json.JSONDecodeError: data=[json.loads(x) for x in raw.splitlines() if x.strip()]
root=Path(a.image_root); out=[]
for i,x in enumerate(data):
 image=x.get('image') or x.get('image_name') or x.get('image_path')
 if not image: raise SystemExit(f'row {i}: missing image field')
 image_path=Path(image); image_path=image_path if image_path.is_absolute() else root/image_path
 question=x.get('text') or x.get('question')
 answer=x.get('label') or x.get('answer') or x.get('ground_truth')
 if question is None or answer is None: raise SystemExit(f'row {i}: missing text/question or label/answer')
 qid=x.get('question_id',f'pope_{a.split}_{i:06d}')
 out.append({'question_id':str(qid),'dataset':'pope','category':x.get('category',a.split),'image':str(image_path),'question':question,'ground_truth':str(answer).lower()})
dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
with dst.open('w',encoding='utf-8') as f:
 for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(f'Wrote {len(out)} rows to {dst}')
