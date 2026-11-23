import argparse,csv,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='results/raw');p.add_argument('--out',default='results/reports');a=p.parse_args()
def norm(s):return ' '.join(re.sub(r'[^a-z0-9 ]',' ',str(s).lower()).split())
def correct(r):
 pred=norm(r.get('prediction',''));gt=norm(r.get('ground_truth',''))
 if gt in ('yes','no'):return bool(pred) and pred.split()[0]==gt
 return pred==gt or bool(gt and gt in pred)
load=lambda f:[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);blind=load(root/'vanilla_blind.jsonl');ids={str(x['question_id']) for x in blind if correct(x)};(out/'blind_solvable_ids.json').write_text(json.dumps(sorted(ids),indent=2));summary=[]
for f in sorted(root.glob('*.jsonl')):
 data=load(f)
 if not data:continue
 vis=[x for x in data if str(x['question_id']) not in ids]
 
 def cls(x): return norm(x.get('prediction','')).split()[0] if norm(x.get('prediction','')) else ''
 yes_tp=sum(cls(x)=='yes' and norm(x.get('ground_truth',''))=='yes' for x in data); yes_fp=sum(cls(x)=='yes' and norm(x.get('ground_truth',''))=='no' for x in data); yes_fn=sum(cls(x)=='no' and norm(x.get('ground_truth',''))=='yes' for x in data)
 prec=yes_tp/(yes_tp+yes_fp) if yes_tp+yes_fp else 0.0; rec=yes_tp/(yes_tp+yes_fn) if yes_tp+yes_fn else 0.0; f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
 summary.append({'file':f.name,'method':data[0].get('method'),'condition':data[0].get('condition'),'n':len(data),'accuracy_all':sum(correct(x) for x in data)/len(data),'yes_precision':prec,'yes_recall':rec,'yes_f1':f1,'visual_dependent_n':len(vis),'visual_dependent_accuracy':sum(correct(x) for x in vis)/len(vis) if vis else None,'mean_latency_sec':sum(float(x.get('latency_sec',0)) for x in data)/len(data)})
if summary:
 with (out/'h3_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(out/'h3_metrics.json').write_text(json.dumps(summary,indent=2));print('Blind-solvable',len(ids),'/',len(blind));print('Wrote',out)
