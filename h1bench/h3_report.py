"""Compute H3 Score A (all questions) and Score B (blind-unsolvable only)."""
import argparse, csv, json
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('--blind', required=True); p.add_argument('--results', required=True); p.add_argument('--out', default='results/h3')
a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def read(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip(): rows.append(json.loads(line))
    return {r['id']: r for r in rows}

blind = read(a.blind)
blind_solvable = {k for k,v in blind.items() if v.get('correct') is True}
visual_files = sorted(Path(a.results).glob('*_visual.jsonl'))
summary=[]; blind_rows=[]
for k,v in blind.items():
    blind_rows.append({'id':k,'dataset':v.get('dataset'),'gold':v.get('gold'),'blind_prediction':v.get('prediction'),'blind_correct':v.get('correct')})
with (out/'blind_predictions.csv').open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=['id','dataset','gold','blind_prediction','blind_correct']); w.writeheader(); w.writerows(blind_rows)
for path in visual_files:
    method=path.name.replace('_visual.jsonl',''); rows=read(path)
    for dataset in sorted({v.get('dataset','unknown') for v in rows.values()}):
        ds=[v for v in rows.values() if v.get('dataset','unknown')==dataset]
        valid=[v for v in ds if v.get('status') not in ('error','oom')]
        score_a=sum(bool(v.get('correct')) for v in valid)/len(valid) if valid else None
        visual=[v for v in valid if v['id'] not in blind_solvable]
        score_b=sum(bool(v.get('correct')) for v in visual)/len(visual) if visual else None
        tp=sum(v.get('prediction','').strip().lower().startswith('yes') and v.get('gold','').strip().lower().startswith('yes') for v in valid)
        fp=sum(v.get('prediction','').strip().lower().startswith('yes') and not v.get('gold','').strip().lower().startswith('yes') for v in valid)
        fn=sum(not v.get('prediction','').strip().lower().startswith('yes') and v.get('gold','').strip().lower().startswith('yes') for v in valid)
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        summary.append({'method':method,'dataset':dataset,'n_total':len(ds),'n_valid':len(valid),
          'n_blind_solvable':sum(v['id'] in blind_solvable for v in ds), 'n_vision_dependent':len(visual),
          'score_a_accuracy':score_a,'score_b_accuracy':score_b,'score_a_precision_yes':precision,
          'score_a_recall_yes':recall,'score_a_f1_yes':f1,'errors':len(ds)-len(valid)})
fields=list(summary[0]) if summary else ['method','dataset']
with (out/'h3_scores.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
unsupported = sorted(x.name.replace('.unsupported','') for x in Path(a.results).glob('*.unsupported'))
report={'blind_items':len(blind),'blind_solvable_items':len(blind_solvable),'vision_dependent_items':len(blind)-len(blind_solvable),
        'methods_evaluated':[p.name.replace('_visual.jsonl','') for p in visual_files], 'unsupported_or_failed_methods':unsupported, 'scores':'h3_scores.csv'}
(out/'h3_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
