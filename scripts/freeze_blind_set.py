import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--blind',required=True);p.add_argument('--out',required=True);a=p.parse_args()
def norm(s):return ' '.join(re.sub(r'[^a-z0-9 ]',' ',str(s).lower()).split())
def ok(x):
 pred=norm(x.get('prediction','')).split(' ');gt=norm(x.get('ground_truth',''))
 if gt in ('yes','no'):return bool(pred) and pred[0]==gt
 return norm(x.get('prediction',''))==gt or bool(gt and gt in norm(x.get('prediction','')))
rows=[json.loads(x) for x in Path(a.blind).read_text().splitlines() if x.strip()];ids=[str(x['question_id']) for x in rows if ok(x)]
o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps({'source':'vanilla_blind','question_ids':ids},indent=2));print(len(ids),'/',len(rows),'blind-solvable')
