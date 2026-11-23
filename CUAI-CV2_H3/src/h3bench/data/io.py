import json
from pathlib import Path

def read_jsonl(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f'{path}:{n}: invalid JSON: {e}')
            if 'question_id' not in row: raise ValueError(f'{path}:{n}: question_id missing')
            rows.append(row)
    return rows

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
