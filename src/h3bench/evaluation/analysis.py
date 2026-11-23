from collections import defaultdict

def correct(row):
    if isinstance(row.get('is_correct'), bool): return row['is_correct']
    if 'prediction' not in row or 'ground_truth' not in row: return None
    norm=lambda x: ' '.join(str(x).strip().lower().split()).rstrip(' .,!?:;')
    return norm(row['prediction']) == norm(row['ground_truth'])

def build(blind_rows, prediction_rows):
    blind_map={r['question_id']:correct(r) for r in blind_rows}
    if any(v is None for v in blind_map.values()):
        raise ValueError('Blind rows need is_correct or prediction + ground_truth')
    blind_ids=set(blind_map)
    visual_ids={qid for qid,ok in blind_map.items() if not ok}
    grouped=defaultdict(list)
    for r in prediction_rows:
        if r.get('condition') not in ('original','compressed'): continue
        grouped[(r.get('dataset','unknown'),r.get('method','unknown'),r.get('condition','unknown'))].append(r)
    summaries=[]
    for (dataset,method,condition),rows in sorted(grouped.items()):
        by_id={r['question_id']:r for r in rows}
        ids=blind_ids.intersection(by_id)
        all_ok=[correct(by_id[q]) for q in ids]
        vis_ok=[correct(by_id[q]) for q in ids.intersection(visual_ids)]
        if any(x is None for x in all_ok+vis_ok):
            raise ValueError(f'Missing correctness labels for {dataset}/{method}/{condition}')
        summaries.append({
            'dataset':dataset,'method':method,'condition':condition,
            'n_common':len(ids),'accuracy_all':sum(all_ok)/len(all_ok) if all_ok else None,
            'n_visual_dependent':len(vis_ok),'accuracy_visual_dependent':sum(vis_ok)/len(vis_ok) if vis_ok else None,
            'blind_solvable_n':sum(blind_map[q] for q in ids),
            'blind_solvable_ratio':sum(blind_map[q] for q in ids)/len(ids) if ids else None,
        })
    return summaries
