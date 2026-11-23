import argparse, json, os, time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def read_jsonl(path):
    rows=[]
    for i,line in enumerate(Path(path).open(encoding='utf-8'),1):
        if line.strip():
            r=json.loads(line)
            for k in ('question_id','question','ground_truth'):
                if k not in r: raise ValueError(f'{path}:{i}: missing {k}')
            rows.append(r)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--model',required=True)
    ap.add_argument('--condition',choices=['blind','original'],required=True)
    ap.add_argument('--revision',default=None); ap.add_argument('--method',default='vanilla'); ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--max-new-tokens',type=int,default=32)
    a=ap.parse_args()
    if a.method!='vanilla':
        raise SystemExit(f'{a.method} adapter is not wired into this baseline runner. Use the method-specific upstream adapter; refusing to label vanilla output as {a.method}.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dtype=torch.float16 if device=='cuda' else torch.float32
    print(f'Loading {a.model} on {device} ({dtype})',flush=True)
    processor=AutoProcessor.from_pretrained(a.model,revision=a.revision)
    model=LlavaForConditionalGeneration.from_pretrained(a.model,revision=a.revision,torch_dtype=dtype,low_cpu_mem_usage=True,device_map='auto' if device=='cuda' else None)
    if device=='cpu': model.to(device)
    model.eval()
    rows=read_jsonl(a.manifest)
    if a.limit: rows=rows[:a.limit]
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if out.exists():
        for line in out.open(encoding='utf-8'):
            try: done.add(json.loads(line)['question_id'])
            except Exception: pass
    with out.open('a',encoding='utf-8') as f:
        for n,r in enumerate(rows,1):
            if str(r['question_id']) in done: continue
            image=None
            if a.condition=='original':
                if not r.get('image') or not Path(r['image']).is_file():
                    raise FileNotFoundError(f"Missing image for question {r['question_id']}: {r.get('image')}")
                image=Image.open(r['image']).convert('RGB')
                prompt=f"USER: <image>\n{r['question']}\nAnswer with a short answer. ASSISTANT:"
            else:
                prompt=f"USER: {r['question']}\nAnswer with a short answer. ASSISTANT:"
            inputs=processor(text=prompt,images=image,return_tensors='pt') if image is not None else processor(text=prompt,return_tensors='pt')
            inputs={k:v.to(model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
            t=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
            latency=time.perf_counter()-t
            text=processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
            obj={'question_id':str(r['question_id']),'dataset':r.get('dataset','unknown'),'method':a.method,'condition':a.condition,'prediction':text,'ground_truth':str(r['ground_truth']),'image':r.get('image'),'latency_sec':latency,'model':a.model}
            f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush()
            if n%10==0: print(f'{n}/{len(rows)} completed',flush=True)
    print(f'Wrote predictions: {out}')
if __name__=='__main__': main()
