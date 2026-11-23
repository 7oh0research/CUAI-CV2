import argparse,json,os,time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--model',required=True); p.add_argument('--condition',choices=['blind','original','compressed'],required=True); p.add_argument('--out',required=True); p.add_argument('--limit',type=int,default=0); p.add_argument('--max-new-tokens',type=int,default=32); a=p.parse_args()
rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]; rows=rows[:a.limit] if a.limit else rows
revision=os.getenv('HF_MODEL_REVISION','a272c74'); proc=AutoProcessor.from_pretrained(a.model,revision=revision); cfg=None
if a.condition=='compressed': cfg={'use_fastv':True,'fastv_k':int(os.getenv('FASTV_K','2')),'fastv_r':float(os.getenv('FASTV_R','0.8888889')),'image_token_start_index':int(os.getenv('FASTV_IMAGE_TOKEN_START','5')),'image_token_length':576}
model=LlavaForConditionalGeneration.from_pretrained(a.model,revision=revision,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',attn_implementation='eager',fastv_config=cfg).eval()
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); done=set()
if out.exists():
 for x in out.read_text().splitlines():
  try: done.add(str(json.loads(x)['question_id']))
  except: pass
with out.open('a',encoding='utf-8') as f:
 for i,r in enumerate(rows,1):
  qid=str(r['question_id'])
  if qid in done: continue
  useimg=a.condition!='blind'; prompt=f"USER: {'<image>' if useimg else ''}\n{r['question']}\nAnswer with a short answer. ASSISTANT:"
  im=None
  if useimg:
   if not r.get('image') or not Path(r['image']).is_file(): raise FileNotFoundError(r.get('image'))
   im=Image.open(r['image']).convert('RGB')
  ins=proc(text=prompt,images=im,return_tensors='pt') if im else proc(text=prompt,return_tensors='pt'); ins={k:v.to(model.device) if hasattr(v,'to') else v for k,v in ins.items()}
  t=time.perf_counter()
  with torch.inference_mode(): ids=model.generate(**ins,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
  pred=proc.batch_decode(ids[:,ins['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
  f.write(json.dumps({**r,'method':'fastv','condition':a.condition,'prediction':pred,'latency_sec':time.perf_counter()-t,'token_budget':64 if a.condition=='compressed' else None},ensure_ascii=False)+'\n'); f.flush()
 print(f'{i}/{len(rows)}')
print('Saved',out)
