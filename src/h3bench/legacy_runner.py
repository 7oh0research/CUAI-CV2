import argparse,json,os,sys,time
from pathlib import Path
from PIL import Image
import torch
p=argparse.ArgumentParser(); p.add_argument('--method',choices=['sparsevlm','visionzip'],required=True); p.add_argument('--condition',choices=['blind','original','compressed'],required=True); p.add_argument('--manifest',required=True); p.add_argument('--model',required=True); p.add_argument('--out',required=True); p.add_argument('--limit',type=int,default=0); p.add_argument('--max-new-tokens',type=int,default=32); a=p.parse_args()
root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/'third_party'/( 'sparsevlm' if a.method=='sparsevlm' else 'llava')))
if a.method=='visionzip': sys.path.insert(0,str(root/'third_party/visionzip'))
from llava.constants import IMAGE_TOKEN_INDEX,DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token,process_images
from llava.utils import disable_torch_init
disable_torch_init(); tok,model,iproc,_=load_pretrained_model(a.model,None,Path(a.model).name,device='cuda')
if a.method=='visionzip' and a.condition=='compressed':
 from visionzip import visionzip
 model=visionzip(model,dominant=int(os.getenv('VISIONZIP_DOMINANT','54')),contextual=int(os.getenv('VISIONZIP_CONTEXTUAL','10')))
model.eval(); rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]; rows=rows[:a.limit] if a.limit else rows
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True); done=set()
if out.exists():
 for x in out.read_text().splitlines():
  try:done.add(str(json.loads(x)['question_id']))
  except:pass
with out.open('a',encoding='utf-8') as f:
 for i,r in enumerate(rows,1):
  qid=str(r['question_id'])
  if qid in done:continue
  useimg=a.condition!='blind'; conv=conv_templates['llava_v1'].copy(); q=(DEFAULT_IMAGE_TOKEN+'\n' if useimg else '')+r['question']+'\nAnswer with a short answer.';conv.append_message(conv.roles[0],q);conv.append_message(conv.roles[1],None)
  ids=tokenizer_image_token(conv.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda(); imgs=None
  if useimg:
   if not r.get('image') or not Path(r['image']).is_file():raise FileNotFoundError(r.get('image'))
   imgs=process_images([Image.open(r['image']).convert('RGB')],iproc,model.config).to(device='cuda',dtype=torch.float16)
  if a.method=='sparsevlm' and a.condition=='compressed':os.environ['RETAIN_TOKN']=str(int(os.getenv('TOKEN_BUDGET','64')))
  t=time.perf_counter()
  with torch.inference_mode(): outids=model.generate(ids,images=imgs,do_sample=False,temperature=0,max_new_tokens=a.max_new_tokens,use_cache=True)
  pred=tok.batch_decode(outids[:,ids.shape[1]:],skip_special_tokens=True)[0].strip()
  f.write(json.dumps({**r,'method':a.method,'condition':a.condition,'prediction':pred,'latency_sec':time.perf_counter()-t,'token_budget':64 if a.condition=='compressed' else None},ensure_ascii=False)+'\n');f.flush()
  if i%10==0:print(a.method,a.condition,i,'/',len(rows),flush=True)
print('Saved',out)
