"""MME 전체 채점용 답변 생성(R=32, bs=1). 출력은 공식 calculation.py 입력 형식(subtask별 txt)."""
import argparse, os, torch, yaml
from h1bench.run_one import build_model, build_batch
from h1bench.adapters import ADAPTERS
from h1bench.data import load_mme
ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--attn"); ap.add_argument("--R", type=int, default=32)
a = ap.parse_args()
cfg = yaml.safe_load(open("configs/methods.yaml"))
tok, model, proc = build_model(cfg["model_path"], a.attn)
ADAPTERS[a.method](model, a.R, cfg["methods"][a.method])
out_dir = f"results/mme/{a.method}"; os.makedirs(out_dir, exist_ok=True)
files = {}
for s in load_mme(os.environ["MME_ROOT"]):
    q = s["question"] + " Answer the question using a single word or phrase."
    inp, att, pix, sizes, _, _ = build_batch(tok, model, proc, [dict(s, question=q)])
    with torch.inference_mode():
        o = model.generate(inp, attention_mask=att, images=pix, image_sizes=sizes, do_sample=False, max_new_tokens=16, use_cache=True)
    ans = tok.decode(o[0], skip_special_tokens=True).strip().replace("\n", " ")
    f = files.setdefault(s["subtask"], open(f"{out_dir}/{s['subtask']}.txt", "w"))
    f.write(f"{os.path.basename(s['image'])}\t{s['question']}\t{s['answer']}\t{ans}\n")
for f in files.values(): f.close()
print(out_dir)
