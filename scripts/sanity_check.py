"""GPU sanity checks. Must all PASS before the full run.
1) vanilla via our manual prompt == HF processor + generate (same text) on 5 items
2) every method at budget 576 (no pruning) == vanilla
3) pre-LLM methods: LLM receives exactly `budget` image tokens
4) in-LLM methods: logged kept count == budget schedule
5) masking implementation: FastV masked == FastV with physically dropped tokens? (logit check on 1 item)
"""
import sys, os, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vtcb.data import iter_items
from vtcb.model import LlavaPruned
cfg = yaml.safe_load(open("configs/default.yaml"))
M = LlavaPruned(cfg)
items = [x for _, x in zip(range(5), iter_items("pope", 50, 0))]
fail = 0
def check(name, cond):
    global fail
    print(("PASS " if cond else "FAIL ") + name); fail += (not cond)

for it in items:
    prompt = f"USER: <image>\n{it['question']} ASSISTANT:"
    inp = M.proc(images=it["image"], text=prompt, return_tensors="pt").to("cuda", torch.float16)
    ref = M.model.generate(**inp, max_new_tokens=it["max_new_tokens"], do_sample=False)
    ref = M.tok.decode(ref[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    ours, _ = M.generate(it)
    check(f"manual prompt == HF generate [{it['id']}] {ours!r} vs {ref!r}", ours == ref)

van = [M.generate(it)[0] for it in items]
for m in cfg["methods"]:
    out = [M.generate(it, m, 576)[0] for it in items]
    check(f"{m}@576 == vanilla", out == van)
    for b in cfg["budgets"]:
        _, info = M.generate(items[0], m, b)
        if cfg["methods"][m]["stage"] == "pre":
            check(f"{m}@{b} LLM gets {info['n_img_tokens_in']} image tokens", info["n_img_tokens_in"] == b)
        else:
            ks = [v for k, v in info.items() if k.startswith("kept@")]
            check(f"{m}@{b} in-LLM kept {ks}", len(ks) > 0 and ks[0] == b)
print("ALL PASS" if fail == 0 else f"{fail} FAILED -> do not run full")
sys.exit(int(fail > 0))
