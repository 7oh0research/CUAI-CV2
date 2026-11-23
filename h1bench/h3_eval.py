"""H3 evaluation: blind baseline filtering, then visual inference for baseline and compression methods."""
import argparse, json, os, re, sys, time, traceback
from pathlib import Path
import torch, yaml
from PIL import Image
from h1bench.adapters import ADAPTERS
from h1bench.h3_data import load_h3_data


def normalize_yesno(text):
    s = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    words = s.split()
    for w in words[:8]:
        if w in ("yes", "no"):
            return w
    return "unknown"


def is_correct(pred, gold):
    p, g = normalize_yesno(pred), normalize_yesno(gold)
    if p != "unknown" and g != "unknown": return p == g
    return pred.strip().lower() == gold.strip().lower()


def prompt_for(question, tokenizer, with_image):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    conv = conv_templates["vicuna_v1"].copy()
    user = ((DEFAULT_IMAGE_TOKEN + "\n") if with_image else "") + question + " Answer the question using a single word: yes or no."
    conv.append_message(conv.roles[0], user); conv.append_message(conv.roles[1], None)
    if with_image:
        return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    return tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.cuda()


def load_model(model_path, attn):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    name = get_model_name_from_path(model_path)
    tok, model, proc, _ = load_pretrained_model(model_path, None, name, device_map="cuda:0",
        torch_dtype=torch.float16, attn_implementation=attn)
    model.eval()
    return tok, model, proc


def infer_one(tok, model, proc, sample, with_image, max_new_tokens):
    ids = prompt_for(sample["question"], tok, with_image)
    kwargs = {"input_ids": ids, "do_sample": False, "max_new_tokens": max_new_tokens, "use_cache": True}
    if with_image:
        image = Image.open(sample["image"]).convert("RGB")
        from llava.mm_utils import process_images
        pix = process_images([image], proc, model.config).to(device="cuda", dtype=torch.float16)
        kwargs.update(images=pix, image_sizes=[image.size])
    with torch.inference_mode():
        out = model.generate(**kwargs)
    new_ids = out[0, ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--mode", choices=["blind", "visual"], required=True)
    ap.add_argument("--config", default="configs/h3_only.yaml")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8")); exp = cfg["experiment"]
    data = load_h3_data(os.environ["MME_ROOT"], os.environ.get("POPE_ROOT"), os.environ.get("POPE_FILE"),
                        exp.get("max_mme_items", 0), exp.get("max_pope_items", 0), exp.get("seed", 42))
    tok, model, proc = load_model(cfg["model_path"], cfg.get("attention", "sdpa"))
    if a.mode == "visual":
        mcfg = yaml.safe_load(open("configs/methods.yaml", encoding="utf-8"))["methods"][a.method]
        ADAPTERS[a.method](model, cfg["methods"][a.method]["R"], mcfg)
    out = Path(a.output)
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    else: done = set()
    rows = []
    started = time.time()
    for idx, s in enumerate(data, 1):
        if s["id"] in done: continue
        try:
            pred = infer_one(tok, model, proc, s, a.mode == "visual", exp["max_new_tokens"])
            rows.append({"id": s["id"], "dataset": s["dataset"], "question": s["question"],
                         "gold": s["answer"], "prediction": pred, "correct": is_correct(pred, s["answer"]),
                         "mode": a.mode, "method": a.method})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"id": s["id"], "dataset": s["dataset"], "gold": s["answer"], "prediction": "",
                         "correct": False, "status": "oom", "mode": a.mode, "method": a.method})
        except Exception as e:
            rows.append({"id": s["id"], "dataset": s["dataset"], "gold": s["answer"], "prediction": "",
                         "correct": False, "status": "error", "error": repr(e), "mode": a.mode, "method": a.method})
        if len(rows) >= 25:
            with out.open("a", encoding="utf-8") as f:
                for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.clear()
        if idx % 100 == 0: print(f"[{a.method}/{a.mode}] {idx}/{len(data)} elapsed={time.time()-started:.1f}s", flush=True)
    if rows:
        with out.open("a", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] {a.method}/{a.mode}: {out}")

if __name__ == "__main__":
    main()
