"""벤치마크 x 조건(image/blank/noimage)으로 vanilla 모델을 실행하고 문항별 결과를 JSONL로 저장한다."""
import argparse, json, os, platform, subprocess, time
import yaml, torch
from tqdm import tqdm
from .data import load_benchmark
from .prompts import build_text
from .scoring import score
from .model import Vlm, blank_like

def done_ids(path):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            try: s.add(json.loads(line)["id"])
            except Exception: pass
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run-name", default="full")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_dir = os.path.join("results", a.run_name); raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    vlm = Vlm(cfg["model_id"], cfg["dtype"], cfg["attn_implementation"])
    meta = {"config": cfg, "max_samples": a.max_samples, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "attn_used": vlm.attn_used, "python": platform.python_version()}
    try:
        import transformers; meta["transformers"] = transformers.__version__
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: pass
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2)
    print(json.dumps({k: meta[k] for k in ("gpu", "torch", "cuda", "attn_used")}))

    names = a.benchmarks or list(cfg["benchmarks"])
    bs = cfg["batch_size"]
    for name in names:
        b = cfg["benchmarks"][name]
        items = load_benchmark(name, b, a.max_samples, cfg["seed"])
        print(f"[{name}] {len(items)} items")
        for cond in cfg["conditions"]:
            path = os.path.join(raw, f"{name}__{cond}.jsonl")
            done = done_ids(path)
            todo = [it for it in items if it["id"] not in done]
            if not todo:
                print(f"  {cond}: 완료됨, 건너뜀"); continue
            with open(path, "a") as f:
                for i in tqdm(range(0, len(todo), bs), desc=f"{name}/{cond}"):
                    batch = todo[i:i + bs]
                    texts = [build_text(it) for it in batch]
                    if cond == "image":
                        imgs = [it["image"] for it in batch]
                    elif cond == "blank":
                        imgs = [blank_like(it["image"], cfg["blank_gray_value"]) for it in batch]
                    else:
                        imgs = None
                    outs = vlm.generate(texts, imgs, b["max_new_tokens"])
                    for it, o in zip(batch, outs):
                        pred, sc, ch = score(it, o)
                        rec = {"id": it["id"], "bench": name, "cond": cond, "type": it["type"],
                               "category": it["category"], "raw": o, "pred": pred, "score": sc,
                               "chance": ch, "gold": it.get("answer", it.get("answers"))}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
    print("실행 완료")

if __name__ == "__main__":
    main()
