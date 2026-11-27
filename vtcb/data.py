"""Benchmark adapters -> one common item format.

Every adapter returns a dict:
  id, image (PIL or None), question (full prompt text incl. options and answer-format
  instruction), answer (str), answers (list|None), kind in {yesno, mc, vqa, relaxed, ocr, exact},
  n_choices (int|None), category (str), max_new_tokens (int)

Field names were written for the HF datasets in BENCHES but were NOT verified against the
live datasets. Fields are indexed with [] on purpose: a schema mismatch raises KeyError
instead of silently producing empty prompts. Run scripts/inspect_datasets.py first.
"""
import random
from datasets import load_dataset

SA = "Answer the question using a single word or phrase."
MC = "Answer with the option's letter from the given choices directly."
LET = "ABCDEFGHIJ"

BENCHES = {  # name: (hf path, config, split)
    "pope":        ("lmms-lab/POPE", None, "test"),
    "mme":         ("lmms-lab/MME", None, "test"),
    "textvqa":     ("lmms-lab/textvqa", None, "validation"),
    "scienceqa":   ("lmms-lab/ScienceQA", "ScienceQA-IMG", "test"),
    "mmbench":     ("lmms-lab/MMBench", "en", "dev"),
    "gqa":         ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
    "mmstar":      ("Lin-Chen/MMStar", None, "val"),
    "ai2d":        ("lmms-lab/ai2d", None, "test"),
    "seedbench":   ("lmms-lab/SEED-Bench", None, "test"),
    "chartqa":     ("lmms-lab/ChartQA", None, "test"),
    "ocrbench":    ("echo840/OCRBench", None, "test"),
    "realworldqa": ("lmms-lab/RealWorldQA", None, "test"),
    "vizwiz":      ("lmms-lab/VizWiz-VQA", None, "val"),
}
GQA_IMAGES = ("lmms-lab/GQA", "testdev_balanced_images", "testdev")

def _mc(q, opts, hint=None):
    ctx = f"Context: {hint}\n" if hint else ""
    return f"{ctx}{q}\n" + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts)) + f"\n{MC}"

def _item(id_, img, q, ans, kind, cat="", answers=None, n=None, mnt=16):
    return dict(id=str(id_), image=img, question=q, answer=str(ans), answers=answers,
                kind=kind, n_choices=n, category=str(cat), max_new_tokens=mnt)

def _rgb(im):
    return None if im is None else im.convert("RGB")

def _ok(v):
    return v not in (None, "", "nan") and not (isinstance(v, float) and v != v)

def adapt(name, r, ctx):
    i = ctx["i"]
    if name == "pope":
        return _item(r["question_id"], _rgb(r["image"]), f"{r['question']}\n{SA}",
                     r["answer"].strip().lower(), "yesno", r["category"], n=2)
    if name == "mme":  # question already ends with "Please answer yes or no."
        return _item(f"{r['question_id']}|{i}", _rgb(r["image"]), r["question"],
                     r["answer"].strip().lower(), "yesno", r["category"], n=2)
    if name == "textvqa":
        return _item(r["question_id"], _rgb(r["image"]), f"{r['question']}\n{SA}",
                     r["answers"][0], "vqa", "", answers=list(r["answers"]))
    if name == "scienceqa":
        opts = list(r["choices"]); hint = r["hint"] if _ok(r["hint"]) else None
        return _item(i, _rgb(r["image"]), _mc(r["question"], opts, hint),
                     LET[int(r["answer"])], "mc", r["category"], n=len(opts))
    if name == "mmbench":
        opts = [r[c] for c in "ABCD" if _ok(r[c])]
        hint = r["hint"] if _ok(r["hint"]) else None
        return _item(r["index"], _rgb(r["image"]), _mc(r["question"], opts, hint),
                     r["answer"], "mc", r["category"], n=len(opts))
    if name == "gqa":
        return _item(r["id"], _rgb(ctx["gqa_images"][r["imageId"]]), f"{r['question']}\n{SA}",
                     r["answer"], "exact", "")
    if name == "mmstar":  # options are already in the question text
        return _item(r["index"], _rgb(r["image"]), f"{r['question']}\n{MC}",
                     r["answer"].strip(), "mc", r["category"], n=4)
    if name == "ai2d":
        opts = list(r["options"])
        return _item(i, _rgb(r["image"]), _mc(r["question"], opts),
                     LET[int(r["answer"])], "mc", "", n=len(opts))
    if name == "seedbench":
        if r["data_type"] != "image":
            return None
        img = r["image"][0] if isinstance(r["image"], list) else r["image"]
        opts = [r[f"choice_{c}"] for c in "abcd"]
        return _item(r["question_id"], _rgb(img), _mc(r["question"], opts),
                     r["answer"], "mc", r["question_type_id"], n=4)
    if name == "chartqa":
        return _item(i, _rgb(r["image"]), f"{r['question']}\n{SA}", r["answer"], "relaxed", r["type"])
    if name == "ocrbench":
        ans = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
        return _item(i, _rgb(r["image"]), r["question"], ans[0], "ocr",
                     r["question_type"], answers=list(ans), mnt=32)
    if name == "realworldqa":  # question already has its own format instruction
        a = r["answer"].strip()
        kind = "mc" if len(a) == 1 and a in LET else "exact"
        return _item(i, _rgb(r["image"]), r["question"], a, kind, "", n=(4 if kind == "mc" else None))
    if name == "vizwiz":
        q = (f"{r['question']}\nWhen the provided information is insufficient, "
             f"respond with 'Unanswerable'.\n{SA}")
        return _item(r["question_id"], _rgb(r["image"]), q, r["answers"][0], "vqa",
                     r["category"], answers=list(r["answers"]))
    raise ValueError(name)

def iter_items(name, max_samples=None, seed=0):
    path, cfg, split = BENCHES[name]
    ds = load_dataset(path, cfg, split=split)
    idx = list(range(len(ds)))
    if name == "seedbench":  # filter to image questions before sampling
        idx = [j for j, t in enumerate(ds["data_type"]) if t == "image"]
    if max_samples and max_samples < len(idx):
        idx = sorted(random.Random(seed).sample(idx, max_samples))
    ctx = {}
    if name == "gqa":
        im = load_dataset(*GQA_IMAGES[:2], split=GQA_IMAGES[2])
        ctx["gqa_images"] = dict(zip(im["id"], range(len(im))))
        ctx["gqa_images"] = _LazyImages(im, ctx["gqa_images"])
    seen = set()
    for j in idx:            # generator: images are decoded one item at a time
        ctx["i"] = j
        it = adapt(name, ds[j], ctx)
        if it is None:
            continue
        if it["id"] in seen:
            it["id"] = f"{it['id']}#{j}"
        seen.add(it["id"])
        yield it

class _LazyImages:
    def __init__(self, ds, index): self.ds, self.index = ds, index
    def __getitem__(self, k): return self.ds[self.index[k]]["image"]
