"""벤치마크를 공통 형식으로 읽는다. 필드가 없으면 조용히 넘어가지 않고 오류를 낸다."""
import random
from datasets import load_dataset

def _get(ex, *keys, default=None, required=True):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    if required:
        raise KeyError(f"필드 {keys} 없음. 실제 필드: {list(ex.keys())}")
    return default

def _valid(v):
    return v is not None and str(v).strip() not in ("", "nan", "None")

def to_item(name, btype, ex, idx):
    img = _get(ex, "image", "decoded_image")
    if isinstance(img, list):
        img = img[0]
    it = {"id": str(_get(ex, "question_id", "index", "id", default=idx, required=False)),
          "bench": name, "type": btype, "image": img,
          "category": str(_get(ex, "category", "subject", "l2-category", default="all", required=False))}
    q = str(_get(ex, "question")).strip()
    if btype == "yesno":
        it["question"] = q
        it["answer"] = str(_get(ex, "answer")).strip().lower()
    elif btype == "open":
        it["question"] = q
        it["answers"] = [str(a).strip().lower() for a in _get(ex, "answers")]
    elif btype == "mc":
        hint = _get(ex, "hint", default="", required=False)
        if "choices" in ex and ex["choices"] is not None:          # ScienceQA 형식
            choices = list(ex["choices"])
            ans = int(_get(ex, "answer"))
            letters = [chr(65 + i) for i in range(len(choices))]
            answer = letters[ans]
        else:                                                       # MMBench 형식 (A,B,C,D 열)
            letters, choices = [], []
            for L in "ABCDEFGH":
                if L in ex and _valid(ex[L]):
                    letters.append(L); choices.append(str(ex[L]))
            answer = str(_get(ex, "answer")).strip().upper()
        it["question"] = q
        it["hint"] = str(hint) if _valid(hint) else ""
        it["choices"] = choices
        it["letters"] = letters
        it["answer"] = answer
    return it

def load_benchmark(name, b, max_samples=None, seed=0):
    ds = load_dataset(b["hf_id"], b.get("hf_config"), split=b["split"])
    idx = list(range(len(ds)))
    if max_samples and max_samples < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:max_samples])
    items, seen = [], set()
    for i in idx:
        it = to_item(name, b["type"], ds[i], i)
        if it["id"] in seen:                    # id 중복 시 위치 번호를 붙여 고유화
            it["id"] = f"{it['id']}__{i}"
        seen.add(it["id"])
        items.append(it)
    return items
