import re

_YES = re.compile(r"\byes\b"); _NO = re.compile(r"\bno\b")

def parse_yesno(s):
    t = s.strip().lower(); y, n = _YES.search(t), _NO.search(t)
    if y and n:
        return "yes" if y.start() < n.start() else "no"
    return "yes" if y else "no" if n else None

def parse_mc(s, n=None):
    t = s.strip(); L = "ABCDEFGHIJ"[: n or 10]
    m = re.match(rf"^\(?([{L}])(?:[\).:\s]|$)", t)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{L}])\b", t)
    return m.group(1) if m else None

_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ART = {"a", "an", "the"}

def norm(s):
    s = s.lower().replace("\n", " ").strip().rstrip(".")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(_NUM.get(x, x) for x in s.split() if x not in _ART)

def vqa_score(pred, answers):
    """Simplified official VQA accuracy (leave-one-annotator-out, min(1, matches/3))."""
    p = norm(pred); a = [norm(x) for x in answers]
    if len(a) == 1:
        return float(p == a[0])
    return sum(min(1.0, sum(o == p for o in a[:i] + a[i + 1:]) / 3) for i in range(len(a))) / len(a)

def relaxed(pred, gt, tol=0.05):
    p = pred.strip().rstrip(".").replace("%", "").replace(",", "")
    g = gt.strip().replace("%", "").replace(",", "")
    try:
        pf, gf = float(p), float(g)
        return float(abs(pf - gf) <= tol * abs(gf)) if gf != 0 else float(pf == gf)
    except ValueError:
        return float(norm(p) == norm(g))

def ocr_score(pred, answers):
    p = pred.lower().strip(); ps = re.sub(r"\s+", "", p)
    return float(any(a.lower().strip() in p or re.sub(r"\s+", "", a.lower()) in ps for a in answers))

def score(item, raw):
    """-> (score in [0,1], parsed, parse_ok)"""
    k = item["kind"]
    if k == "yesno":
        p = parse_yesno(raw); return float(p == item["answer"]), p, p is not None
    if k == "mc":
        p = parse_mc(raw, item["n_choices"]); return float(p == item["answer"]), p, p is not None
    if k == "vqa":
        return vqa_score(raw, item["answers"]), raw.strip(), True
    if k == "relaxed":
        return relaxed(raw, item["answer"]), raw.strip(), True
    if k == "ocr":
        return ocr_score(raw, item["answers"]), raw.strip(), True
    if k == "exact":
        return float(norm(raw) == norm(item["answer"])), raw.strip(), True
    raise ValueError(k)

def chance(item):
    return 0.5 if item["kind"] == "yesno" else (1.0 / item["n_choices"] if item["kind"] == "mc" else 0.0)
