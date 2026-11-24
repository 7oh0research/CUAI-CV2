import re

_ARTICLES = {"a", "an", "the"}
_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[\"'`.,!?;:()\[\]]", " ", s)
    toks = [_NUM.get(t, t) for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def parse_yesno(out):
    t = out.strip().lower()
    m = re.match(r"^\W*(yes|no)\b", t)
    if m:
        return m.group(1)
    has_y, has_n = re.search(r"\byes\b", t), re.search(r"\bno\b", t)
    if has_y and not has_n: return "yes"
    if has_n and not has_y: return "no"
    return "other"

def parse_mc(out, letters):
    t = out.strip()
    m = re.match(r"^\W*\(?([A-H])\)?(?:[\.\):\s]|$)", t)
    if m and m.group(1) in letters:
        return m.group(1)
    m = re.search(r"\b([A-H])\b", t)
    if m and m.group(1) in letters:
        return m.group(1)
    return "other"

def vqa_score(pred, answers):
    p = norm(pred)
    matches = sum(1 for a in answers if norm(a) == p)
    return min(matches / 3.0, 1.0)

def score(it, out):
    """(파싱된 답, 점수 0~1, 문항별 우연 정답 확률)"""
    if it["type"] == "yesno":
        p = parse_yesno(out)
        return p, float(p == it["answer"]), 0.5
    if it["type"] == "mc":
        p = parse_mc(out, it["letters"])
        return p, float(p == it["answer"]), 1.0 / max(len(it["letters"]), 1)
    p = out.strip()
    return p, vqa_score(p, it["answers"]), 0.0
