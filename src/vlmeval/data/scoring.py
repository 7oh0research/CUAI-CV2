"""Answer normalisation and 0/1 correctness (exact match against any gold answer)."""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().strip().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def first_yes_no(s: str) -> str:
    w = normalize(s).split()
    return w[0] if w and w[0] in ("yes", "no") else normalize(s)


def is_correct(pred: str, gold: list[str], kind: str) -> int:
    if kind == "yesno":
        return int(first_yes_no(pred) in {normalize(g) for g in gold})
    return int(normalize(pred) in {normalize(g) for g in gold})
