"""Benchmark adapters (HuggingFace `lmms-lab/*` mirrors) -> a uniform Sample.

Dataset ids and column names below follow the lmms-lab mirrors and must be verified on the
first run (`python scripts/run_experiment.py --dry-run`). Subsets are drawn once with a
fixed seed so every method sees exactly the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

SHORT = "\nAnswer the question using a single word or phrase."


@dataclass
class Sample:
    sample_id: str
    image: object            # PIL.Image
    question: str            # full prompt text (without the USER/ASSISTANT template)
    gold: list[str]
    kind: str                # "yesno" | "open"


def _subset(ds, n: int, seed: int):
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    return ds.select(sorted(int(i) for i in idx))


def _pope(row, i):
    return Sample(str(row.get("question_id", i)), row["image"], row["question"] + SHORT,
                  [str(row["answer"])], "yesno")


def _textvqa(row, i):
    return Sample(str(row.get("question_id", i)), row["image"], row["question"] + SHORT,
                  [str(a) for a in row["answers"]], "open")


def _mme(row, i):
    q = row["question"]
    return Sample(str(row.get("question_id", i)), row["image"], q, [str(row["answer"])], "yesno")


# name -> (HF dataset id, row adapter)
BENCHMARKS = {
    "pope": ("lmms-lab/POPE", _pope),
    "textvqa": ("lmms-lab/textvqa", _textvqa),
    "mme": ("lmms-lab/MME", _mme),
    # "gqa": TODO - lmms-lab/GQA keeps questions and images in separate configs.
}


def load_benchmark(name: str, split: str, n_samples: int, seed: int) -> Iterator[Sample]:
    from datasets import load_dataset

    hf_id, adapt = BENCHMARKS[name]
    ds = _subset(load_dataset(hf_id, split=split), n_samples, seed)
    for i, row in enumerate(ds):
        s = adapt(row, i)
        s.sample_id = f"{name}-{s.sample_id}-{i}"  # unique even if the source id repeats
        s.image = s.image.convert("RGB")
        yield s
