"""Synthetic H1 records to exercise the analysis pipeline without a GPU.

FLOPs are the real analytic values for LLaVA-1.5-7B; the *latencies are simulated* from a made-up cost model
(compute-bound prefill, weight-bound decode, a fixed overhead per in-LLM pruning stage, an eager-attention
penalty). They carry NO empirical meaning: use them for tests and for checking the report layout only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import flops as fl
from .methods import BUILTIN
from .profiler import theoretical
from .schema import COLUMNS


def make_synthetic(seed: int = 0, repeats: int = 8, max_new: int = 64, batch_sizes=(1, 4, 16),
                   budgets=(192, 128, 64), attn_impls=("sdpa", "eager"),
                   methods=("random", "avgpool", "cls_attn", "fastv", "pdrop", "sparsevlm_lite")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d, v = fl.LLAVA15_7B, fl.CLIP_L14_336
    rows = []
    combos = [("none", 576)] + [(m, b) for m in methods for b in budgets]
    for impl in attn_impls:
        for bs in batch_sizes:
            for name, budget in combos:
                spec = BUILTIN[name]
                th = theoretical(d, v, spec, budget, 576, 40, impl, 1 if spec.scorer == "last" else 12)
                stages = th.pop("stages")
                util = min(1.0, 0.35 + 0.15 * np.log2(bs + 1))
                pre = bs * th["prefill_flops"] / (60e12 * util) * 1e3
                pre *= 1.5 if impl == "eager" else 1.0
                pre += len(stages) * (2.0 if spec.scorer == "last" else 4.0)
                vis = 10.0 + 6.0 * bs
                dec0 = 24.0 + (3.5 * len(stages) if spec.scorer == "text" else 0.0) + 0.0015 * bs * th["n_visual_in"]
                kv_mb = bs * (th["n_visual_in"] + 40 + max_new) * 32 * 2 * 4096 * 2 / 2**20
                for r in range(repeats):
                    noise = lambda: float(rng.lognormal(0, 0.02))
                    rows.append({
                        "model": "synthetic", "attn_impl": impl, "method": name, "budget": budget,
                        "batch_size": bs, "repeat": r, "status": "ok", **th, "n_text": 40, "max_new": max_new,
                        "vision_ms": vis * noise(), "prefill_ms": pre * noise(),
                        "decode_ms": [dec0 * (1 + 0.0004 * j) * noise() for j in range(max_new - 1)],
                        "peak_mem_mb": 14500 + kv_mb + (bs * 32 * 600 * 600 * 4 / 2**20 if impl == "eager" else 0),
                        "base_mem_mb": 14000.0, "kv_mb": kv_mb, "device": "synthetic", "dtype": "float16",
                        "torch_version": "n/a",
                    })
    df = pd.DataFrame(rows)
    return df.reindex(columns=COLUMNS)
