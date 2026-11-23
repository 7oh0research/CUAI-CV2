"""Synthetic per-sample records so the analysis pipeline can be developed without a GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BASELINE, BLIND

# (accuracy drop per unit ratio, latency efficiency: fraction of ideal speed-up realised)
_METHODS = {"random": (0.30, 0.95), "avgpool": (0.18, 0.95), "cls_attn": (0.10, 0.60)}


def make_records(n: int = 300, ratios=(0.5, 0.75, 0.9), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for bench, base_acc, blind_acc in [("pope", 0.86, 0.5), ("textvqa", 0.58, 0.05)]:
        for i in range(n):
            sid = f"{bench}-{i}"
            blind_ok = int(rng.random() < blind_acc)
            ok = int(blind_ok or rng.random() < base_acc)
            base_ms = 200.0 + rng.normal(0, 5)

            def add(method, ratio, correct, pred, ms, toks):
                rows.append(dict(benchmark=bench, method=method, ratio=ratio, sample_id=sid,
                                 gold=["yes"], pred=pred, correct=correct, n_visual_tokens=toks,
                                 prefill_ms=ms * 0.8, total_ms=ms, peak_mem_mb=15000.0))

            add(BASELINE, 0.0, ok, "yes" if ok else "no", base_ms, 576)
            add(BLIND, 0.0, blind_ok, "yes" if blind_ok else "no", base_ms * 0.3, 0)
            for m, (drop, eff) in _METHODS.items():
                for r in ratios:
                    # Only visually dependent items are hurt by compression.
                    hurt = ok and not blind_ok and rng.random() < drop * r
                    c = 0 if hurt else ok
                    ms = base_ms * (1 - eff * r * 0.8) + rng.normal(0, 3)
                    add(m, r, c, "yes" if c else "no", ms, int(576 * (1 - r)))
    return pd.DataFrame(rows)
