"""Statistics: paired bootstrap CIs, exact McNemar, ranking comparison and stability."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .panel import Panel


def bootstrap_metric(
    p: Panel, fn: Callable[[Panel, np.ndarray], pd.Series], n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Percentile 95% CI per config; resamples whole samples so configs stay paired."""
    rng = np.random.default_rng(seed)
    reps = np.vstack([fn(p, rng.integers(0, p.n, p.n)).to_numpy() for _ in range(n_boot)])
    point = fn(p, np.arange(p.n))
    return pd.DataFrame(
        {
            "value": point,
            "lo": np.nanpercentile(reps, 2.5, axis=0),
            "hi": np.nanpercentile(reps, 97.5, axis=0),
        },
        index=point.index,
    )


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for paired 0/1 correctness vectors."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def kendall(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return float("nan")
    return float(stats.kendalltau(a[common], b[common]).statistic)


def reversed_pairs(a: pd.Series, b: pd.Series) -> list[tuple[str, str]]:
    """Pairs (x, y) with a[x] > a[y] but b[x] < b[y] (strict rank reversals)."""
    out = []
    for x, y in combinations(a.index.intersection(b.index), 2):
        if (a[x] - a[y]) * (b[x] - b[y]) < 0:
            hi, lo = (x, y) if a[x] > a[y] else (y, x)
            out.append((hi, lo))
    return out


def rank_stability(
    p: Panel,
    fn: Callable[[Panel, np.ndarray], pd.Series],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, pd.Series | float]:
    """How reproducible is the ranking under resampling of samples?

    Returns the mean Kendall tau of bootstrap rankings vs the full-sample ranking, the
    frequency of the full-sample top-1 staying top-1, and per-pair order consistency.
    """
    rng = np.random.default_rng(seed)
    full = fn(p, np.arange(p.n))
    pairs = list(combinations(full.index, 2))
    keep = {pr: 0 for pr in pairs}
    taus, top1 = [], 0
    for _ in range(n_boot):
        s = fn(p, rng.integers(0, p.n, p.n))
        taus.append(kendall(full, s))
        top1 += int(s.idxmax() == full.idxmax())
        for x, y in pairs:
            keep[(x, y)] += int((full[x] - full[y]) * (s[x] - s[y]) > 0)
    return {
        "mean_tau": float(np.nanmean(taus)),
        "top1_stability": top1 / n_boot,
        "pair_consistency": pd.Series({f"{x} > {y}" if full[x] > full[y] else f"{y} > {x}": v / n_boot
                                       for (x, y), v in keep.items()}),
    }
