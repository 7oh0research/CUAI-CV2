"""Metrics over a Panel. Each takes (panel, idx) and returns a Series indexed by config label.

Conventional metrics (what the surveyed papers report):
  retention_pooled   accuracy / baseline accuracy, pooled over all samples
  retention_macro    mean over benchmarks of per-benchmark retention
Candidate new metrics (definitions frozen in docs/metric_spec.md):
  worst_retention    min over benchmarks of per-benchmark retention            (H2)
  vd_retention       retention on visually-dependent items only                (H3)
  flip_rate          fraction of items whose answer differs from baseline      (H4)
  car                retained-correct rate: correct in both / baseline correct (H4)
  latency_reduction  1 - median measured latency / baseline median latency     (H1)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .panel import Panel

Metric = Callable[[Panel, np.ndarray], pd.Series]


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) > 0, np.asarray(a) / np.asarray(b), np.nan)


def _series(p: Panel, values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=p.labels)


def retention_pooled(p: Panel, idx: np.ndarray) -> pd.Series:
    return _series(p, _div(p.correct[idx].mean(0), p.base_correct[idx].mean()))


def _per_bench_retention(p: Panel, idx: np.ndarray) -> np.ndarray:
    rows = []
    for b in np.unique(p.bench[idx]):
        sel = idx[p.bench[idx] == b]
        rows.append(_div(p.correct[sel].mean(0), p.base_correct[sel].mean()))
    return np.vstack(rows)


def retention_macro(p: Panel, idx: np.ndarray) -> pd.Series:
    return _series(p, np.nanmean(_per_bench_retention(p, idx), axis=0))


def worst_retention(p: Panel, idx: np.ndarray) -> pd.Series:
    return _series(p, np.nanmin(_per_bench_retention(p, idx), axis=0))


def visual_dependent_mask(p: Panel) -> np.ndarray:
    """Baseline answers correctly with the image but incorrectly without it."""
    return (p.base_correct == 1) & (p.blind_correct == 0)


def vd_retention(p: Panel, idx: np.ndarray) -> pd.Series:
    m = visual_dependent_mask(p)[idx]
    sel = idx[m]
    if len(sel) == 0:
        return _series(p, np.full(len(p.configs), np.nan))
    return _series(p, p.correct[sel].mean(0))  # baseline accuracy is 1 on this subset


def flip_rate(p: Panel, idx: np.ndarray) -> pd.Series:
    return _series(p, 1.0 - p.same_pred[idx].mean(0))


def car(p: Panel, idx: np.ndarray) -> pd.Series:
    both = (p.correct[idx] * p.base_correct[idx][:, None]).sum(0)
    return _series(p, _div(both, p.base_correct[idx].sum()))


def latency_reduction(p: Panel, idx: np.ndarray, kind: str = "total") -> pd.Series:
    lat, base = (p.total_ms, p.base_total_ms) if kind == "total" else (p.prefill_ms, p.base_prefill_ms)
    return _series(p, 1.0 - np.median(lat[idx], axis=0) / np.median(base[idx]))


def pareto_auc(p: Panel, idx: np.ndarray, x_max: float = 1.0, kind: str = "total") -> pd.Series:
    """Normalised area under retention vs measured-latency-reduction, per method.

    The curve starts at (0, 1) and drops to 0 beyond the method's fastest operating point,
    so a method that cannot get faster is penalised instead of ignored.
    """
    ret = retention_pooled(p, idx).to_numpy()
    red = latency_reduction(p, idx, kind).to_numpy()
    out = {}
    grid = np.linspace(0.0, x_max, 1001)
    for m in p.methods:
        cols = [i for i, (mm, _) in enumerate(p.configs) if mm == m and red[i] > 0]
        pts = sorted((red[i], ret[i]) for i in cols)
        xs = np.array([0.0] + [x for x, _ in pts])
        ys = np.array([1.0] + [y for _, y in pts])
        yy = np.interp(grid, xs, ys, right=0.0)
        out[m] = float(((yy[1:] + yy[:-1]) / 2 * np.diff(grid)).sum() / x_max)
    return pd.Series(out)


CONFIG_METRICS: dict[str, Metric] = {
    "retention_pooled": retention_pooled,
    "retention_macro": retention_macro,
    "worst_retention": worst_retention,
    "vd_retention": vd_retention,
    "car": car,
    "neg_flip_rate": lambda p, i: -flip_rate(p, i),  # higher is better, like the others
    "latency_reduction": latency_reduction,
}


def metric_table(p: Panel, names: list[str] | None = None) -> pd.DataFrame:
    idx = np.arange(p.n)
    names = names or list(CONFIG_METRICS)
    return pd.DataFrame({n: CONFIG_METRICS[n](p, idx) for n in names})
