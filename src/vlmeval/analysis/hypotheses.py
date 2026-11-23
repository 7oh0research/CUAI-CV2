"""Hypothesis tests H1-H5 (see docs/research_plan.md). Each returns tidy DataFrames.

The core question for every hypothesis: does the conventional metric rank the methods
differently from the more faithful one?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .panel import Panel, label
from .stats import kendall, reversed_pairs


def _at_ratio(s: pd.Series, p: Panel, ratio: float) -> pd.Series:
    """Method-indexed slice of a config-indexed series at one ratio."""
    out = {m: s[label(m, ratio)] for m in p.methods if (m, ratio) in p.configs}
    return pd.Series(out)


def _compare(p: Panel, name_a: str, name_b: str) -> pd.DataFrame:
    idx = np.arange(p.n)
    a, b = M.CONFIG_METRICS[name_a](p, idx), M.CONFIG_METRICS[name_b](p, idx)
    rows = []
    for r in p.ratios:
        sa, sb = _at_ratio(a, p, r), _at_ratio(b, p, r)
        rows.append(
            {
                "ratio": r,
                "metric_a": name_a,
                "metric_b": name_b,
                "kendall_tau": kendall(sa, sb),
                "rank_a": " > ".join(sa.sort_values(ascending=False).index),
                "rank_b": " > ".join(sb.sort_values(ascending=False).index),
                "reversals": "; ".join(f"{x}>{y}" for x, y in reversed_pairs(sa, sb)),
            }
        )
    return pd.DataFrame(rows)


def h1_theory_vs_measured(p: Panel) -> pd.DataFrame:
    """Token reduction (theory) vs measured latency reduction, per config."""
    idx = np.arange(p.n)
    df = pd.DataFrame(
        {
            "theoretical_reduction": [r for _, r in p.configs],
            "measured_total_reduction": M.latency_reduction(p, idx, "total").to_numpy(),
            "measured_prefill_reduction": M.latency_reduction(p, idx, "prefill").to_numpy(),
        },
        index=p.labels,
    )
    df["gap_total"] = df.theoretical_reduction - df.measured_total_reduction
    return df


def h1_ranking(p: Panel) -> pd.DataFrame:
    """Retention-per-theoretical-cost ranking vs retention-per-measured-cost ranking."""
    idx = np.arange(p.n)
    ret = M.retention_pooled(p, idx)
    red = M.latency_reduction(p, idx)
    theo = pd.Series({lab: r for lab, (_, r) in zip(p.labels, p.configs)})
    a, b = ret * theo, ret * red.clip(lower=0)  # speed-weighted retention under each cost model
    sa = pd.Series({m: float(np.mean([a[label(m, r)] for r in p.ratios if (m, r) in p.configs])) for m in p.methods})
    sb = pd.Series({m: float(np.mean([b[label(m, r)] for r in p.ratios if (m, r) in p.configs])) for m in p.methods})
    return pd.DataFrame(
        [{
            "kendall_tau": kendall(sa, sb),
            "rank_theoretical": " > ".join(sa.sort_values(ascending=False).index),
            "rank_measured": " > ".join(sb.sort_values(ascending=False).index),
            "reversals": "; ".join(f"{x}>{y}" for x, y in reversed_pairs(sa, sb)),
        }]
    )


def h2_average_hides_collapse(p: Panel) -> pd.DataFrame:
    """Per-benchmark retention table plus macro-average vs worst-benchmark ranking."""
    rows = []
    for b in np.unique(p.bench):
        idx = np.flatnonzero(p.bench == b)
        r = M.retention_pooled(p, idx)
        rows += [{"benchmark": b, "config": c, "retention": v} for c, v in r.items()]
    per_bench = pd.DataFrame(rows).pivot(index="config", columns="benchmark", values="retention")
    return per_bench, _compare(p, "retention_macro", "worst_retention")


def h3_blind_solvable(p: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of baseline-correct items a blind model also solves; effect on method spread."""
    solved = {}
    for b in np.unique(p.bench):
        sel = (p.bench == b) & (p.base_correct == 1)
        solved[b] = float(np.nanmean(p.blind_correct[sel])) if sel.any() else np.nan
    share = pd.Series(solved, name="blind_solvable_share_of_baseline_correct").to_frame()

    idx = np.arange(p.n)
    all_ret, vd_ret = M.retention_pooled(p, idx), M.vd_retention(p, idx)
    rows = []
    for r in p.ratios:
        sa, sb = _at_ratio(all_ret, p, r), _at_ratio(vd_ret, p, r)
        rows.append(
            {
                "ratio": r,
                "spread_all": float(sa.max() - sa.min()),
                "spread_vd": float(sb.max() - sb.min()),
                "kendall_tau": kendall(sa, sb),
                "reversals": "; ".join(f"{x}>{y}" for x, y in reversed_pairs(sa, sb)),
            }
        )
    return share, pd.DataFrame(rows)


def h4_flips(p: Panel) -> pd.DataFrame:
    idx = np.arange(p.n)
    out = _compare(p, "retention_pooled", "car")
    flips = M.flip_rate(p, idx)
    out["max_flip_rate"] = [float(_at_ratio(flips, p, r).max()) for r in out.ratio]
    return out


def h5_ratio_dependence(p: Panel) -> pd.DataFrame:
    """Ranking by pooled retention at each ratio; reversals across ratios."""
    idx = np.arange(p.n)
    ret = M.retention_pooled(p, idx)
    ranks = {r: _at_ratio(ret, p, r) for r in p.ratios}
    rows = []
    for i, r1 in enumerate(p.ratios):
        for r2 in p.ratios[i + 1:]:
            rows.append(
                {
                    "ratio_a": r1,
                    "ratio_b": r2,
                    "kendall_tau": kendall(ranks[r1], ranks[r2]),
                    "reversals": "; ".join(f"{x}>{y}" for x, y in reversed_pairs(ranks[r1], ranks[r2])),
                }
            )
    return pd.DataFrame(rows)
