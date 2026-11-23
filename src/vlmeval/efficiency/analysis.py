"""H1 analysis: does the FLOPs ranking of methods agree with the measured latency/memory ranking?

Unit of analysis = a *setting*: (model, attn_impl, batch_size, budget, phase, output length L). Inside a
setting we have the methods (plus the uncompressed baseline) with
  - an analytic FLOPs value (deterministic), and
  - R repeated measurements of latency (or peak memory).
For each setting we report Kendall tau / Spearman between FLOPs rank and measured rank (bootstrap CI over
repeats), and the pairs whose order is *reliably* reversed: FLOPs differs by > `flops_tol` in one direction
while the measurement ordering is the opposite in >= `conf` of bootstrap resamples. Latency noise alone
therefore cannot create a reported inversion.

Phases: prefill (LLM prefill), decode (mean ms/token), e2e (vision + prefill + decode up to L tokens),
memory (peak GPU memory vs e2e FLOPs).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

SETTING = ["model", "device", "attn_impl", "batch_size", "budget", "phase", "L"]
H1_SUPPORT_TAU, H1_REFUTE_TAU = 0.6, 0.9   # pre-registered thresholds (docs/h1_protocol.md)


def expand(df: pd.DataFrame, lengths=(1, 32, 128, 512)) -> pd.DataFrame:
    """Per-repeat (flops, value) rows for every phase. `value` is ms (or MB for the memory phase)."""
    out = []
    ok = df[df.status == "ok"]
    has_by_len = ok.peak_mem_by_len.map(lambda m: isinstance(m, dict) and len(m) > 0).any()
    for r in ok.itertuples(index=False):
        dec = np.asarray(r.decode_ms, dtype=float)
        n = len(dec)
        cs = np.concatenate([[0.0], np.cumsum(dec)])
        fixed = r.vision_ms + r.prefill_ms
        step0, slope = r.decode_flops_step0, r.decode_flops_slope

        def e2e_flops(L):
            k = L - 1
            return r.vision_flops + r.prefill_flops + k * step0 + slope * k * (k - 1) / 2

        base = dict(model=r.model, device=r.device, attn_impl=r.attn_impl, batch_size=r.batch_size,
                    budget=r.budget, method=r.method, repeat=r.repeat, rep=f"{r.max_new}-{r.repeat}")
        out.append({**base, "phase": "prefill", "L": 0, "flops": r.prefill_flops, "value": r.prefill_ms})
        if n:
            out.append({**base, "phase": "decode", "L": 0, "flops": step0 + slope * (n - 1) / 2,
                        "value": dec.mean()})
        for L in lengths:
            if L - 1 <= n:
                out.append({**base, "phase": "e2e", "L": L, "flops": e2e_flops(L), "value": fixed + cs[L - 1]})
        mem = r.peak_mem_by_len
        if isinstance(mem, dict):
            for L in lengths:
                if mem.get(str(L)) is not None:
                    out.append({**base, "phase": "memory", "L": L, "flops": e2e_flops(L), "value": mem[str(L)]})
        elif not has_by_len and r.peak_mem_mb is not None and not pd.isna(r.peak_mem_mb):
            out.append({**base, "phase": "memory", "L": 0, "flops": e2e_flops(n + 1), "value": r.peak_mem_mb})
    return pd.DataFrame(out)


def _tau(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return stats.kendalltau(a, b).statistic


def settings(long: pd.DataFrame, n_boot: int = 300, flops_tol: float = 0.05, meas_tol: float = 0.02,
             conf: float = 0.95,
             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (setting_table, method_table, pair_table)."""
    rng = np.random.default_rng(seed)
    srows, mrows, prows = [], [], []
    none = long[long.method == "none"]
    comp = long[long.method != "none"]
    for key, g in comp.groupby(SETTING):
        key = dict(zip(SETTING, key))
        ref = none[(none.model == key["model"]) & (none.device == key["device"]) & (none.attn_impl == key["attn_impl"])
                   & (none.batch_size == key["batch_size"]) & (none.phase == key["phase"]) & (none.L == key["L"])]
        g = pd.concat([g, ref])
        names = sorted(g.method.unique())
        if len(names) < 3 or "none" not in names:
            continue
        fl = np.array([g[g.method == m].flops.mean() for m in names])
        samples = [g[g.method == m].sort_values("repeat").value.to_numpy() for m in names]
        med = np.array([np.median(s) for s in samples])
        boots = np.stack([np.median(s[rng.integers(0, len(s), (n_boot, len(s)))], axis=1) for s in samples], axis=1)

        taus = np.array([_tau(fl, b) for b in boots])
        sp = stats.spearmanr(fl, med).statistic if np.ptp(fl) and np.ptp(med) else np.nan
        n_pairs = n_inv = 0
        i_none = names.index("none")
        slower_than_none = 0
        for i, j in combinations(range(len(names)), 2):
            if abs(fl[i] - fl[j]) <= flops_tol * max(fl[i], fl[j]):
                continue
            a, b = (i, j) if fl[i] < fl[j] else (j, i)     # FLOPs says a is cheaper
            n_pairs += 1
            frac = float((boots[:, a] > boots[:, b] * (1 + meas_tol)).mean())
            reliable = frac >= conf
            n_inv += reliable
            if reliable:
                prows.append({**key, "method_a": names[a], "method_b": names[b],
                              "flops_ratio": fl[a] / fl[b], "measured_ratio": med[a] / med[b]})
            if b == i_none and reliable:
                slower_than_none += 1
        tau_lo = np.nanpercentile(taus, 2.5) if np.isfinite(taus).any() else np.nan
        tau_hi = np.nanpercentile(taus, 97.5) if np.isfinite(taus).any() else np.nan
        srows.append({**key, "n_methods": len(names), "tau": np.nanmean(taus) if np.isfinite(taus).any() else np.nan,
                      "tau_lo": tau_lo, "tau_hi": tau_hi,
                      "h1_support": bool(tau_hi < H1_SUPPORT_TAU), "h1_refute": bool(tau_lo >= H1_REFUTE_TAU),
                      "spearman": sp, "n_pairs": n_pairs, "n_inverted": int(n_inv),
                      "flops_spread": (fl.max() - fl.min()) / fl.max(), "meas_spread": (med.max() - med.min()) / med.max(),
                      "slower_than_baseline": slower_than_none})
        for k, m in enumerate(names):
            mrows.append({**key, "method": m, "flops": fl[k], "value": med[k],
                          "flops_speedup": fl[i_none] / fl[k], "measured_speedup": med[i_none] / med[k],
                          "realization": (med[i_none] / med[k]) / (fl[i_none] / fl[k])})
    return pd.DataFrame(srows), pd.DataFrame(mrows), pd.DataFrame(prows)


def summarize(st: pd.DataFrame) -> pd.DataFrame:
    """Headline table: per (phase, attn_impl, batch_size) mean tau, share of settings with an inversion and
    the share that supports (tau CI upper < 0.6) / refutes (tau CI lower >= 0.9) H1."""
    g = st.assign(has_inv=st.n_inverted > 0).groupby(["phase", "attn_impl", "batch_size"])
    return g.agg(settings=("tau", "size"), mean_tau=("tau", "mean"), min_tau=("tau", "min"),
                 flops_spread=("flops_spread", "mean"), meas_spread=("meas_spread", "mean"),
                 share_with_inversion=("has_inv", "mean"), share_support=("h1_support", "mean"),
                 share_refute=("h1_refute", "mean"), inverted_pairs=("n_inverted", "sum"),
                 pairs=("n_pairs", "sum")).reset_index()


def per_repeat_tau(long: pd.DataFrame, **filters) -> np.ndarray:
    """Kendall tau (FLOPs vs measured) of every single repeat, over all settings matching `filters`.

    Repeats are paired across methods by `rep` (= max_new-repeat); the uncompressed baseline is included.
    """
    sub = long
    for k, v in filters.items():
        sub = sub[sub[k].isin(v) if isinstance(v, (list, tuple, set)) else sub[k] == v]
    taus = []
    for _, g in sub.groupby(SETTING):
        # only a repeat index measured for every method of this setting is comparable
        piv = g.pivot_table(index="rep", columns="method", values="value", aggfunc="first").dropna()
        fl = g.groupby("method").flops.mean().reindex(piv.columns).to_numpy()
        taus += [t for t in (_tau(fl, row) for row in piv.to_numpy()) if np.isfinite(t)]
    return np.asarray(taus)


def perm_test_tau(long: pd.DataFrame, a: dict, b: dict, n_perm: int = 10000, seed: int = 0) -> dict:
    """Do two execution conditions differ in how well FLOPs predicts the measured order?

    `a`, `b` are filter dicts, e.g. {"batch_size": 1, "phase": "e2e", "L": 128} vs {... "batch_size": 16 ...}.
    Statistic = difference of mean per-repeat tau; null = condition labels exchangeable (two-sided, label
    permutation). Per-repeat taus within a setting share methods, so treat p as a screening value and report it
    next to the bootstrap CIs of the setting table.
    """
    ta, tb = per_repeat_tau(long, **a), per_repeat_tau(long, **b)
    if len(ta) < 2 or len(tb) < 2:
        return {"tau_a": np.nan, "tau_b": np.nan, "diff": np.nan, "p": np.nan, "n_a": len(ta), "n_b": len(tb)}
    obs = ta.mean() - tb.mean()
    pool, rng = np.concatenate([ta, tb]), np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        hits += abs(pool[:len(ta)].mean() - pool[len(ta):].mean()) >= abs(obs) - 1e-12
    return {"tau_a": ta.mean(), "tau_b": tb.mean(), "diff": obs, "p": (hits + 1) / (n_perm + 1),
            "n_a": len(ta), "n_b": len(tb)}


def frequent_inversions(pairs: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    g = pairs.groupby(["method_a", "method_b"])
    return (g.agg(settings=("phase", "size"), median_flops_ratio=("flops_ratio", "median"),
                  median_measured_ratio=("measured_ratio", "median"))
            .sort_values("settings", ascending=False).head(top).reset_index())
