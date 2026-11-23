"""Aligned (sample x config) arrays built from per-sample records.

Every metric and bootstrap works on a Panel plus an index array of sample rows, so a
bootstrap replicate is just `rng.integers(0, n, n)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import BASELINE, BLIND

KEY = ["benchmark", "sample_id"]


def label(method: str, ratio: float) -> str:
    return f"{method}@{ratio:g}"


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


@dataclass
class Panel:
    configs: list[tuple[str, float]]
    bench: np.ndarray             # (n,) benchmark name per sample
    correct: np.ndarray           # (n, k) 0/1
    same_pred: np.ndarray         # (n, k) 1 if prediction equals the baseline prediction
    base_correct: np.ndarray      # (n,)
    blind_correct: np.ndarray     # (n,) NaN when no blind run exists
    total_ms: np.ndarray          # (n, k)
    prefill_ms: np.ndarray        # (n, k)
    base_total_ms: np.ndarray     # (n,)
    base_prefill_ms: np.ndarray   # (n,)

    @property
    def n(self) -> int:
        return len(self.bench)

    @property
    def labels(self) -> list[str]:
        return [label(m, r) for m, r in self.configs]

    @property
    def methods(self) -> list[str]:
        return sorted({m for m, _ in self.configs})

    @property
    def ratios(self) -> list[float]:
        return sorted({r for _, r in self.configs})

    def col(self, method: str, ratio: float) -> int:
        return self.configs.index((method, ratio))


def _series(df: pd.DataFrame, method: str, ratio: float, col: str) -> pd.Series:
    sub = df[(df.method == method) & (df.ratio.round(6) == round(ratio, 6))]
    return sub.drop_duplicates(KEY).set_index(KEY)[col]


def build_panel(df: pd.DataFrame, benchmarks: list[str] | None = None) -> Panel:
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    if benchmarks:
        df = df[df.benchmark.isin(benchmarks)]
    df["pred_n"] = _norm(df["pred"])

    configs = sorted(
        {(m, round(float(r), 6)) for m, r in zip(df.method, df.ratio) if m not in (BASELINE, BLIND)}
    )
    if not configs:
        raise ValueError("no compressed configs in records")

    base_c = _series(df, BASELINE, 0.0, "correct")
    base_p = _series(df, BASELINE, 0.0, "pred_n")
    if base_c.empty:
        raise ValueError("baseline ('none') records are required")

    cols = {c: {k: _series(df, c[0], c[1], k) for k in ("correct", "pred_n", "total_ms", "prefill_ms")}
            for c in configs}
    index = base_c.index
    for c in configs:
        index = index.intersection(cols[c]["correct"].index)
    index = index.sort_values()

    def stack(key: str) -> np.ndarray:
        return np.column_stack([cols[c][key].reindex(index).to_numpy(dtype=float if key != "pred_n" else object)
                                for c in configs])

    pred = stack("pred_n")
    base_pred = base_p.reindex(index).to_numpy(dtype=object)
    blind_c = _series(df, BLIND, 0.0, "correct").reindex(index).to_numpy(dtype=float)

    return Panel(
        configs=configs,
        bench=index.get_level_values("benchmark").to_numpy(),
        correct=stack("correct"),
        same_pred=(pred == base_pred[:, None]).astype(float),
        base_correct=base_c.reindex(index).to_numpy(dtype=float),
        blind_correct=blind_c,
        total_ms=stack("total_ms"),
        prefill_ms=stack("prefill_ms"),
        base_total_ms=_series(df, BASELINE, 0.0, "total_ms").reindex(index).to_numpy(dtype=float),
        base_prefill_ms=_series(df, BASELINE, 0.0, "prefill_ms").reindex(index).to_numpy(dtype=float),
    )
