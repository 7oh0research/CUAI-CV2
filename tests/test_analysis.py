import numpy as np
import pandas as pd
import pytest

from vlmeval.analysis import hypotheses as H
from vlmeval.analysis import metrics as M
from vlmeval.analysis.panel import build_panel
from vlmeval.analysis.stats import bootstrap_metric, kendall, mcnemar_exact, rank_stability, reversed_pairs
from vlmeval.synthetic import make_records


@pytest.fixture(scope="module")
def panel():
    return build_panel(make_records(n=200))


def test_panel_shapes(panel):
    assert panel.correct.shape == (panel.n, len(panel.configs))
    assert set(panel.methods) == {"random", "avgpool", "cls_attn"}


def test_retention_bounds_and_order(panel):
    idx = np.arange(panel.n)
    ret = M.retention_pooled(panel, idx)
    assert (ret <= 1.0 + 1e-9).all()
    assert ret["random@0.9"] < ret["cls_attn@0.9"]  # synthetic: random hurts most


def test_vd_retention_is_stricter_than_pooled(panel):
    idx = np.arange(panel.n)
    assert (M.vd_retention(panel, idx) <= M.retention_pooled(panel, idx) + 1e-9).all()


def test_car_ignores_lucky_gains():
    df = make_records(n=50)
    p = build_panel(df)
    idx = np.arange(p.n)
    assert (M.car(p, idx) <= 1.0).all()


def test_latency_reflects_inefficient_method(panel):
    red = M.latency_reduction(panel, np.arange(panel.n))
    assert red["cls_attn@0.9"] < red["avgpool@0.9"]


def test_pareto_auc_penalises_slow_method(panel):
    auc = M.pareto_auc(panel, np.arange(panel.n))
    assert auc["avgpool"] > auc["cls_attn"] or auc["random"] > auc["cls_attn"]
    assert (auc >= 0).all() and (auc <= 1).all()


def test_mcnemar():
    a = np.array([1, 1, 1, 0, 0, 1] * 10)
    assert mcnemar_exact(a, a) == 1.0
    assert mcnemar_exact(a, 1 - a) < 0.05


def test_kendall_and_reversals():
    a = pd.Series({"x": 3, "y": 2, "z": 1})
    b = pd.Series({"x": 1, "y": 2, "z": 3})
    assert kendall(a, b) == -1.0
    assert ("x", "z") in reversed_pairs(a, b)


def test_bootstrap_ci_contains_point(panel):
    ci = bootstrap_metric(panel, M.retention_pooled, n_boot=50)
    assert ((ci.lo <= ci.value + 1e-9) & (ci.value <= ci.hi + 1e-9)).all()


def test_rank_stability_keys(panel):
    out = rank_stability(panel, M.retention_pooled, n_boot=30)
    assert -1 <= out["mean_tau"] <= 1
    assert 0 <= out["top1_stability"] <= 1


def test_hypotheses_run(panel):
    assert not H.h1_theory_vs_measured(panel).empty
    assert len(H.h1_ranking(panel)) == 1
    per_bench, cmp2 = H.h2_average_hides_collapse(panel)
    assert not per_bench.empty and not cmp2.empty
    share, spread = H.h3_blind_solvable(panel)
    assert share.loc["pope"].iloc[0] > share.loc["textvqa"].iloc[0]
    assert not spread.empty
    assert not H.h4_flips(panel).empty
    assert not H.h5_ratio_dependence(panel).empty
