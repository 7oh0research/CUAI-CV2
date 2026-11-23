"""Write the H1 report (markdown + CSV + figures) from per-sample efficiency records."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import (H1_REFUTE_TAU, H1_SUPPORT_TAU, expand, frequent_inversions, perm_test_tau,  # noqa: E402
                       settings, summarize)


def _md(df: pd.DataFrame, floatfmt=".3g") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt) if len(df) else "_(none)_"


def _heatmap(st: pd.DataFrame, path: Path) -> None:
    t = st.assign(col=st.phase + st.L.map(lambda L: f"(L={L})" if L else ""),
                  row=st.attn_impl + " bs=" + st.batch_size.astype(str))
    piv = t.pivot_table(index="row", columns="col", values="tau", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 2, 0.5 * len(piv) + 1.5))
    im = ax.imshow(piv.to_numpy(), vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Kendall tau (FLOPs rank vs measured rank)")
    ax.set_title("H1: agreement between FLOPs ranking and measured ranking (mean over budgets)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(ms: pd.DataFrame, path: Path) -> None:
    impls = sorted(ms.attn_impl.unique())
    fig, axes = plt.subplots(1, len(impls), figsize=(5.5 * len(impls), 4.5), squeeze=False)
    for ax, impl in zip(axes[0], impls):
        for name, g in ms[ms.attn_impl == impl].groupby("method"):
            ax.scatter(g.flops / 1e12, g.value, label=name, s=40)
        ax.set_xlabel("analytic FLOPs (T)")
        ax.set_ylabel("measured e2e latency (ms)")
        ax.set_title(f"attn={impl}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _tests(long: pd.DataFrame, ok: pd.DataFrame, lengths, n_perm: int) -> pd.DataFrame:
    """Permutation tests: does the batch size / output length / backend change how well FLOPs predicts the order?"""
    bss, impls = sorted(ok.batch_size.unique()), sorted(ok.attn_impl.unique())
    Ls = [L for L in lengths if L - 1 <= int(ok.max_new.max())]
    ref = dict(phase="e2e", L=max(Ls))
    rows = []
    if len(bss) > 1:
        for impl in impls:
            rows.append(("batch " + f"{bss[0]} vs {bss[-1]}", impl, f"e2e L={max(Ls)}",
                         perm_test_tau(long, {**ref, "attn_impl": impl, "batch_size": bss[0]},
                                       {**ref, "attn_impl": impl, "batch_size": bss[-1]}, n_perm)))
    if len(Ls) > 1:
        for impl in impls:
            rows.append((f"output length {min(Ls)} vs {max(Ls)}", impl, f"e2e bs={bss[0]}",
                         perm_test_tau(long, {"phase": "e2e", "L": min(Ls), "attn_impl": impl, "batch_size": bss[0]},
                                       {"phase": "e2e", "L": max(Ls), "attn_impl": impl, "batch_size": bss[0]}, n_perm)))
    for a, b in zip(impls, impls[1:]):
        rows.append((f"backend {a} vs {b}", "-", f"e2e L={max(Ls)}",
                     perm_test_tau(long, {**ref, "attn_impl": a}, {**ref, "attn_impl": b}, n_perm)))
    return pd.DataFrame([{"factor": f, "attn_impl": i, "phase": p, "tau_a": r["tau_a"], "tau_b": r["tau_b"],
                          "diff": r["diff"], "p_perm": r["p"]} for f, i, p, r in rows])


def write_report(records: pd.DataFrame, out_dir: str | Path, lengths=(1, 32, 128, 512), n_boot: int = 300,
                 title: str = "H1 report", n_perm: int = 2000) -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = records[records.status == "ok"]
    n_oom = int((records.status == "oom").sum())
    n_unsup = int((records.status == "unsupported").sum())
    long = expand(records, lengths)
    st, ms, pairs = settings(long, n_boot=n_boot)
    tests = _tests(long, ok, lengths, n_perm)
    for name, df in {"settings": st, "methods": ms, "inversions": pairs, "permutation_tests": tests}.items():
        df.to_csv(out / f"{name}.csv", index=False)

    head = summarize(st)
    ref_bs = int(ok.batch_size.min())
    ref_budget = int(ok[ok.method != "none"].budget.min())
    ref_L = max(L for L in lengths if L - 1 <= int(ok.max_new.max()))
    envs = ok[["device", "torch_version", "cuda_version", "transformers_version", "flash_attn_version",
               "git_commit"]].drop_duplicates()
    n_set = max(len(st), 1)
    sections = [f"# {title}", "",
                "- environment: " + "; ".join(f"{r.device} | torch {r.torch_version} | cuda {r.cuda_version} | "
                                              f"transformers {r.transformers_version} | flash-attn "
                                              f"{r.flash_attn_version} | commit {r.git_commit}"
                                              for r in envs.itertuples()),
                f"- dtype: {ok.dtype.iloc[0]}; models: {', '.join(sorted(ok.model.unique()))}; "
                f"methods: {', '.join(sorted(ok.method.unique()))}",
                f"- repeats per config (max): {int(ok.repeat.max()) + 1}; OOM configs: {n_oom}; "
                f"unsupported kernel configs: {n_unsup}",
                f"- settings analysed: {len(st)} (model x device x attn_impl x batch x budget x phase x L)",
                f"- **H1 verdict inputs:** {st.h1_support.mean():.0%} of settings have tau CI upper bound < "
                f"{H1_SUPPORT_TAU} (support); {st.h1_refute.mean():.0%} have CI lower bound >= {H1_REFUTE_TAU} "
                f"(refute); {(st.n_inverted > 0).mean():.0%} contain >=1 reliable inversion "
                f"({int(st.n_inverted.sum())} of {int(st.n_pairs.sum())} FLOPs-separated pairs)", "",
                "## 1. FLOPs rank vs measured rank (Kendall tau, higher = FLOPs predicts the order)", "",
                _md(head), "",
                "`share_with_inversion` = fraction of settings with >=1 *reliably* reversed pair "
                "(FLOPs differs >5%, measured order opposite in >=95% of bootstrap resamples, measured gap >2%). "
                "`flops_spread` / `meas_spread` = (max-min)/max across methods: when FLOPs barely differ "
                "(e.g. decode) tau is dominated by near-ties and the inversion count is the safer statistic.", ""]

    for impl in sorted(ok.attn_impl.unique()):
        for phase, L in (("prefill", 0), ("decode", 0), ("e2e", ref_L), ("memory", ref_L)):
            sub = ms[(ms.attn_impl == impl) & (ms.batch_size == ref_bs) & (ms.budget == ref_budget)
                     & (ms.phase == phase) & (ms.L == L)]
            if sub.empty:
                continue
            t = sub[["method", "flops", "flops_speedup", "value", "measured_speedup", "realization"]].copy()
            t["flops"] = t.flops / 1e12
            names = ["method", "FLOPs (T)", "FLOPs speedup", "measured (ms)" if phase != "memory" else "peak mem (MB)",
                     "measured speedup", "realization = measured / FLOPs speedup"]
            if phase == "e2e":
                t["tokens/s"] = ref_bs * L / sub.value * 1e3
                names.append("tokens/s")
            t.columns = names
            sections += [f"## 2. attn={impl}, {phase}{f' (L={L})' if L else ''}, batch={ref_bs}, "
                         f"budget={ref_budget}", "", _md(t.sort_values("FLOPs (T)")), ""]

    sections += ["## 3. Most frequent reliable inversions (method_a has fewer FLOPs but is slower / larger than "
                 "method_b)", "", _md(frequent_inversions(pairs)), ""]
    slow = pairs[pairs.method_b == "none"]
    if len(slow):
        sections += ["## 4. Break-even: methods with fewer FLOPs than the uncompressed baseline that are "
                     "nevertheless slower/larger", "",
                     _md(slow.groupby(["method_a", "attn_impl", "phase"]).size().rename("settings")
                         .reset_index().sort_values("settings", ascending=False).head(15)), ""]
    sections += ["## 5. Permutation tests (does the condition change how well FLOPs predicts the order?)", "",
                 _md(tests), ""]
    k = ok.dropna(subset=["kernel_family"])
    if len(k):
        kt = (k.groupby(["attn_impl", "method"]).agg(kernel_family=("kernel_family", lambda s: ",".join(sorted(set(s)))),
                                                    matches_request=("kernel_ok", "all")).reset_index())
        sections += ["## 6. Kernel verification (what actually ran)", "",
                     _md(kt[~kt.matches_request]) if (~kt.matches_request).any() else "All configs ran the requested "
                     "kernel family.", "", _md(kt.groupby(["attn_impl", "kernel_family"]).size().rename("configs")
                                              .reset_index()), ""]
    o = ok.dropna(subset=["ovh_select_ms"])
    o = o[(o.batch_size == ref_bs) & (o.budget == ref_budget) & (o.method != "none")]
    if len(o):
        sections += [f"## 7. Pruning overhead inside prefill (ms; batch={ref_bs}, budget={ref_budget})", "",
                     _md(o.groupby(["attn_impl", "method"])[["ovh_score_ms", "ovh_select_ms", "ovh_gather_ms"]]
                         .mean().reset_index()), ""]
    clk = ok.dropna(subset=["sm_clock_mhz"])
    if len(clk):
        sections += ["## 8. GPU state after each config (throttling audit)", "",
                     _md(clk.groupby("attn_impl")[["sm_clock_mhz", "temp_c", "power_w"]]
                         .agg(["min", "max"]).pipe(lambda d: d.set_axis(["_".join(c) for c in d.columns], axis=1))
                         .reset_index()), ""]

    _heatmap(st, out / "tau_heatmap.png")
    e2e = ms[(ms.phase == "e2e") & (ms.L == ref_L) & (ms.batch_size == ref_bs) & (ms.budget == ref_budget)]
    if len(e2e):
        _scatter(e2e, out / "flops_vs_latency.png")
        sections += [f"![tau](tau_heatmap.png)", f"![flops vs latency](flops_vs_latency.png)", ""]
    (out / "report.md").write_text("\n".join(sections), encoding="utf-8")
    return {"settings": st, "methods": ms, "inversions": pairs, "summary": head, "tests": tests}
