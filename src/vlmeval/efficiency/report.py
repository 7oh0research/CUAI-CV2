"""Write the H1 report (markdown + CSV + figures) from per-sample efficiency records."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import expand, frequent_inversions, settings, summarize  # noqa: E402


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


def write_report(records: pd.DataFrame, out_dir: str | Path, lengths=(1, 16, 64), n_boot: int = 300,
                 title: str = "H1 report") -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = records[records.status == "ok"]
    n_oom = int((records.status == "oom").sum())
    long = expand(records, lengths)
    st, ms, pairs = settings(long, n_boot=n_boot)
    for name, df in {"settings": st, "methods": ms, "inversions": pairs}.items():
        df.to_csv(out / f"{name}.csv", index=False)

    head = summarize(st)
    ref_bs = int(ok.batch_size.min())
    ref_budget = int(ok[ok.method != "none"].budget.min())
    ref_L = max(L for L in lengths if L - 1 <= int(ok.max_new.max()))
    sections = [f"# {title}", "",
                f"- device: {ok.device.iloc[0]}, dtype: {ok.dtype.iloc[0]}, torch: {ok.torch_version.iloc[0]}",
                f"- models: {', '.join(sorted(ok.model.unique()))}; methods: {', '.join(sorted(ok.method.unique()))}",
                f"- repeats per config: {int(ok.repeat.max()) + 1}; OOM configs skipped: {n_oom}",
                f"- settings analysed: {len(st)} (model x attn_impl x batch x budget x phase x L)", "",
                "## 1. FLOPs rank vs measured rank (Kendall tau, higher = FLOPs predicts the order)", "",
                _md(head), "",
                "`share_with_inversion` = fraction of settings with >=1 *reliably* reversed pair "
                "(FLOPs differs >5%, measured order opposite in >=95% of bootstrap resamples, measured gap >2%). "
                "`flops_spread` / `meas_spread` = (max-min)/max across methods: when FLOPs barely differ "
                "(e.g. decode) tau is dominated by near-ties and the inversion count is the safer statistic.", ""]

    for impl in sorted(ok.attn_impl.unique()):
        for phase, L in (("prefill", 0), ("decode", 0), ("e2e", ref_L)):
            sub = ms[(ms.attn_impl == impl) & (ms.batch_size == ref_bs) & (ms.budget == ref_budget)
                     & (ms.phase == phase) & (ms.L == L)]
            if sub.empty:
                continue
            t = sub[["method", "flops", "flops_speedup", "value", "measured_speedup", "realization"]].copy()
            t["flops"] = t.flops / 1e12
            t.columns = ["method", "FLOPs (T)", "FLOPs speedup", "measured (ms)", "measured speedup",
                         "realization = measured / FLOPs speedup"]
            sections += [f"## 2. attn={impl}, {phase}{f' (L={L})' if L else ''}, batch={ref_bs}, "
                         f"budget={ref_budget}", "", _md(t.sort_values("FLOPs (T)")), ""]

    sections += ["## 3. Most frequent reliable inversions (method_a has fewer FLOPs but is slower / larger than "
                 "method_b)", "", _md(frequent_inversions(pairs)), ""]
    slow = pairs[pairs.method_b == "none"]
    if len(slow):
        sections += ["## 4. Methods with fewer FLOPs than the uncompressed baseline that are nevertheless slower/larger",
                     "", _md(slow.groupby(["method_a", "attn_impl", "phase"]).size().rename("settings")
                             .reset_index().sort_values("settings", ascending=False).head(15)), ""]

    _heatmap(st, out / "tau_heatmap.png")
    e2e = ms[(ms.phase == "e2e") & (ms.L == ref_L) & (ms.batch_size == ref_bs) & (ms.budget == ref_budget)]
    if len(e2e):
        _scatter(e2e, out / "flops_vs_latency.png")
        sections += [f"![tau](tau_heatmap.png)", f"![flops vs latency](flops_vs_latency.png)", ""]
    (out / "report.md").write_text("\n".join(sections), encoding="utf-8")
    return {"settings": st, "methods": ms, "inversions": pairs, "summary": head}
