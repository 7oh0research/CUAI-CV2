"""Analysis. Reads results/<mode>/raw, writes results/<mode>/{tables,figures,report.md,blind_ids}.

Definitions (pre-registered, see DESIGN.md):
  correct(x)       = score >= 0.5
  blind item       = vanilla_blank correct AND vanilla_noimage correct
  nonblind         = all items minus blind items
  difficult@b      = pool2d_b wrong  (VTC-Bench: downsampling fails at the same budget)
  retention        = acc(method_b) / acc(vanilla_image) on the same subset
  rank change      = Kendall tau between method orders on `all` and `nonblind`
  reversed pair    = paired bootstrap 95% CI of acc(m1)-acc(m2) excludes 0 on `all`
                     AND excludes 0 with the opposite sign on `nonblind`
"""
import sys, os, json, glob, itertools, yaml
import numpy as np, pandas as pd
from scipy.stats import kendalltau, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
cfg_path = sys.argv[2] if len(sys.argv) > 2 else "configs/default.yaml"
cfg = yaml.safe_load(open(cfg_path))
A = cfg["analysis"]; B = int(A["bootstrap"]); rng = np.random.default_rng(cfg["seed"])
root = f"results/{mode}"
for d in ("tables", "figures", "blind_ids"):
    os.makedirs(f"{root}/{d}", exist_ok=True)

rows = []
for f in glob.glob(f"{root}/raw/*/*.jsonl"):
    rows += [json.loads(l) for l in open(f)]
if not rows:
    sys.exit(f"no results in {root}/raw")
df = pd.DataFrame(rows)
df["cond"] = np.where(df.method == "vanilla", "vanilla_" + df.control, df.method + "_" + df.budget.astype(str))
df["ok"] = (df.score >= 0.5).astype(float)

methods = [m for m in cfg["methods"]]
budgets = cfg["budgets"]
rep = [f"# VTC prune-bench report ({mode})\n",
       f"model: `{cfg['model']}` | methods: {', '.join(methods)} | budgets: {budgets}\n"]

def wide(bench, col="ok"):
    w = df[df.bench == bench].pivot_table(index="id", columns="cond", values=col, aggfunc="first")
    return w.dropna()                     # only items finished in every condition

def boot_ci(x):
    x = np.asarray(x, float); n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    m = x[rng.integers(0, n, (B, n))].mean(1)
    lo = (1 - A["ci"]) / 2
    return tuple(np.quantile(m, [lo, 1 - lo]))

def paired_sig(a, b):
    """sign of mean(a-b) if bootstrap CI excludes 0 else 0"""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    if n < 10:
        return 0
    m = d[rng.integers(0, n, (B, n))].mean(1)
    lo, hi = np.quantile(m, [(1 - A["ci"]) / 2, 1 - (1 - A["ci"]) / 2])
    return 1 if lo > 0 else (-1 if hi < 0 else 0)

# ---------------- 1. controls / blind filter -----------------------------------
ctrl, acc_rows, tau_rows, pair_rows, flip_rows = [], [], [], [], []
subsets_by_bench = {}
for bench in sorted(df.bench.unique()):
    W = wide(bench)
    need = {"vanilla_image", "vanilla_blank", "vanilla_noimage"}
    if not need <= set(W.columns):
        print(f"[skip] {bench}: controls missing"); continue
    sub = df[(df.bench == bench) & (df.cond == "vanilla_image")].set_index("id")
    chance = np.where(sub.kind == "yesno", 0.5,
                      np.where(sub.kind == "mc", 1 / sub.n_choices.fillna(1).clip(lower=1), 0.0))
    ch = float(pd.Series(chance, index=sub.index).reindex(W.index).mean())
    blind = (W.vanilla_blank == 1) & (W.vanilla_noimage == 1)
    pd.Series(W.index[blind]).to_csv(f"{root}/blind_ids/{bench}.txt", index=False, header=False)
    ai, an = W.vanilla_image.mean(), W.vanilla_noimage.mean()
    lpr = (an - ch) / (ai - ch) if ai > ch else np.nan
    k = int(W.vanilla_noimage.sum()); n = len(W)
    p = binomtest(k, n, ch, alternative="greater").pvalue if 0 < ch < 1 else (0.0 if k > 0 else 1.0)
    pok = df[(df.bench == bench) & (df.cond == "vanilla_image")].parse_ok.mean()
    ctrl.append(dict(bench=bench, n=n, chance=round(ch, 3), acc_image=ai, acc_blank=W.vanilla_blank.mean(),
                     acc_noimage=an, LPR=lpr, p=p, blind_frac=blind.mean(), parse_ok=pok,
                     in_rank_test=bool(blind.mean() >= A["min_blind_frac"])))
    subsets = {"all": W.index, "nonblind": W.index[~blind]}
    for b in budgets:
        if f"pool2d_{b}" in W:
            diff = W[f"pool2d_{b}"] == 0
            subsets[f"difficult@{b}"] = W.index[diff]
            subsets[f"difficult_nonblind@{b}"] = W.index[diff & ~blind]
    subsets_by_bench[bench] = subsets

    # ---------------- 2. method accuracy on subsets ----------------------------
    for b in budgets:
        conds = [f"{m}_{b}" for m in methods if f"{m}_{b}" in W]
        for sname in ["all", "nonblind", f"difficult@{b}", f"difficult_nonblind@{b}"]:
            if sname not in subsets:
                continue
            idx = subsets[sname]; base = W.loc[idx, "vanilla_image"].mean()
            for c in conds:
                x = W.loc[idx, c]; lo, hi = boot_ci(x)
                acc_rows.append(dict(bench=bench, budget=b, subset=sname.split("@")[0], method=c.rsplit("_", 1)[0],
                                     n=len(idx), acc=x.mean(), lo=lo, hi=hi, vanilla=base,
                                     retention=x.mean() / base if base > 0 else np.nan))
        # ---------------- 3. rank change all -> nonblind ------------------------
        ms = [c for c in conds]
        if len(ms) >= 3:
            a_all = W.loc[subsets["all"], ms].mean(); a_nb = W.loc[subsets["nonblind"], ms].mean()
            tau = kendalltau(a_all.values, a_nb.values).statistic
            taus = []
            ia, inb = np.array(subsets["all"]), np.array(subsets["nonblind"])
            for _ in range(min(B, 500)):
                sa = W.loc[ia[rng.integers(0, len(ia), len(ia))], ms].mean()
                sn = W.loc[inb[rng.integers(0, len(inb), len(inb))], ms].mean()
                taus.append(kendalltau(sa.values, sn.values).statistic)
            nrev = 0
            for m1, m2 in itertools.combinations(ms, 2):
                s_all = paired_sig(W.loc[subsets["all"], m1], W.loc[subsets["all"], m2])
                s_nb = paired_sig(W.loc[subsets["nonblind"], m1], W.loc[subsets["nonblind"], m2])
                rev = s_all != 0 and s_nb == -s_all
                nrev += rev
                pair_rows.append(dict(bench=bench, budget=b, m1=m1, m2=m2, sig_all=s_all, sig_nonblind=s_nb,
                                      reversed=rev, changed=(s_all != s_nb)))
            spread_all, spread_nb = a_all.max() - a_all.min(), a_nb.max() - a_nb.min()
            tau_rows.append(dict(bench=bench, budget=b, kendall_tau=tau,
                                 tau_lo=np.nanquantile(taus, .025), tau_hi=np.nanquantile(taus, .975),
                                 reversed_pairs=nrev, spread_all=spread_all, spread_nonblind=spread_nb,
                                 best_all=a_all.idxmax().rsplit("_", 1)[0], best_nonblind=a_nb.idxmax().rsplit("_", 1)[0]))
    # ---------------- 4. H4: answer flips vs vanilla --------------------------
    P = df[df.bench == bench].pivot_table(index="id", columns="cond", values="raw", aggfunc="first").reindex(W.index)
    for c in [c for c in W.columns if not c.startswith("vanilla")]:
        flip = (P[c].str.strip().str.lower() != P.vanilla_image.str.strip().str.lower()).mean()
        flip_rows.append(dict(bench=bench, cond=c, acc_delta=W[c].mean() - W.vanilla_image.mean(),
                              flip_rate=flip, gain=((W[c] == 1) & (W.vanilla_image == 0)).mean(),
                              loss=((W[c] == 0) & (W.vanilla_image == 1)).mean()))

C = pd.DataFrame(ctrl); ACC = pd.DataFrame(acc_rows); TAU = pd.DataFrame(tau_rows)
PAIR = pd.DataFrame(pair_rows); FLIP = pd.DataFrame(flip_rows)
for name, t in [("controls", C), ("accuracy", ACC), ("rank_change", TAU), ("pairs", PAIR), ("flips", FLIP)]:
    t.to_csv(f"{root}/tables/{name}.csv", index=False)

def fmt(t, index=False):
    if not len(t):
        return "(empty)\n"
    try:
        return t.to_markdown(index=index, floatfmt=".3f")
    except ImportError:
        return "```\n" + t.round(3).to_string(index=index) + "\n```"
rep += ["\n## 1. Controls and blind filter (H3 premise)\n",
        "blind = solved by vanilla under both `blank` and `noimage`. Benchmarks with "
        f"blind_frac >= {A['min_blind_frac']} enter the ranking test.\n\n", fmt(C)]

if len(ACC):
    piv = ACC[ACC.subset.isin(["all", "nonblind"])].pivot_table(
        index=["bench", "budget", "method"], columns="subset", values="retention").reset_index()
    rep += ["\n\n## 2. Retention (acc / vanilla acc on the same subset)\n\n", fmt(piv)]
    H2 = ACC[ACC.subset == "all"].groupby(["method", "budget"]).retention.mean().unstack()
    rep += ["\n\n### Mean retention over benchmarks (subset=all)\n\n", fmt(H2, index=True)]
if len(TAU):
    T = TAU.merge(C[["bench", "in_rank_test"]], on="bench")
    rep += ["\n\n## 3. Rank change all -> nonblind (H3 main test)\n\n", fmt(T),
            f"\n\nBenchmarks in the test: reversed significant pairs = "
            f"{int(T[T.in_rank_test].reversed_pairs.sum())}; "
            f"median tau = {T[T.in_rank_test].kendall_tau.median():.3f}\n"]
if len(FLIP):
    rep += ["\n\n## 4. Answer flips vs vanilla (H4)\n\n", fmt(FLIP)]

# ---------------- figures -------------------------------------------------------
if len(ACC):
    benches = sorted(ACC.bench.unique()); nb = len(benches)
    cols = min(4, nb); rws = int(np.ceil(nb / cols))
    for sname in ["all", "nonblind", "difficult"]:
        fig, axs = plt.subplots(rws, cols, figsize=(4 * cols, 3.2 * rws), squeeze=False)
        for ax, bench in zip(axs.flat, benches):
            S = ACC[(ACC.bench == bench) & (ACC.subset == sname)]
            for m in methods:
                s = S[S.method == m].sort_values("budget")
                if len(s):
                    ax.errorbar(s.budget, s.acc, yerr=[s.acc - s.lo, s.hi - s.acc], marker="o", ms=3,
                                capsize=2, label=m)
            if len(S):
                ax.axhline(S.vanilla.iloc[0], color="k", ls="--", lw=1)
            ax.set_title(f"{bench} [{sname}]", fontsize=9); ax.set_xscale("log", base=2)
            ax.set_xticks(budgets); ax.set_xticklabels(budgets); ax.set_xlabel("visual tokens kept")
        for ax in list(axs.flat)[nb:]:
            ax.axis("off")
        axs.flat[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{root}/figures/acc_{sname}.png", dpi=130); plt.close(fig)
        rep.append(f"\n\n![acc {sname}](figures/acc_{sname}.png)")
if len(TAU):
    M_ = TAU.pivot_table(index="bench", columns="budget", values="kendall_tau")
    fig, ax = plt.subplots(figsize=(1.2 * len(budgets) + 2, 0.45 * len(M_) + 1.2))
    im = ax.imshow(M_.values, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(M_.columns))); ax.set_xticklabels(M_.columns)
    ax.set_yticks(range(len(M_))); ax.set_yticklabels(M_.index)
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]):
            ax.text(j, i, f"{M_.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Kendall tau: ranking on all vs nonblind"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(f"{root}/figures/rank_tau.png", dpi=130); plt.close(fig)
    rep.append("\n\n![tau](figures/rank_tau.png)")

open(f"{root}/report.md", "w").write("".join(rep))
print(f"wrote {root}/report.md")
