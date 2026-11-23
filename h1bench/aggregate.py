import json, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from h1bench import flops as F

def q(x, p): return float(np.percentile(x, p)) if len(x) else None
groups = {}
for p in glob.glob("results/raw/*.jsonl"):
    rows = [json.loads(l) for l in open(p)]
    s = rows[0]; b = [r for r in rows if r["type"] == "batch"]
    groups.setdefault((s["method"], s["R"], s["attn_requested"], s["batch_size"]), []).append((s, b))
out, layer_rows, impl_rows = [], [], []
for (m, R, at, bs), reps in sorted(groups.items()):
    s0 = reps[0][0]; ok = [(s, b) for s, b in reps if s["status"] in ("ok", "fallback")]
    row = dict(method=m, R=R, attn_requested=at, batch_size=bs, n_reps=len(reps))
    if not ok:
        row.update(status=s0["status"], error=s0.get("error", "")[:200]); out.append(row); continue
    B = [x for _, b in ok for x in b]
    pf = [x["prefill_ms"] / bs for x in B]  # 샘플 1개당 (bs>1이면 batch 시간/bs)
    pf_batch = [x["prefill_ms"] for x in B]
    dec = [x["decode_ms_per_tok"] for x in B]
    meds = [np.median([x["prefill_ms"] for x in b]) for _, b in ok]
    vis = B[0]["layer_visual_tokens"]
    row.update(status=ok[0][0]["status"], attn_actual=ok[0][0]["attn_actual"], n_samples=sum(x["n_items"] for x in B),
               flops_analytic_noselect_T=ok[0][0]["flops_analytic_noselect_T"],
               flops_analytic_T=ok[0][0]["flops_analytic_noselect_T"],  # selection FLOPs 는 별도 계산 시 갱신
               flops_profiler_T=next((s["flops_profiler_T"] for s, _ in ok if s.get("flops_profiler_T")), None),
               t_cpu_preproc_ms=np.median([x["t_cpu_preproc_ms"] for x in B]),
               t_vision_ms=np.median([x["t_vision_ms"] for x in B]),
               t_select_ms=(np.median([x["t_select_ms"] for x in B]) if B[0]["t_select_ms"] is not None else None),
               t_llm_prefill_ms=np.median([x["t_llm_prefill_ms"] for x in B]),
               prefill_ms_mean=np.mean(pf_batch), prefill_ms_std=np.std(pf_batch), prefill_ms_median=np.median(pf_batch),
               prefill_ms_p90=q(pf_batch, 90), prefill_ms_p99=q(pf_batch, 99),
               prefill_ms_per_sample_median=np.median(pf),
               decode_ms_per_tok_mean=np.mean(dec), decode_ms_per_tok_std=np.std(dec), decode_ms_per_tok_median=np.median(dec),
               e2e_ms_median=np.median([x["e2e_ms"] for x in B]),
               tokens_per_s=bs * B[0]["n_decode"] / (np.median([x["e2e_ms"] for x in B]) / 1000),
               samples_per_s=bs / (np.median([x["e2e_ms"] for x in B]) / 1000),
               mem_resident_GB=ok[0][0]["mem_resident_GB"],
               mem_peak_prefill_GB=max(x["mem_peak_prefill_GB"] for x in B), mem_peak_total_GB=max(x["mem_peak_total_GB"] for x in B),
               mem_reserved_peak_GB=max(x["mem_reserved_peak_GB"] for x in B), kv_cache_GB=np.median([x["kv_cache_GB"] for x in B]),
               visual_tokens_final=vis[-1], unstable=(max(meds) - min(meds)) / min(meds) > 0.05,
               throttled=any(s.get("throttled") for s, _ in ok))
    out.append(row)
    for i, v in enumerate(vis):
        vals = [x["layer_visual_tokens"][i] for x in B]
        layer_rows.append(dict(method=m, R=R, attn=at, bs=bs, layer=i, mean=np.mean(vals), min=min(vals), max=max(vals)))
    for i, l in enumerate(ok[0][0]["attn_layers"]):
        impl_rows.append(dict(method=m, R=R, attn_requested=at, bs=bs, layer=i, **l))
df = pd.DataFrame(out); df.to_csv("results/summary.csv", index=False)
pd.DataFrame(layer_rows).to_csv("results/layer_tokens.csv", index=False)
pd.DataFrame(impl_rows).to_csv("results/attn_impl_actual.csv", index=False)
# ratios: 같은 attn, 같은 bs 의 baseline 대비
ok = df[df.status.isin(["ok", "fallback"])].copy()
base = ok[ok.method == "baseline"].set_index(["attn_requested", "batch_size"])
rr = []
for _, r in ok[ok.method != "baseline"].iterrows():
    k = (r.attn_requested, r.batch_size)
    if k not in base.index: continue
    b = base.loc[k]
    d = dict(method=r.method, R=r.R, attn=r.attn_requested, batch_size=r.batch_size,
             flops_ratio=r.flops_analytic_T / b.flops_analytic_T if r.flops_analytic_T else None,
             prefill_ratio=r.prefill_ms_median / b.prefill_ms_median, decode_ratio=r.decode_ms_per_tok_median / b.decode_ms_per_tok_median,
             e2e_ratio=r.e2e_ms_median / b.e2e_ms_median, mem_peak_ratio=r.mem_peak_total_GB / b.mem_peak_total_GB)
    d["gap"] = d["prefill_ratio"] - d["flops_ratio"] if d["flops_ratio"] else None
    rr.append(d)
rt = pd.DataFrame(rr); rt.to_csv("results/ratios.csv", index=False)
os.makedirs("results/figs", exist_ok=True)
if len(rt):
    attns = sorted(rt.attn.unique()); fig, ax = plt.subplots(1, len(attns), figsize=(5 * len(attns), 4.5), squeeze=False)
    for a, x in zip(attns, ax[0]):
        sub = rt[rt.attn == a]
        for i, (m, g) in enumerate(sub.groupby("method")):
            x.scatter(g.flops_ratio, g.prefill_ratio, label=m, marker="osD^v<>p"[i % 8])
        x.plot([0, 1], [0, 1], "k--", lw=1); x.set_title(f"attn={a}"); x.set_xlabel("flops_ratio"); x.set_ylabel("prefill_ratio")
    ax[0][-1].legend(fontsize=8); fig.tight_layout(); fig.savefig("results/figs/fig1_flops_vs_prefill.png", dpi=150)
okp = ok[ok.method != "baseline"]
if len(okp):
    A = sorted(okp.attn_requested.unique()); Bs = sorted(okp.batch_size.unique())
    fig, ax = plt.subplots(len(A), len(Bs), figsize=(4 * len(Bs), 3.5 * len(A)), squeeze=False)
    for i, a in enumerate(A):
        for j, b in enumerate(Bs):
            sub = okp[(okp.attn_requested == a) & (okp.batch_size == b)]
            for m, g in sub.groupby("method"):
                g = g.sort_values("R"); ax[i][j].plot(g.R, g.prefill_ms_median, "o-", label=m)
            ax[i][j].set_title(f"attn={a}, bs={b}"); ax[i][j].set_xlabel("R"); ax[i][j].set_ylabel("prefill_ms")
    ax[0][-1].legend(fontsize=7); fig.tight_layout(); fig.savefig("results/figs/fig2_R_vs_prefill.png", dpi=150)
print(df.groupby("status").size().to_string())
