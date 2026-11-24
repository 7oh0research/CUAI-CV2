"""문항별 결과로 H3 지표를 계산하고 보고서를 만든다. results/<run>/raw만 읽는다."""
import argparse, glob, json, os
import numpy as np, pandas as pd, yaml
from scipy.stats import norm as N

def load(raw_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        raise SystemExit(f"{raw_dir}에 결과가 없음")
    return pd.DataFrame(rows)

def boot_ci(x, B, rng):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return tuple(np.percentile(m, [2.5, 97.5]))

def metrics(w, btype, B, rng):
    r = {"n": len(w)}
    for c in ("image", "blank", "noimage"):
        if f"score_{c}" in w: r[f"acc_{c}"] = w[f"score_{c}"].mean()
    chance = w["chance"].mean(); r["chance"] = chance
    if btype in ("yesno", "mc"):
        r["majority"] = w["gold"].astype(str).value_counts(normalize=True).iloc[0]
    else:
        r["majority"] = np.nan
    if "score_noimage" in w:
        lo, hi = boot_ci(w["score_noimage"], B, rng); r["acc_noimage_ci_lo"], r["acc_noimage_ci_hi"] = lo, hi
        denom = r.get("acc_image", np.nan) - chance
        r["LPR"] = (r["acc_noimage"] - chance) / denom if denom and denom > 0 else np.nan
        if btype in ("yesno", "mc"):      # Poisson-binomial 정규 근사, 단측 검정
            p = w["chance"].values; k = w["score_noimage"].sum()
            mu, var = p.sum(), (p * (1 - p)).sum()
            r["p_value"] = float(1 - N.cdf((k - 0.5 - mu) / np.sqrt(var))) if var > 0 else np.nan
        else:                              # 개방형: chance=0, 정답이 하나라도 있으면 사전지식 존재
            r["p_value"] = 0.0 if r["acc_noimage"] > 0 else 1.0
    if {"score_noimage", "score_blank"} <= set(w):
        solv = (w["score_noimage"] >= 1.0) & (w["score_blank"] >= 1.0)
        r["blind_solvable"] = solv.mean()
        if "score_image" in w:
            corr = w["score_image"] >= 1.0
            r["blind_share_of_correct"] = (solv & corr).sum() / max(corr.sum(), 1)
            r["flip_rate_image_vs_noimage"] = (w["pred_image"].astype(str) != w["pred_noimage"].astype(str)).mean()
    if btype == "yesno" and "pred_noimage" in w:
        r["yes_rate_noimage"] = (w["pred_noimage"] == "yes").mean()
        r["yes_rate_image"] = (w["pred_image"] == "yes").mean() if "pred_image" in w else np.nan
        r["gold_yes_rate"] = (w["gold"].astype(str) == "yes").mean()
    if "pred_noimage" in w:
        r["unparsed_noimage"] = (w["pred_noimage"] == "other").mean()
    return r

def wide(df):
    base = df[df.cond == df.cond.iloc[0]][["bench", "id", "type", "category", "chance", "gold"]].drop_duplicates(["bench", "id"])
    for c in df.cond.unique():
        s = df[df.cond == c][["bench", "id", "score", "pred"]].rename(columns={"score": f"score_{c}", "pred": f"pred_{c}"})
        base = base.merge(s, on=["bench", "id"], how="inner")
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run-name", default="full")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); dec = cfg["decision"]
    out = os.path.join("results", a.run_name)
    df = load(os.path.join(out, "raw"))
    df["gold"] = df["gold"].apply(lambda g: json.dumps(g) if isinstance(g, list) else g)
    W = wide(df); rng = np.random.default_rng(cfg["seed"])
    summ, cats = [], []
    os.makedirs(os.path.join(out, "blind_solvable_ids"), exist_ok=True)
    for bench, w in W.groupby("bench"):
        bt = w["type"].iloc[0]
        r = {"bench": bench, "type": bt, **metrics(w, bt, dec["bootstrap"], rng)}
        r["H3_meaningful"] = bool(r.get("p_value", 1) < dec["alpha"] and (r.get("LPR") or 0) >= dec["lpr_threshold"])
        summ.append(r)
        for cat, wc in w.groupby("category"):
            if len(wc) >= 20:
                cats.append({"bench": bench, "category": cat, **metrics(wc, bt, 200, rng)})
        if {"score_noimage", "score_blank"} <= set(w):
            ids = w[(w.score_noimage >= 1) & (w.score_blank >= 1)]["id"]
            open(os.path.join(out, "blind_solvable_ids", f"{bench}.txt"), "w").write("\n".join(map(str, ids)))
    S = pd.DataFrame(summ); C = pd.DataFrame(cats)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    C.to_csv(os.path.join(out, "per_category.csv"), index=False)
    W.to_parquet(os.path.join(out, "items_wide.parquet"), index=False)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cols = [c for c in ("acc_image", "acc_blank", "acc_noimage", "chance") if c in S]
        ax = S.set_index("bench")[cols].plot.bar(figsize=(9, 4)); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
        ax.set_title(f"Vanilla {cfg['model_id']} — image vs blind"); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_accuracy.png"), dpi=150); plt.close()
    except Exception as e:
        print("그림 생성 실패:", e)

    meta = json.load(open(os.path.join(out, "run_meta.json"))) if os.path.exists(os.path.join(out, "run_meta.json")) else {}
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v)
    show = [c for c in ["bench", "n", "acc_image", "acc_blank", "acc_noimage", "chance", "majority", "LPR",
                        "blind_solvable", "blind_share_of_correct", "p_value", "yes_rate_noimage", "unparsed_noimage", "H3_meaningful"] if c in S]
    lines = ["# H3 Blind Test Report", "",
             f"- model: {cfg['model_id']}  |  GPU: {meta.get('gpu')}  |  torch {meta.get('torch')} (CUDA {meta.get('cuda')})  |  attn: {meta.get('attn_used')}",
             f"- run: {a.run_name}  |  max_samples: {meta.get('max_samples')}  |  started: {meta.get('started')}",
             f"- 판정 규칙(사전 고정): p_value < {dec['alpha']} AND LPR >= {dec['lpr_threshold']}", "",
             "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    for _, r in S.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in show) + " |")
    k = int(S["H3_meaningful"].sum())
    lines += ["", f"**결론: {len(S)}개 벤치마크 중 {k}개에서 blind 문항이 유의미함.** "
              + ("H3 본 실험(압축 방법 비교)으로 진행할 근거가 있음." if k else "H3의 실험적 근거가 약함."), "",
              "## 해석 시 주의", "- yes/no 과제는 한쪽으로만 답해도 약 50%가 나온다. yes_rate_noimage와 gold_yes_rate를 함께 볼 것.",
              "- unparsed_noimage가 높으면 이미지가 없을 때 형식을 지키지 않은 것이므로 정확도가 과소평가될 수 있다.",
              "- blind_solvable은 noimage와 blank 두 조건 모두에서 맞힌 문항 비율이다.",
              "- 하위 범주 결과: per_category.csv / blind 문항 목록: blind_solvable_ids/", "", "![](fig_accuracy.png)"]
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines))
    print("\n".join(lines[:len(S) + 8]))

if __name__ == "__main__":
    main()
