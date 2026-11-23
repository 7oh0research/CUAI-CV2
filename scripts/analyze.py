"""Turn per-sample records into the pilot report (tables, CIs, hypothesis tests, plot).

    python scripts/analyze.py --records outputs/pilot/records.jsonl --out outputs/pilot/report
    python scripts/analyze.py --synthetic --out outputs/synthetic_report   # pipeline check, no GPU
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vlmeval.analysis import hypotheses as H
from vlmeval.analysis import metrics as M
from vlmeval.analysis.panel import build_panel
from vlmeval.analysis.stats import bootstrap_metric, kendall, rank_stability
from vlmeval.io import load_records
from vlmeval.synthetic import make_records


def _md(df) -> str:
    return df.round(4).to_markdown() if hasattr(df, "to_markdown") else df.round(4).to_string()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = make_records() if args.synthetic else load_records(args.records)
    p = build_panel(df)
    idx = np.arange(p.n)
    sections: list[str] = [f"# Pilot report\n\nsamples: {p.n}, configs: {len(p.configs)}\n"]

    def add(title: str, body: str) -> None:
        sections.append(f"## {title}\n\n{body}\n")

    table = M.metric_table(p)
    table.to_csv(out / "metrics.csv")
    add("Metrics per config", _md(table))

    ci = {n: bootstrap_metric(p, M.CONFIG_METRICS[n], args.n_boot) for n in ("retention_pooled", "vd_retention", "car")}
    for n, c in ci.items():
        c.to_csv(out / f"ci_{n}.csv")
        add(f"95% bootstrap CI: {n}", _md(c))

    add("Pareto AUC (retention vs measured latency reduction)", _md(M.pareto_auc(p, idx).to_frame("auc")))

    add("H1 theory vs measured", _md(H.h1_theory_vs_measured(p)) + "\n\n" + _md(H.h1_ranking(p)))
    per_bench, h2 = H.h2_average_hides_collapse(p)
    add("H2 average hides collapse", _md(per_bench) + "\n\n" + _md(h2))
    share, h3 = H.h3_blind_solvable(p)
    add("H3 blind-solvable items", _md(share) + "\n\n" + _md(h3))
    add("H4 answer flips", _md(H.h4_flips(p)))
    add("H5 ratio dependence", _md(H.h5_ratio_dependence(p)))

    # Meta-evaluation: do metrics agree, and is each ranking reproducible under resampling?
    names = ["retention_pooled", "retention_macro", "vd_retention", "car"]
    tau = {a: {b: kendall(M.CONFIG_METRICS[a](p, idx), M.CONFIG_METRICS[b](p, idx)) for b in names} for a in names}
    import pandas as pd

    add("Meta-eval: Kendall tau between metrics (over configs)", _md(pd.DataFrame(tau)))
    stab = {n: rank_stability(p, M.CONFIG_METRICS[n], min(args.n_boot, 500)) for n in names}
    add("Meta-eval: ranking stability under bootstrap",
        _md(pd.DataFrame({n: {"mean_tau": s["mean_tau"], "top1_stability": s["top1_stability"]} for n, s in stab.items()})))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ret, red = M.retention_pooled(p, idx), M.latency_reduction(p, idx)
        fig, ax = plt.subplots(figsize=(5, 4))
        for m in p.methods:
            labs = [lab for lab, (mm, _) in zip(p.labels, p.configs) if mm == m]
            ax.plot([red[l] for l in labs], [ret[l] for l in labs], "o-", label=m)
        ax.set_xlabel("measured latency reduction")
        ax.set_ylabel("retention (pooled)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "pareto.png", dpi=150)
    except ImportError:
        pass

    (out / "report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
