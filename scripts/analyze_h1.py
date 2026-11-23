"""H1: analyse efficiency records (FLOPs rank vs measured latency/memory rank).

    python scripts/analyze_h1.py --records outputs/h1/records.jsonl --out outputs/h1/report
    python scripts/analyze_h1.py --synthetic --out outputs/h1_synthetic   # pipeline check; latencies are simulated
"""

from __future__ import annotations

import argparse

from vlmeval.efficiency.report import write_report
from vlmeval.efficiency.schema import load
from vlmeval.efficiency.synthetic import make_synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/h1/records.jsonl")
    ap.add_argument("--out", default="outputs/h1/report")
    ap.add_argument("--synthetic", action="store_true", help="simulated latencies, pipeline check only")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 16, 64], help="output lengths L for e2e latency")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    if args.synthetic:
        df, title = make_synthetic(), "H1 report (SYNTHETIC latencies: pipeline check only)"
    else:
        df, title = load(args.records), "H1 report"
    res = write_report(df, args.out, args.lengths, args.n_boot, title)
    print(res["summary"].to_string(index=False))
    print(f"\nreport: {args.out}/report.md")


if __name__ == "__main__":
    main()
