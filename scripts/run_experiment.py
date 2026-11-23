"""Run the full (benchmark x method x ratio) grid and log one JSONL row per sample.

    python scripts/run_experiment.py --config configs/pilot.yaml
    python scripts/run_experiment.py --config configs/pilot.yaml --dry-run   # data only, no GPU

Interrupted runs resume: rows already in records.jsonl are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from vlmeval.data.loaders import load_benchmark
from vlmeval.data.scoring import is_correct
from vlmeval.io import append_records, done_keys
from vlmeval.schema import BASELINE, BLIND


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--dry-run", action="store_true", help="load data and print a sample; no model")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out = Path(cfg["output_dir"]) / "records.jsonl"
    seed = cfg["seed"]

    if args.dry_run:
        for name, b in cfg["benchmarks"].items():
            s = next(iter(load_benchmark(name, b["split"], 2, seed)))
            print(name, s.sample_id, s.kind, repr(s.question[:80]), s.gold[:3])
        return

    from vlmeval.models.llava import LlavaRunner

    m = cfg["model"]
    runner = LlavaRunner(m["name"], m["dtype"], m["attn_implementation"])
    grid = [(BASELINE, 0.0), (BLIND, 0.0)] + [(me, r) for me in cfg["methods"] for r in cfg["ratios"]]
    done = done_keys(out)

    for name, b in cfg["benchmarks"].items():
        for i, s in enumerate(tqdm(load_benchmark(name, b["split"], b["n_samples"], seed), desc=name)):
            if i < cfg["warmup_samples"]:  # discard cold-start effects (CUDA init, allocator)
                for method, ratio in grid:
                    runner.answer(None if method == BLIND else s.image, s.question,
                                  None if method == BLIND else method, ratio, cfg["max_new_tokens"])
            rows = []
            for method, ratio in grid:
                if (name, method, float(ratio), s.sample_id) in done:
                    continue
                r = runner.answer(None if method == BLIND else s.image, s.question,
                                  None if method == BLIND else method, ratio,
                                  cfg["max_new_tokens"], seed=seed + i)
                rows.append({"benchmark": name, "method": method, "ratio": float(ratio),
                             "sample_id": s.sample_id, "gold": s.gold, "correct": is_correct(r["pred"], s.gold, s.kind),
                             **r})
            append_records(out, rows)


if __name__ == "__main__":
    main()
