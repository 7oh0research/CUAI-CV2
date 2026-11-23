"""Count H1 configs / runs and give a rough GPU-hour estimate from a config (no GPU, no model needed).

    python scripts/plan_h1.py --config configs/h1.yaml

The time model is deliberately crude (decode ~ weight-read + python-loop bound, prefill ~ FLOPs / effective
throughput); calibrate the constants with `--step-ms` / `--tflops` after the pilot run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from vlmeval.efficiency.families import static_family
from vlmeval.efficiency.methods import BUILTIN, spec_from_config
from vlmeval.efficiency.profiler import _passes, expand_grids, theoretical


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/h1.yaml")
    ap.add_argument("--step-ms", type=float, default=22.0, help="assumed vanilla decode ms/step at batch 1")
    ap.add_argument("--tflops", type=float, default=90.0, help="assumed effective fp16 TFLOPs for prefill")
    ap.add_argument("--overhead-s", type=float, default=8.0, help="per-config diagnostics + setup seconds")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mcfg = cfg.get("model", {})
    fam = static_family(mcfg.get("family", "llava"), mcfg)
    n_vis = fam.n_vis
    combos = expand_grids(cfg, n_vis)
    passes = _passes(cfg)
    custom = cfg.get("custom_methods", {})
    d, v = fam.llm, fam.vdims
    combos = [c for c in combos if c[2] == "none" or fam.supports(spec_from_config(c[2], custom.get(c[2])))]

    by_grid = Counter((impl, bs) for impl, bs, _, _ in combos)
    print(f"configs: {len(combos)}  (methods x budgets x backends x batch sizes after fractional design)")
    print("per (backend, batch):", dict(sorted(by_grid.items())))
    meas = sum(p["repeats"] for p in passes)
    print(f"measured repeats per config: {meas} -> total measured runs: {meas * len(combos)} "
          f"(+ warm-up {sum(p['warmup'] for p in passes) * len(combos)})")

    total_s = 0.0
    for impl, bs, name, budget in combos:
        spec = BUILTIN["none"] if name == "none" else spec_from_config(name, custom.get(name))
        t = theoretical(d, v, spec, budget, n_vis, 40, impl, 1 if spec.scorer == "last" else 12, fam.tiles)
        # batch amortises the weight read; KV traffic grows with batch and kept tokens
        step = args.step_ms * (1 + 0.04 * bs * (t["n_visual_final"] + 40) / 576)
        prefill = bs * (t["prefill_flops"] + t["vision_flops"]) / (args.tflops * 1e12) * 1e3 + 15
        for p in passes:
            n = p["warmup"] + p["repeats"]
            total_s += n * (prefill + (p["max_new"] - 1) * step) / 1e3
        total_s += args.overhead_s + sum(cfg.get("memory_lengths", [])) * step / 1e3
    print(f"estimated GPU time: {total_s / 3600:.1f} h (rough; model load + OOM retries not included). "
          "Recalibrate --step-ms / --tflops from the pilot run.")


if __name__ == "__main__":
    main()
