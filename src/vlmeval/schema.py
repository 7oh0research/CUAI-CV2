"""Per-sample record schema shared by the runner and the analysis code.

One JSONL row per (benchmark, method, ratio, sample). Always store per-sample data:
every new metric is derived from it, so averages alone force a re-run.
"""

from __future__ import annotations

BASELINE = "none"   # full 576 visual tokens
BLIND = "blind"     # no image at all (H3: language-prior-solvable items)

COLUMNS = [
    "benchmark",
    "method",
    "ratio",          # fraction of visual tokens removed (0.0 for none/blind)
    "sample_id",
    "gold",           # list of acceptable answers
    "pred",
    "correct",        # 0/1
    "n_visual_tokens",
    "prefill_ms",
    "total_ms",
    "peak_mem_mb",
]
