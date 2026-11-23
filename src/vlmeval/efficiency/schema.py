"""Record schema for the H1 efficiency study: one JSONL row per (config, repeat).

`decode_ms` holds the latency of every decode step, so end-to-end latency for *any* output length L
is derived later (vision + prefill + sum(decode_ms[:L-1])). FLOPs are stored analytically and match
the exact token schedule that ran, so FLOPs rank vs latency rank is computed from the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

KEY = ["model", "attn_impl", "method", "budget", "batch_size", "max_new", "repeat"]

COLUMNS = KEY + [
    "status",              # ok | oom | unsupported (forced kernel not available)
    "n_visual_in",         # visual tokens entering the LLM (after encoder-side pruning)
    "n_visual_final",      # visual tokens in the deepest LLM layer (after in-LLM pruning)
    "n_text",
    "vision_ms",           # vision tower + projector + encoder-side pruning
    "prefill_ms",          # LLM prefill incl. in-LLM scoring/pruning, up to the first generated token
    "decode_ms",           # list: latency of each decode step
    "peak_mem_mb",         # torch.cuda.max_memory_allocated (weights included)
    "base_mem_mb",         # allocated memory before the run (weights)
    "kv_mb",
    "vision_flops",
    "prefill_flops",       # LLM prefill (+ scoring pass when no attention map is free)
    "decode_flops_step0",  # first decode step; step j costs step0 + j * decode_flops_slope
    "decode_flops_slope",
    "peak_mem_by_len",     # dict {L: peak MB}: separate runs with max_new=L (first decode-pass repeat only)
    "ovh_score_ms",        # extra attention-scoring pass in prefill (0 with eager: the map is a by-product)
    "ovh_select_ms",       # head-mean + top-k + sort
    "ovh_gather_ms",       # index build + gather of hidden states / positions
    "kernel_family",       # attention kernel actually executed (kernels.py), first repeat only
    "kernel_ok",           # executed family matches the requested attn_impl
    "sm_clock_mhz",        # GPU state sampled right after the config (throttling audit)
    "temp_c",
    "power_w",
    "device",
    "dtype",
    "torch_version",
    "cuda_version",
    "transformers_version",
    "flash_attn_version",
    "git_commit",
]


def append(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in COLUMNS}) + "\n")


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"method": str, "model": str})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")
    return df


def done_keys(path: str | Path) -> set[tuple]:
    path = Path(path)
    if not path.exists():
        return set()
    df = load(path)
    return {tuple(r) for r in df[KEY].itertuples(index=False, name=None)}
