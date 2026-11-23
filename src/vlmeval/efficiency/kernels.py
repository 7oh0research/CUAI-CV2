"""Verify which attention kernel a configuration *actually* ran (does not trust the requested backend).

A short torch.profiler trace is classified by kernel/op names. This is how the "FlashAttention-compatible"
claim of a method is checked against reality: e.g. `sdpa` may silently run the memory-efficient or math kernel.
"""

from __future__ import annotations

from collections import Counter

import torch

FAMILIES = ("fa2", "sdpa_flash", "sdpa_cudnn", "sdpa_efficient", "sdpa_math", "sdpa_unknown", "eager", "unknown")

# (family, substrings); first match wins, specific before generic.
_RULES = [
    ("sdpa_flash", ("pytorch_flash", "_scaled_dot_product_flash_attention")),
    ("fa2", ("flash_fwd", "flash_attn")),
    ("sdpa_cudnn", ("cudnn_attention", "cudnn_sdpa", "sdpa_fprop", "_scaled_dot_product_cudnn")),
    ("sdpa_efficient", ("fmha_cutlass", "efficient_attention", "attention_kernel_forward", "mem_eff")),
    ("sdpa_math", ("_scaled_dot_product_attention_math",)),
    ("sdpa_unknown", ("scaled_dot_product_attention",)),
]
EXPECTED = {"eager": {"eager"}, "sdpa": {"sdpa_flash", "sdpa_cudnn", "sdpa_efficient", "sdpa_math"},
            "flash": {"sdpa_flash"}, "fa2": {"fa2"}}


def classify(names) -> str:
    low = [n.lower() for n in names]
    for fam, subs in _RULES:
        if any(s in n for n in low for s in subs):
            return fam
    return "eager" if any("softmax" in n for n in low) else "unknown"


def kernel_ok(attn_impl: str, family: str) -> bool:
    return family in EXPECTED[attn_impl]


def trace_kernels(fn, attn_impl: str) -> dict:
    """Run `fn()` once under the profiler. -> {family, ok, top} (top = most frequent attention-like names)."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts) as prof:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    fam = classify(names)
    keys = ("attention", "flash", "fmha", "softmax", "sdpa")
    top = Counter(n for n in names if any(k in n.lower() for k in keys)).most_common(5)
    return {"family": fam, "ok": kernel_ok(attn_impl, fam), "top": [n for n, _ in top]}
