"""Latency / memory / FLOPs profiling grid for H1.

Grid = methods x budgets x batch sizes x attention kernels. Output length is *not* a grid axis: every
run decodes `max_new` tokens and logs each step, so any shorter output length is derived afterwards.
Timing uses CUDA events (CPU: perf_counter, for pipeline smoke tests only; CPU numbers mean nothing).
"""

from __future__ import annotations

import itertools
import time

import torch

from . import flops as fl
from .engine import Engine
from .methods import BUILTIN, MethodSpec, layer_tokens, resolve, spec_from_config
from .schema import append, done_keys


class Clock:
    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.marks: list = []

    def stamp(self) -> None:
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append(e)
        else:
            self.marks.append(time.perf_counter())

    def deltas_ms(self) -> list[float]:
        if self.cuda:
            torch.cuda.synchronize()
            return [a.elapsed_time(b) for a, b in zip(self.marks, self.marks[1:])]
        return [(b - a) * 1e3 for a, b in zip(self.marks, self.marks[1:])]


def theoretical(dims: fl.LLMDims, vdims: fl.VisionDims, spec: MethodSpec, budget: int, n_vis: int, n_text: int,
                attn_impl: str, P_rows: int) -> dict:
    """Analytic FLOPs and token schedule for one config. P_rows = query rows used for scoring."""
    n_in, stages = resolve(spec, budget, n_vis, dims.layers)
    lt = layer_tokens(n_text, n_in, stages, dims.layers)
    # With eager attention the map is a free by-product; with fused kernels scoring is an extra pass.
    score = 0 if attn_impl == "eager" else sum(fl.scoring_flops(dims, P_rows, lt[l]) for l, _ in stages)
    return {
        "n_visual_in": n_in,
        "n_visual_final": stages[-1][1] if stages else n_in,
        "vision_flops": fl.vision_flops(vdims),
        "prefill_flops": fl.prefill_flops(dims, lt, score),
        "decode_flops_step0": fl.decode_flops(dims, [n + 1 for n in lt]),
        "decode_flops_slope": fl.decode_slope(dims),
        "stages": stages,
    }


def run_once(engine: Engine, spec: MethodSpec, budget: int, pixel_values, prefix_ids, suffix_ids, max_new: int,
             gen) -> dict:
    dev = pixel_values.device
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    clock = Clock(dev)
    clock.stamp()
    feats = engine.visual(pixel_values, spec, budget, gen)
    clock.stamp()
    embeds = engine.build_embeds(feats, prefix_ids, suffix_ids)
    n_in = feats.shape[1]
    _, stages = resolve(spec, budget, n_in, engine.dims.layers)
    _, st = engine.generate(embeds, prefix_ids.shape[1], n_in, suffix_ids.shape[1], stages, spec.scorer, max_new,
                            stamp=clock.stamp)
    d = clock.deltas_ms()
    return {
        "vision_ms": d[0], "prefill_ms": d[1], "decode_ms": d[2:],
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        "base_mem_mb": base / 2**20 if cuda else None,
        "kv_mb": st.kv_bytes() / 2**20,
    }


def profile_grid(model, prefix_ids, suffix_ids, cfg: dict, out_path: str, model_name: str,
                 progress=print) -> None:
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vcfg = model.config.vision_config
    n_vis = (vcfg.image_size // vcfg.patch_size) ** 2
    engines = {impl: Engine(model, impl) for impl in cfg["attn_impls"]}
    dims = next(iter(engines.values())).dims
    vdims = fl.VisionDims.from_config(vcfg, dims.hidden)
    n_text = prefix_ids.shape[1] + suffix_ids.shape[1]
    custom = cfg.get("custom_methods", {})
    specs = {m: spec_from_config(m, custom.get(m)) for m in cfg["methods"]}

    done = done_keys(out_path)
    combos = []
    for impl, b in itertools.product(cfg["attn_impls"], cfg["batch_sizes"]):
        combos.append((impl, b, "none", n_vis, BUILTIN["none"]))
        for m, bud in itertools.product(cfg["methods"], cfg["budgets"]):
            if m != "none":
                combos.append((impl, b, m, bud, specs[m]))

    meta = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "dtype": str(dtype).replace("torch.", ""), "torch_version": torch.__version__}
    for i, (impl, bs, name, budget, spec) in enumerate(combos):
        if (model_name, impl, name, budget, bs, 0) in done:
            continue
        progress(f"[{i + 1}/{len(combos)}] {impl} bs={bs} {name} budget={budget}")
        th = theoretical(dims, vdims, spec, budget, n_vis, n_text, impl,
                         suffix_ids.shape[1] if spec.scorer == "text" else 1)
        th.pop("stages")
        base = {"model": model_name, "attn_impl": impl, "method": name, "budget": budget, "batch_size": bs,
                "n_text": n_text, "max_new": cfg["max_new_tokens"], **th, **meta}
        g = torch.Generator().manual_seed(cfg["seed"])
        pix = torch.randn(bs, 3, vcfg.image_size, vcfg.image_size, generator=g).to(dev, dtype)
        rows = []
        try:
            for r in range(-cfg["warmup"], cfg["repeats"]):
                gen = torch.Generator(device=dev).manual_seed(cfg["seed"] + max(r, 0))
                out = run_once(engines[impl], spec, budget, pix, prefix_ids, suffix_ids, cfg["max_new_tokens"], gen)
                if r >= 0:
                    rows.append({**base, "repeat": r, "status": "ok", **out})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            progress("  OOM -> recorded, skipping config")
            rows = [{**base, "repeat": 0, "status": "oom"}]
        append(out_path, rows)
