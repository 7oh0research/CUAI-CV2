"""Latency / memory / FLOPs profiling grid for H1.

Grid = methods x budgets x batch sizes x attention kernels (fractional via `grids`). Output length is *not*
a grid axis: the decode pass decodes `max_new` tokens and logs every step, so any shorter output length L is
derived afterwards (valid because generation is greedy and fixed-length). A cheap prefill pass (max_new=1)
gives many repeats of the prefill-dominated regime. Peak memory depends on L through the KV buffers, so it is
measured by separate runs per L (`memory_lengths`). Timing uses CUDA events (CPU: perf_counter, for pipeline
smoke tests only; CPU numbers mean nothing).
"""

from __future__ import annotations

import itertools
import random
import time

import torch

from . import flops as fl
from .engine import Engine
from .env import collect_env, gpu_state
from .families import family_from_model
from .kernels import trace_kernels
from .methods import BUILTIN, DART_PIVOTS, MethodSpec, layer_tokens, resolve, spec_from_config
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
                attn_impl: str, P_rows: int, tiles: int = 1) -> dict:
    """Analytic FLOPs and token schedule for one config. P_rows = query rows used for scoring."""
    n_in, stages = resolve(spec, budget, n_vis, dims.layers, tiles)
    lt = layer_tokens(n_text, n_in, stages, dims.layers)
    if spec.scorer == "dart":
        # similarity to pivot tokens: independent of the attention kernel
        score = sum(2 * lt[l] * DART_PIVOTS * dims.kv_heads * dims.head_dim for l, _ in stages)
    elif attn_impl == "eager":   # the map is a free by-product; fused kernels need an extra scoring pass
        score = 0
    else:
        score = sum(fl.scoring_flops(dims, P_rows, lt[l]) for l, _ in stages)
    if spec.name == "downsample":   # the ViT itself runs on the smaller image
        vision = fl.vision_flops(vdims.resized(n_in // tiles), tiles)
    else:
        vision = fl.vision_flops(vdims, tiles)
    if spec.kind == "encoder" and spec.encoder == "divprune":   # N x N cosine-distance matrix
        vision += 2 * n_vis * n_vis * vdims.proj_out
    return {
        "n_visual_in": n_in,
        "n_visual_final": stages[-1][1] if stages else n_in,
        "vision_flops": vision,
        "prefill_flops": fl.prefill_flops(dims, lt, score),
        "decode_flops_step0": fl.decode_flops(dims, [n + 1 for n in lt]),
        "decode_flops_slope": fl.decode_slope(dims),
        "stages": stages,
    }


def run_once(engine: Engine, spec: MethodSpec, budget: int, pixel_values, prefix_ids, suffix_ids, max_new: int,
             gen, tiles: int = 1) -> dict:
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
    _, stages = resolve(spec, budget, n_in, engine.dims.layers, tiles)
    _, st = engine.generate(embeds, prefix_ids.shape[1], n_in, suffix_ids.shape[1], stages, spec.scorer, max_new,
                            stamp=clock.stamp)
    d = clock.deltas_ms()
    return {
        "vision_ms": d[0], "prefill_ms": d[1], "decode_ms": d[2:],
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        "base_mem_mb": base / 2**20 if cuda else None,
        "kv_mb": st.kv_bytes() / 2**20,
    }


def _passes(cfg: dict) -> list[dict]:
    if "passes" in cfg:
        return cfg["passes"]
    return [{"max_new": cfg["max_new_tokens"], "warmup": cfg["warmup"], "repeats": cfg["repeats"]}]


def expand_grids(cfg: dict, n_vis: int) -> list[tuple]:
    """Fractional design: top-level axes are defaults, each entry of `grids` may override any axis.
    -> ordered unique [(attn_impl, batch_size, method, budget)]; `none` is added once per (impl, batch)."""
    base = {k: cfg[k] for k in ("methods", "budgets", "batch_sizes", "attn_impls") if k in cfg}
    seen, combos = set(), []
    for grid in cfg.get("grids") or [{}]:
        g = {**base, **{k: v for k, v in grid.items() if k != "name"}}
        for impl, b in itertools.product(g["attn_impls"], g["batch_sizes"]):
            for m, bud in [("none", n_vis)] + [(m, bud) for m in g["methods"] if m != "none" for bud in g["budgets"]]:
                if (impl, b, m, bud) not in seen:
                    seen.add((impl, b, m, bud))
                    combos.append((impl, b, m, bud))
    if cfg.get("shuffle_seed") is not None:   # randomised order decouples config from thermal / clock drift
        random.Random(cfg["shuffle_seed"]).shuffle(combos)
    return combos


def diagnostics(engine: Engine, spec, budget, pix, prefix_ids, suffix_ids, cfg: dict, gen, tiles: int = 1) -> dict:
    """One-off per config: kernel verification, overhead decomposition, memory by output length."""
    out = {}
    # Trace the LLM only: the HF vision tower has its own (always-SDPA) attention and would pollute the family.
    feats = engine.visual(pix, spec, budget, gen)
    embeds = engine.build_embeds(feats, prefix_ids, suffix_ids)
    _, stages = resolve(spec, budget, feats.shape[1], engine.dims.layers, tiles)
    tr = trace_kernels(lambda: engine.generate(embeds, prefix_ids.shape[1], feats.shape[1], suffix_ids.shape[1],
                                               stages, spec.scorer, 2), engine.attn_impl)
    out["kernel_family"], out["kernel_ok"] = tr["family"], tr["ok"]
    run_once(engine, spec, budget, pix, prefix_ids, suffix_ids, 1, gen, tiles)       # warm
    engine.timers = {}
    try:
        run_once(engine, spec, budget, pix, prefix_ids, suffix_ids, 1, gen, tiles)
        out.update({f"ovh_{k}_ms": engine.timers.get(k, 0.0) for k in ("score", "select", "gather")})
    finally:
        engine.timers = None
    mem = {}
    for L in cfg.get("memory_lengths", []):
        try:
            mem[str(L)] = run_once(engine, spec, budget, pix, prefix_ids, suffix_ids, L, gen, tiles)["peak_mem_mb"]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mem[str(L)] = None
    out["peak_mem_by_len"] = mem
    return out


def profile_grid(model, prefix_ids, suffix_ids, cfg: dict, out_path: str, model_name: str,
                 progress=print) -> None:
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mcfg = cfg.get("model", {})
    fam_name = mcfg.get("family", "llava")
    fam = family_from_model(model, fam_name, mcfg)
    n_vis, tiles = fam.n_vis, fam.tiles
    combos = expand_grids(cfg, n_vis)
    custom = cfg.get("custom_methods", {})
    specs = {"none": BUILTIN["none"], **{m: spec_from_config(m, custom.get(m)) for m in
                                         {c[2] for c in combos} - {"none"}}}
    skipped = sorted({c[2] for c in combos if not fam.supports(specs[c[2]])})
    if skipped:
        progress(f"{fam_name}: skipping methods not applicable to this family: {', '.join(skipped)}")
        combos = [c for c in combos if fam.supports(specs[c[2]])]
    impls = sorted({c[0] for c in combos})
    engines = {impl: fam.engine_cls(model, impl) for impl in impls}
    dims = engines[impls[0]].dims
    vdims = fam.vdims
    n_text = prefix_ids.shape[1] + suffix_ids.shape[1]
    passes = _passes(cfg)
    done = done_keys(out_path)
    meta = {**collect_env(dev), "dtype": str(dtype).replace("torch.", ""), "family": fam_name, "tiles": tiles,
            "n_visual_full": n_vis}

    for i, (impl, bs, name, budget) in enumerate(combos):
        spec = specs[name]
        todo = [p for p in passes if (model_name, impl, name, budget, bs, p["max_new"], 0) not in done]
        if not todo:
            continue
        progress(f"[{i + 1}/{len(combos)}] {impl} bs={bs} {name} budget={budget}")
        th = theoretical(dims, vdims, spec, budget, n_vis, n_text, impl,
                         suffix_ids.shape[1] if spec.scorer == "text" else 1, tiles)
        th.pop("stages")
        g = torch.Generator().manual_seed(cfg["seed"])
        pix = fam.pixels(bs, g).to(dev, dtype)
        base = {"model": model_name, "attn_impl": impl, "method": name, "budget": budget, "batch_size": bs,
                "n_text": n_text, **th, **meta}
        for p in todo:
            rows, diag = [], {}
            try:
                for r in range(-p["warmup"], p["repeats"]):
                    gen = torch.Generator(device=dev).manual_seed(cfg["seed"] + max(r, 0))
                    out = run_once(engines[impl], spec, budget, pix, prefix_ids, suffix_ids, p["max_new"], gen,
                                   tiles)
                    if r >= 0:
                        rows.append({**base, "max_new": p["max_new"], "repeat": r, "status": "ok", **out})
                if p is passes[-1] or len(passes) == 1:
                    diag = diagnostics(engines[impl], spec, budget, pix, prefix_ids, suffix_ids, cfg,
                                       torch.Generator(device=dev).manual_seed(cfg["seed"]), tiles)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                progress("  OOM -> recorded, skipping config")
                append(out_path, [{**base, "max_new": p["max_new"], "repeat": 0, "status": "oom"}])
                break
            except (RuntimeError, ImportError) as e:   # forced kernel unavailable (e.g. flash on this GPU/dtype)
                if impl not in ("flash", "fa2"):
                    raise
                progress(f"  {impl} unsupported here -> recorded ({str(e)[:80]})")
                append(out_path, [{**base, "max_new": p["max_new"], "repeat": 0, "status": "unsupported"}])
                break
            rows[0].update(diag)
            state = gpu_state() if dev.type == "cuda" else {}
            for r_ in rows:
                r_.update(state)
            append(out_path, rows)

