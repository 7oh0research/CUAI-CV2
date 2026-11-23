"""H1 profiling: FLOPs (analytic) + latency/memory (measured) for visual-token reduction methods.

    python scripts/profile_efficiency.py --flops-only                       # theoretical table, no GPU/model
    python scripts/profile_efficiency.py --config configs/h1_tiny.yaml --tiny  # CPU smoke test, random tiny model
    python scripts/profile_efficiency.py --config configs/h1.yaml           # real run (GPU, 24GB+ recommended)

Interrupted runs resume: configs already present in records.jsonl are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from vlmeval.efficiency.families import FAMILIES, static_family
from vlmeval.efficiency.methods import BUILTIN, spec_from_config
from vlmeval.efficiency.profiler import profile_grid, theoretical

QUESTION = "Describe the image in detail."


def flops_table(cfg: dict) -> pd.DataFrame:
    mcfg = cfg.get("model", {})
    fam = static_family(mcfg.get("family", "llava"), mcfg)
    d, v, n_vis = fam.llm, fam.vdims, fam.n_vis
    custom = cfg.get("custom_methods", {})
    n_text = 40
    rows = []
    for name in ["none"] + [m for m in cfg["methods"] if m != "none"]:
        spec = spec_from_config(name, custom.get(name))
        if not fam.supports(spec):
            continue
        for b in ([n_vis] if name == "none" else cfg["budgets"]):
            t = theoretical(d, v, spec, b, n_vis, n_text, "eager", 1 if spec.scorer == "last" else 12, fam.tiles)
            rows.append({"method": name, "budget": b, "visual_in": t["n_visual_in"], "visual_final": t["n_visual_final"],
                         "prefill_TFLOPs": t["prefill_flops"] / 1e12,
                         "total_TFLOPs(+vision)": (t["prefill_flops"] + t["vision_flops"]) / 1e12})
    df = pd.DataFrame(rows)
    full = df[df.method == "none"]["prefill_TFLOPs"].iloc[0]
    df["FLOPs_speedup"] = full / df["prefill_TFLOPs"]
    return df


def preflight() -> None:
    import transformers

    print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name}, {p.total_memory / 2**30:.1f} GiB, {arch}")
    if arch not in torch.cuda.get_arch_list():  # e.g. RTX 5090 (sm_120) needs a CUDA 12.8+ build
        raise SystemExit(f"this torch build has no kernels for {arch}: {torch.cuda.get_arch_list()}; "
                         "install a cu128+ build (see docs/remote_gpu.md)")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    print("fp16 matmul ok:", (a @ a).float().abs().mean().item() > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="check torch/CUDA/GPU arch support and exit")
    ap.add_argument("--config", default="configs/h1.yaml")
    ap.add_argument("--tiny", action="store_true", help="random tiny model on CPU (pipeline smoke test)")
    ap.add_argument("--family", default=None, choices=FAMILIES, help="override model.family from the config")
    ap.add_argument("--flops-only", action="store_true", help="print the analytic FLOPs table and exit")
    ap.add_argument("--out", default=None, help="override output_dir")
    args = ap.parse_args()
    if args.preflight:
        return preflight()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.flops_only:
        print(flops_table(cfg).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return

    out = Path(args.out or cfg["output_dir"]) / "records.jsonl"
    fam = args.family or cfg.get("model", {}).get("family", "llava")
    cfg.setdefault("model", {})["family"] = fam
    if args.tiny:
        from vlmeval.efficiency.tiny import build_tiny

        model, tiny_cfg = build_tiny(fam, cfg.get("tiny_layers", 12))
        cfg["model"].update({k: v for k, v in tiny_cfg.items() if k in ("tiles", "image_size")})
        prefix = torch.arange(4, 7)[None]
        suffix = torch.arange(10, 18)[None]
        name = f"tiny-{fam}"
    else:
        from transformers import AutoTokenizer

        from vlmeval.efficiency.families import load_model, prompt_ids

        name = cfg["model"]["name"]
        model = load_model(name, fam, getattr(torch, cfg["model"]["dtype"]))
        prefix, suffix = prompt_ids(AutoTokenizer.from_pretrained(name), fam, cfg.get("question", QUESTION))
    if next(model.parameters()).dtype != torch.float32 and not torch.cuda.is_available():
        raise SystemExit("half precision needs a GPU; use --tiny for a CPU smoke test")

    profile_grid(model, prefix, suffix, cfg, str(out), name)
    print(f"records: {out}\nnext: python scripts/analyze_h1.py --records {out} --out {out.parent / 'report'}")


if __name__ == "__main__":
    main()
