"""Visual-token compression methods. All operate on projected image features [N, D].

Methods are training-free and applied between the projector and the LLM, so every method
shares one code path and differs only in *which* tokens survive. In-LLM methods (e.g. FastV)
are intentionally not included in the pilot; see docs/research_plan.md (extensions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Context:
    keep: int                      # number of visual tokens to keep
    cls_scores: torch.Tensor | None = None  # [N] CLS->patch attention from the vision tower
    generator: torch.Generator | None = None


def none(feats: torch.Tensor, ctx: Context) -> torch.Tensor:
    return feats


def random(feats: torch.Tensor, ctx: Context) -> torch.Tensor:
    idx = torch.randperm(feats.shape[0], generator=ctx.generator)[: ctx.keep]
    return feats[idx.sort().values.to(feats.device)]


def avgpool(feats: torch.Tensor, ctx: Context) -> torch.Tensor:
    """Average-pool the patch grid to an h x w grid with h*w ~= keep (exact when keep factors well)."""
    n, d = feats.shape
    side = int(math.isqrt(n))
    h = max(1, min(side, math.isqrt(ctx.keep)))
    w = max(1, min(side, round(ctx.keep / h)))
    grid = feats.reshape(1, side, side, d).permute(0, 3, 1, 2).float()
    out = F.adaptive_avg_pool2d(grid, (h, w))
    return out.permute(0, 2, 3, 1).reshape(-1, d).to(feats.dtype)


def cls_attn(feats: torch.Tensor, ctx: Context) -> torch.Tensor:
    """Encoder-side top-k by CLS attention (VisionZip-style, without contextual merging)."""
    if ctx.cls_scores is None:
        raise ValueError("cls_attn needs cls_scores from the vision tower")
    idx = ctx.cls_scores.topk(ctx.keep).indices.sort().values
    return feats[idx]


METHODS = {"none": none, "random": random, "avgpool": avgpool, "cls_attn": cls_attn}
