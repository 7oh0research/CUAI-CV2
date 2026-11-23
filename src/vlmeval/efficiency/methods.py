"""Method specs for the H1 efficiency study.

Two families, which is the whole point of H1:
  * encoder-side (`random`, `avgpool`, `cls_attn`): tokens are dropped before the LLM, no attention
    map is needed inside the LLM, so FlashAttention/SDPA kernels run unchanged.
  * in-LLM (`fastv`, `pdrop`, `sparsevlm_lite`): tokens are dropped at intermediate layers using
    attention scores, so an attention map (or a separate scoring pass) is needed.

The in-LLM methods are *minimal re-implementations* of the pruning rule (not the official code):
  fastv           one stage after layer 1, score = attention of the last prompt token (FastV, K=2)
  pdrop           three stages (layers 7/15/23), score = last-token attention (PyramidDrop-style)
  sparsevlm_lite  stages at layers 2/6/15, score = mean attention of all instruction tokens
                  (SparseVLM's text-guided rater; no token recycling / merging)
Other schedules can be defined in the config under `custom_methods`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Stage:
    layer: int      # pruning happens after this layer (0-indexed); later layers see fewer tokens
    keep: float     # visual tokens kept = round(keep * budget)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str                                   # none | encoder | inllm
    encoder: str | None = None                  # random | avgpool | cls_attn
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    scorer: str = "last"                        # last | text (which query rows score visual tokens)


BUILTIN: dict[str, MethodSpec] = {
    "none": MethodSpec("none", "none"),
    "random": MethodSpec("random", "encoder", encoder="random"),
    "avgpool": MethodSpec("avgpool", "encoder", encoder="avgpool"),
    "cls_attn": MethodSpec("cls_attn", "encoder", encoder="cls_attn"),
    "fastv": MethodSpec("fastv", "inllm", stages=(Stage(1, 1.0),)),
    "pdrop": MethodSpec("pdrop", "inllm", stages=(Stage(7, 2.0), Stage(15, 1.0), Stage(23, 0.5))),
    "sparsevlm_lite": MethodSpec("sparsevlm_lite", "inllm", stages=(Stage(2, 1.5), Stage(6, 1.0), Stage(15, 0.5)),
                                 scorer="text"),
}


def spec_from_config(name: str, cfg: dict | None) -> MethodSpec:
    """`cfg` is {kind, encoder?, scorer?, stages: [{layer, keep}, ...]}; None -> builtin."""
    if cfg is None:
        return BUILTIN[name]
    return MethodSpec(
        name, cfg["kind"], cfg.get("encoder"),
        tuple(Stage(int(s["layer"]), float(s["keep"])) for s in cfg.get("stages", [])),
        cfg.get("scorer", "last"),
    )


def pool_shape(n: int, keep: int) -> tuple[int, int]:
    side = math.isqrt(n)
    h = max(1, min(side, math.isqrt(keep)))
    w = max(1, min(side, round(keep / h)))
    return h, w


def resolve(spec: MethodSpec, budget: int, n_vis: int, n_layers: int) -> tuple[int, list[tuple[int, int]]]:
    """-> (visual tokens entering the LLM, [(layer, kept_after_layer), ...]) with counts non-increasing."""
    if spec.kind == "none":
        return n_vis, []
    if spec.kind == "encoder":
        if spec.encoder == "avgpool":
            h, w = pool_shape(n_vis, budget)
            return h * w, []
        return min(budget, n_vis), []
    entering, out = n_vis, []
    for st in sorted(spec.stages, key=lambda s: s.layer):
        if st.layer >= n_layers - 1:
            continue
        keep = max(1, min(entering, round(st.keep * budget)))
        out.append((st.layer, keep))
        entering = keep
    return n_vis, out


def layer_tokens(n_text: int, n_vis: int, stages: list[tuple[int, int]], n_layers: int) -> list[int]:
    """Sequence length processed by every layer (a stage after layer L affects layers > L)."""
    out, vis, st = [], n_vis, dict(stages)
    for layer in range(n_layers):
        out.append(n_text + vis)
        vis = st.get(layer, vis)
    return out


# ---- encoder-side pruning, batched: feats [B, N, D] -> [B, keep, D] --------------------------------

def _gather(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return feats.gather(1, idx.sort(dim=1).values[..., None].expand(-1, -1, feats.shape[-1]))


def enc_random(feats, keep, gen=None, **_):
    scores = torch.rand(feats.shape[:2], device=feats.device, generator=gen)
    return _gather(feats, scores.topk(keep, dim=1).indices)


def enc_cls_attn(feats, keep, cls_scores=None, **_):
    if cls_scores is None:
        raise ValueError("cls_attn needs cls_scores from the vision tower")
    return _gather(feats, cls_scores.topk(keep, dim=1).indices)


def enc_avgpool(feats, keep, **_):
    b, n, d = feats.shape
    side = math.isqrt(n)
    h, w = pool_shape(n, keep)
    grid = feats.reshape(b, side, side, d).permute(0, 3, 1, 2).float()
    out = F.adaptive_avg_pool2d(grid, (h, w))
    return out.permute(0, 2, 3, 1).reshape(b, -1, d).to(feats.dtype)


ENCODERS = {"random": enc_random, "avgpool": enc_avgpool, "cls_attn": enc_cls_attn}
