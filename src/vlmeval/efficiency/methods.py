"""Method specs for the H1 efficiency study.

Families, which is the whole point of H1:
  * encoder-side (`random`, `avgpool`, `downsample`, `cls_attn`, `divprune`, `visionzip`): tokens are dropped
    before the LLM, no attention map is needed inside the LLM, so FlashAttention/SDPA kernels run unchanged.
  * in-LLM (`fastv`, `pdrop`, `sparsevlm_lite`, `dart`): tokens are dropped at intermediate layers. Attention-
    score based ones need an attention map (or a separate scoring pass); `dart` scores from key similarity.
  * hybrid (`vscan`): encoder-side selection followed by a text-guided in-LLM stage.

All of them are *minimal re-implementations* of the pruning rule (not the official code):
  fastv           one stage after layer 1, score = attention of the last prompt token (FastV, K=2)
  pdrop           three stages (layers 7/15/23), score = last-token attention (PyramidDrop-style)
  sparsevlm_lite  stages at layers 2/6/15, score = mean attention of all instruction tokens
                  (SparseVLM's text-guided rater; no token recycling / merging)
  dart            one stage after layer 1; 8 pivot tokens (largest L1 key norm); keep the tokens least
                  similar (cosine of key states) to any pivot, i.e. drop duplicates. No attention map.
  divprune        max-min diversity (greedy farthest point on cosine distance of projected features)
  visionzip       dominant tokens by CLS attention (84%) + contextual tokens merged from the rest (16%)
  vscan           encoder: 1.5x budget by global (CLS attention) + local (local-contrast) scans, no merging;
                  then one text-guided in-LLM stage after layer 15 down to the budget
  downsample      the image itself is resized before the vision tower (fewer patches => cheaper ViT too)
Other schedules can be defined in the config under `custom_methods`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DART_PIVOTS = 8
CLS_BASED = ("cls_attn", "visionzip", "vscan")      # need a CLS token / CLS attention from the vision tower
PIXEL_SPACE = ("downsample",)                       # act on the image, not on the feature sequence


@dataclass(frozen=True)
class Stage:
    layer: int      # pruning happens after this layer (0-indexed); later layers see fewer tokens
    keep: float     # visual tokens kept = round(keep * budget)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str                                   # none | encoder | inllm (an encoder spec may also have stages)
    encoder: str | None = None                  # key of ENCODERS (or "downsample")
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    scorer: str = "last"                        # last | text | dart (what scores visual tokens inside the LLM)
    enc_keep: float = 1.0                       # encoder output size = round(enc_keep * budget)


BUILTIN: dict[str, MethodSpec] = {
    "none": MethodSpec("none", "none"),
    "random": MethodSpec("random", "encoder", encoder="random"),
    "avgpool": MethodSpec("avgpool", "encoder", encoder="avgpool"),
    "downsample": MethodSpec("downsample", "encoder", encoder="downsample"),
    "cls_attn": MethodSpec("cls_attn", "encoder", encoder="cls_attn"),
    "divprune": MethodSpec("divprune", "encoder", encoder="divprune"),
    "visionzip": MethodSpec("visionzip", "encoder", encoder="visionzip"),
    "vscan": MethodSpec("vscan", "encoder", encoder="vscan", enc_keep=1.5, stages=(Stage(15, 1.0),), scorer="text"),
    "fastv": MethodSpec("fastv", "inllm", stages=(Stage(1, 1.0),)),
    "pdrop": MethodSpec("pdrop", "inllm", stages=(Stage(7, 2.0), Stage(15, 1.0), Stage(23, 0.5))),
    "sparsevlm_lite": MethodSpec("sparsevlm_lite", "inllm", stages=(Stage(2, 1.5), Stage(6, 1.0), Stage(15, 0.5)),
                                 scorer="text"),
    "dart": MethodSpec("dart", "inllm", stages=(Stage(1, 1.0),), scorer="dart"),
}


def spec_from_config(name: str, cfg: dict | None) -> MethodSpec:
    """`cfg` is {kind, encoder?, scorer?, enc_keep?, stages: [{layer, keep}, ...]}; None -> builtin."""
    if cfg is None:
        return BUILTIN[name]
    return MethodSpec(
        name, cfg["kind"], cfg.get("encoder"),
        tuple(Stage(int(s["layer"]), float(s["keep"])) for s in cfg.get("stages", [])),
        cfg.get("scorer", "last"), float(cfg.get("enc_keep", 1.0)),
    )


def pool_shape(n: int, keep: int) -> tuple[int, int]:
    side = math.isqrt(n)
    h = max(1, min(side, math.isqrt(keep)))
    w = max(1, min(side, round(keep / h)))
    return h, w


def resolve(spec: MethodSpec, budget: int, n_vis: int, n_layers: int, tiles: int = 1) -> tuple[int, list[tuple[int, int]]]:
    """-> (visual tokens entering the LLM, [(layer, kept_after_layer), ...]) with counts non-increasing.
    `n_vis` is the total visual token count (over all `tiles`); pooling/downsampling act per tile."""
    if spec.kind == "none":
        return n_vis, []
    entering = n_vis
    if spec.kind == "encoder":
        if spec.encoder in ("avgpool", "downsample"):
            h, w = pool_shape(n_vis // tiles, max(1, budget // tiles))
            entering = tiles * h * w
        else:
            entering = min(round(spec.enc_keep * budget), n_vis)
    n_in, out = entering, []
    for st in sorted(spec.stages, key=lambda s: s.layer):
        if st.layer >= n_layers - 1:
            continue
        keep = max(1, min(entering, round(st.keep * budget)))
        out.append((st.layer, keep))
        entering = keep
    return n_in, out


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


def enc_avgpool(feats, keep, tiles=1, **_):
    b, n, d = feats.shape
    n_t = n // tiles
    side = math.isqrt(n_t)
    h, w = pool_shape(n_t, max(1, keep // tiles))
    grid = feats.reshape(b * tiles, side, side, d).permute(0, 3, 1, 2).float()
    out = F.adaptive_avg_pool2d(grid, (h, w))
    return out.permute(0, 2, 3, 1).reshape(b, tiles * h * w, d).to(feats.dtype)


def enc_divprune(feats, keep, **_):
    """DivPrune-style max-min diversity: greedily add the token farthest (cosine distance) from the chosen set.
    Needs the full N x N distance matrix and `keep` sequential argmax steps -- a real selection overhead."""
    b, n, _d = feats.shape
    x = F.normalize(feats.float(), dim=-1)
    dist = 1 - x @ x.transpose(1, 2)
    rows = torch.arange(b, device=feats.device)
    chosen = [dist.mean(-1).argmax(-1)]                      # start from the most atypical token
    mind = dist[rows, chosen[0]]
    mind[rows, chosen[0]] = -1
    for _ in range(min(keep, n) - 1):
        nxt = mind.argmax(-1)
        chosen.append(nxt)
        mind = torch.minimum(mind, dist[rows, nxt])
        mind[rows, nxt] = -1
    return _gather(feats, torch.stack(chosen, dim=1))


def enc_visionzip(feats, keep, cls_scores=None, **_):
    """VisionZip-style: dominant tokens by CLS attention, remaining tokens merged into evenly spread
    contextual targets (similarity on projected features; the official code merges on key states)."""
    if cls_scores is None:
        raise ValueError("visionzip needs cls_scores from the vision tower")
    b, n, d = feats.shape
    keep = min(keep, n)
    ctx = min(max(1, round(keep * 10 / 64)), keep - 1) if keep > 1 else 0
    dom = keep - ctx
    order = cls_scores.argsort(dim=1, descending=True)
    dom_feats = _gather(feats, order[:, :dom])
    if ctx == 0:
        return dom_feats
    rest = feats.gather(1, order[:, dom:][..., None].expand(-1, -1, d))
    m = rest.shape[1]
    tgt = torch.linspace(0, m - 1, ctx, device=feats.device).round().long().unique()
    is_tgt = torch.zeros(m, dtype=torch.bool, device=feats.device)
    is_tgt[tgt] = True
    targets, others = rest[:, is_tgt], rest[:, ~is_tgt]
    sim = F.normalize(others.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).transpose(1, 2)
    assign = sim.argmax(-1)                                   # [B, m - ctx] -> target index
    acc = targets.float().clone()
    acc.scatter_add_(1, assign[..., None].expand(-1, -1, d), others.float())
    cnt = torch.ones(b, targets.shape[1], device=feats.device)
    cnt.scatter_add_(1, assign, torch.ones_like(assign, dtype=cnt.dtype))
    return torch.cat([dom_feats, (acc / cnt[..., None]).to(feats.dtype)], dim=1)


def enc_vscan(feats, keep, cls_scores=None, tiles=1, **_):
    """VScan-style visual-encoder stage (lite): half global (CLS attention), half local (CLS attention minus
    its 3x3 neighbourhood mean = locally salient patches). No merging; the text-guided in-LLM stage is a Stage."""
    if cls_scores is None:
        raise ValueError("vscan needs cls_scores from the vision tower")
    b, n, _d = feats.shape
    keep = min(keep, n)
    side = math.isqrt(n // tiles)
    c = cls_scores.reshape(b * tiles, 1, side, side)
    local = (c - F.avg_pool2d(c, 3, 1, 1, count_include_pad=False)).reshape(b, n)
    g = max(1, keep // 2)
    gidx = cls_scores.topk(g, dim=1).indices
    lidx = local.scatter(1, gidx, float("-inf")).topk(keep - g, dim=1).indices
    return _gather(feats, torch.cat([gidx, lidx], dim=1))


ENCODERS = {"random": enc_random, "avgpool": enc_avgpool, "cls_attn": enc_cls_attn, "divprune": enc_divprune,
            "visionzip": enc_visionzip, "vscan": enc_vscan}
