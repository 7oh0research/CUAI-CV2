"""Qwen2-VL / Qwen2.5-VL front end for the H1 engine (same LLM loop, KV cache and pruning stages as `Engine`).

What differs from LLaVA:
  * vision tower: native-resolution ViT (no CLS token) + 2x2 patch merger. A 672x672 image -> 48x48 patches ->
    576 LLM tokens, the same visual-token count as LLaVA-1.5. We patchify the image ourselves exactly like the HF
    image processor, so `visual()` takes the same [B,3,H,W] tensor as the LLaVA engine.
  * GQA (4 KV heads) -- handled by the shared loop.
  * M-RoPE: 3 position axes. Prompt tokens before the image use t=h=w=i; image tokens use (P, P+row, P+col); the
    suffix continues from P + max(grid side). Generated tokens continue after the suffix, so `next_pos` is
    *not* the sequence length. In-LLM pruning keeps each surviving token's original position.
  * no CLS attention exists, so `cls_attn`, `visionzip`, `vscan` are unavailable (`NOT_APPLICABLE`).
  * Positions of tokens kept by an *encoder-side* method are renumbered on a near-square grid (we do not carry
    indices out of the encoder functions). Positions never change latency/FLOPs; accuracy is not measured here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .engine import Engine
from .methods import CLS_BASED, ENCODERS, PIXEL_SPACE, MethodSpec, pool_shape

NOT_APPLICABLE = CLS_BASED


class QwenEngine(Engine):
    def _bind_vision(self, model, inner, base) -> None:
        self.vis = inner.visual
        vc = model.config.vision_config
        self.patch, self.tpatch, self.merge = vc.patch_size, vc.temporal_patch_size, vc.spatial_merge_size
        rp = getattr(base.config, "rope_parameters", None) or getattr(base.config, "rope_scaling", None) or {}
        self.mrope_section = list(rp.get("mrope_section", [16, 24, 24]))
        self.grid = (1, 1)

    # ---- visual side ---------------------------------------------------------------------------

    def _patchify(self, pix):
        b, c, h, w = pix.shape
        p, m, tp = self.patch, self.merge, self.tpatch
        gh, gw = h // p, w // p
        x = pix[:, None].expand(-1, tp, -1, -1, -1)
        x = x.reshape(b, 1, tp, c, gh // m, m, p, gw // m, m, p).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        patches = x.reshape(b * gh * gw, c * tp * p * p)
        grid = torch.tensor([[1, gh, gw]] * b, device=pix.device)
        return patches, grid

    @torch.no_grad()
    def visual(self, pixel_values, spec: MethodSpec, budget: int, gen=None) -> torch.Tensor:
        if spec.kind == "encoder" and spec.encoder in NOT_APPLICABLE:
            raise NotImplementedError(f"{spec.encoder} needs a CLS token; Qwen-VL's ViT has none")
        b, _, h, w = pixel_values.shape
        unit = self.patch * self.merge
        gh, gw = h // unit, w // unit
        if spec.kind == "encoder" and spec.encoder in PIXEL_SPACE:
            gh, gw = pool_shape(gh * gw, budget)
            pixel_values = F.interpolate(pixel_values.float(), size=(gh * unit, gw * unit), mode="bilinear",
                                         antialias=True, align_corners=False).to(pixel_values.dtype)
        patches, grid = self._patchify(pixel_values)
        out = self.vis(patches.to(self.vis.dtype), grid_thw=grid)
        feats = getattr(out, "pooler_output", out)
        feats = feats.reshape(b, gh * gw, -1)
        self.grid = (gh, gw)
        if spec.kind != "encoder" or spec.encoder in PIXEL_SPACE:
            return feats
        enc = ENCODERS[spec.encoder](feats, round(spec.enc_keep * budget), gen=gen, tiles=1)
        self.grid = (0, 0)                          # selected tokens: positions renumbered, see module doc
        return enc

    # ---- M-RoPE positions ----------------------------------------------------------------------

    def _init_pos(self, b, t, P, n_vis, S, device):
        gh, gw = self.grid
        if gh * gw != n_vis:
            gh = max(1, math.isqrt(n_vis))
            gw = math.ceil(n_vis / gh)
        i = torch.arange(n_vis, device=device)
        pre = torch.arange(P, device=device)
        vis = torch.stack([torch.full_like(i, P), P + i // gw, P + i % gw])          # [3, n_vis]
        start = P + max(gh, gw)
        suf = start + torch.arange(t - P - n_vis, device=device)
        pos = torch.cat([pre.expand(3, -1), vis, suf.expand(3, -1)], dim=1)
        return pos[:, None].expand(3, b, -1), start + (t - P - n_vis)

    def _gather_pos(self, pos, sel):
        return pos.gather(2, sel[None].expand(3, -1, -1))

    def _decode_pos(self, b, next_pos, device):
        return torch.full((3, b, 1), next_pos, device=device, dtype=torch.long)

    def _cos_sin(self, x, pos):
        cos, sin = self.rot(x, pos)
        if cos.ndim == 4:        # older transformers return per-axis tables and select sections in attention
            sec = self.mrope_section * 2
            pick = lambda m: torch.cat([c[i % 3] for i, c in enumerate(m.split(sec, dim=-1))], dim=-1)
            cos, sin = pick(cos), pick(sin)
        return cos, sin
