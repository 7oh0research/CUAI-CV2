"""Analytic FLOPs for a LLaMA-style decoder (+ CLIP vision tower), matmul FLOPs = 2 * MACs.

H1 compares *theoretical* efficiency with *measured* latency, so the theoretical side must be a
transparent function of the per-layer token schedule only. Nothing here depends on kernels.
Conventions (stated so they can be matched to a paper's own counter):
  * attention score/value matmuls are counted dense (no 1/2 causal discount),
  * norms, rotary, softmax, residual adds are ignored,
  * the LM head is applied to the last position only (we only need next-token logits).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LLMDims:
    hidden: int
    inter: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int

    @classmethod
    def from_config(cls, cfg) -> "LLMDims":
        heads = cfg.num_attention_heads
        return cls(
            hidden=cfg.hidden_size,
            inter=cfg.intermediate_size,
            layers=cfg.num_hidden_layers,
            heads=heads,
            kv_heads=getattr(cfg, "num_key_value_heads", None) or heads,
            head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // heads,
            vocab=cfg.vocab_size,
        )


@dataclass(frozen=True)
class VisionDims:
    """ViT + projector. Defaults describe CLIP (CLS token, plain 2-matrix MLP, full attention, 2-layer MLP projector).
    Qwen-VL ViTs: no CLS (`cls=0`), `merge`=4 patches per LLM token, optional gated MLP (`mlp_mats=3`) and
    window attention in all but `full_layers` layers."""
    hidden: int
    inter: int
    layers: int
    tokens: int          # sequence length inside the ViT (patches + CLS)
    proj_out: int        # LLM hidden size (projector output)
    kept_tokens: int = 576
    cls: int = 1
    merge: int = 1
    mlp_mats: int = 2
    window: int = 0      # tokens per attention window (0 = full attention everywhere)
    full_layers: int = 0  # layers that still use full attention when `window` > 0

    @classmethod
    def from_config(cls, vcfg, llm_hidden: int) -> "VisionDims":
        side = vcfg.image_size // vcfg.patch_size
        return cls(vcfg.hidden_size, vcfg.intermediate_size, vcfg.num_hidden_layers,
                   side * side + 1, llm_hidden, side * side)

    @classmethod
    def from_qwen_config(cls, vcfg, llm_hidden: int, out_tokens: int) -> "VisionDims":
        """Qwen2-VL (`embed_dim`, `mlp_ratio`, `depth`) or Qwen2.5-VL (`hidden_size`, `intermediate_size`, `depth`)."""
        merge = vcfg.spatial_merge_size ** 2
        hidden = getattr(vcfg, "hidden_size", None) or vcfg.embed_dim
        inter = getattr(vcfg, "intermediate_size", None) or int(hidden * vcfg.mlp_ratio)
        win = getattr(vcfg, "window_size", 0) or 0
        full = len(getattr(vcfg, "fullatt_block_indexes", None) or [])
        window_tokens = (win // vcfg.patch_size) ** 2 if win else 0
        return cls(hidden, inter, vcfg.depth, out_tokens * merge, llm_hidden, out_tokens, cls=0, merge=merge,
                   mlp_mats=3 if win else 2, window=window_tokens, full_layers=full)

    def resized(self, out_tokens: int) -> "VisionDims":
        """Same tower, but producing `out_tokens` LLM tokens (e.g. after image downsampling)."""
        return replace(self, tokens=out_tokens * self.merge + self.cls, kept_tokens=out_tokens)


LLAVA15_7B = LLMDims(hidden=4096, inter=11008, layers=32, heads=32, kv_heads=32, head_dim=128, vocab=32064)
CLIP_L14_336 = VisionDims(hidden=1024, inter=4096, layers=24, tokens=577, proj_out=4096)
QWEN2_7B = LLMDims(hidden=3584, inter=18944, layers=28, heads=28, kv_heads=4, head_dim=128, vocab=152064)
# 672x672 image -> 48x48 patches -> 576 merged tokens (same visual-token count as LLaVA-1.5)
QWEN2VL_VIT = VisionDims(hidden=1280, inter=5120, layers=32, tokens=2304, proj_out=3584, kept_tokens=576,
                         cls=0, merge=4)
QWEN25VL_VIT = VisionDims(hidden=1280, inter=3420, layers=32, tokens=2304, proj_out=3584, kept_tokens=576,
                          cls=0, merge=4, mlp_mats=3, window=64, full_layers=4)


def layer_flops(d: LLMDims, q_tokens: int, kv_tokens: int) -> int:
    qd, kvd = d.heads * d.head_dim, d.kv_heads * d.head_dim
    proj = 2 * q_tokens * d.hidden * (qd + 2 * kvd + qd)
    mlp = 2 * q_tokens * 3 * d.hidden * d.inter
    attn = 4 * q_tokens * kv_tokens * qd
    return proj + mlp + attn


def prefill_flops(d: LLMDims, layer_tokens: list[int], score_flops: int = 0) -> int:
    """`layer_tokens[l]` = sequence length processed by layer l (queries == keys)."""
    assert len(layer_tokens) == d.layers
    total = sum(layer_flops(d, n, n) for n in layer_tokens)
    return total + 2 * d.hidden * d.vocab + score_flops


def decode_flops(d: LLMDims, kv_lens: list[int]) -> int:
    """One decode step; `kv_lens[l]` = keys attended by layer l (cache + the new token)."""
    assert len(kv_lens) == d.layers
    return sum(layer_flops(d, 1, n) for n in kv_lens) + 2 * d.hidden * d.vocab


def decode_slope(d: LLMDims) -> int:
    """Extra FLOPs per decode step caused by one more cached token (exactly linear)."""
    return d.layers * 4 * d.heads * d.head_dim


def scoring_flops(d: LLMDims, rows: int, keys: int) -> int:
    """Cost of computing attention probabilities for `rows` queries when no full map exists."""
    return 2 * rows * keys * d.heads * d.head_dim


def vision_flops(v: VisionDims, tiles: int = 1) -> int:
    """ViT + projector for `tiles` independent images (LLaVA-NeXT AnyRes runs the tower once per tile)."""
    n, d = v.tokens, v.hidden
    dense = 2 * n * (4 * d * d + v.mlp_mats * d * v.inter)
    full_layers = v.full_layers if v.window else v.layers
    win_layers = v.layers - full_layers
    attn = full_layers * 4 * n * n * d + win_layers * 4 * n * min(n, v.window) * d
    pd = v.merge * d                       # projector input width (merge>1: Qwen patch merger)
    hid = v.proj_out if v.merge == 1 else pd
    proj = 2 * v.kept_tokens * (pd * hid + hid * v.proj_out)
    return tiles * (v.layers * dense + attn + proj)
