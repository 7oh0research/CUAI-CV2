"""Analytic FLOPs for a LLaMA-style decoder (+ CLIP vision tower), matmul FLOPs = 2 * MACs.

H1 compares *theoretical* efficiency with *measured* latency, so the theoretical side must be a
transparent function of the per-layer token schedule only. Nothing here depends on kernels.
Conventions (stated so they can be matched to a paper's own counter):
  * attention score/value matmuls are counted dense (no 1/2 causal discount),
  * norms, rotary, softmax, residual adds are ignored,
  * the LM head is applied to the last position only (we only need next-token logits).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    hidden: int
    inter: int
    layers: int
    tokens: int          # patches + CLS
    proj_out: int        # LLM hidden size (projector output)
    kept_tokens: int = 576

    @classmethod
    def from_config(cls, vcfg, llm_hidden: int) -> "VisionDims":
        side = vcfg.image_size // vcfg.patch_size
        return cls(vcfg.hidden_size, vcfg.intermediate_size, vcfg.num_hidden_layers,
                   side * side + 1, llm_hidden, side * side)


LLAVA15_7B = LLMDims(hidden=4096, inter=11008, layers=32, heads=32, kv_heads=32, head_dim=128, vocab=32064)
CLIP_L14_336 = VisionDims(hidden=1024, inter=4096, layers=24, tokens=577, proj_out=4096)


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


def vision_flops(v: VisionDims) -> int:
    n, d = v.tokens, v.hidden
    per_layer = 2 * n * (4 * d * d + 2 * d * v.inter) + 4 * n * n * d
    proj = 2 * v.kept_tokens * (d * v.proj_out + v.proj_out * v.proj_out)
    return v.layers * per_layer + proj
