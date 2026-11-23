"""Manual LLaVA-1.5 inference loop used for the H1 efficiency measurements.

Why not `model.generate`? H1 needs (a) in-LLM token pruning at arbitrary layers, (b) control over which
attention kernel runs (`sdpa` = fused/FlashAttention-capable vs `eager` = materialised attention map,
the only mode in which attention probabilities are available for free), and (c) a per-layer KV cache whose
length differs between layers after pruning. The loop below reuses only the *weights* of the HF model
(projections, MLP, norms, rotary table), so it does not depend on HF's cache/mask internals, which
changed across transformers releases.

Sequence layout is always [prefix | visual | suffix]; pruning only ever removes visual tokens, so the
suffix (the instruction) and the last prompt token are never dropped, and token order is preserved.
No padding is used (profiling batches replicate one prompt), so SDPA runs with `is_causal=True`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .flops import LLMDims
from .methods import ENCODERS, MethodSpec


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _rope(q, k, cos, sin):
    cos, sin = cos[:, None].to(q.dtype), sin[:, None].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@dataclass
class State:
    k: list[torch.Tensor] = field(default_factory=list)   # per layer [B, Hkv, len_l + max_new, D]
    v: list[torch.Tensor] = field(default_factory=list)
    lens: list[int] = field(default_factory=list)
    next_pos: int = 0

    def kv_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


ATTN_IMPLS = ("eager", "sdpa", "flash", "fa2")
# eager: materialised attention map (probabilities free). sdpa: torch picks the fused kernel. flash: torch SDPA
# *forced* to its FlashAttention kernel (raises if ineligible, so a silent fallback is impossible). fa2: the
# flash-attn package. Fused backends need a separate scoring pass for attention-based pruning.


class Engine:
    def __init__(self, model, attn_impl: str = "sdpa"):
        assert attn_impl in ATTN_IMPLS, attn_impl
        self.model, self.attn_impl = model, attn_impl
        self.timers: dict[str, float] | None = None   # set to {} for an overhead-decomposition pass
        core = getattr(model, "model", None)
        inner = core if core is not None and hasattr(core, "language_model") else model
        lm = inner.language_model
        if hasattr(lm, "layers"):
            base, self.head = lm, model.lm_head
        else:  # older transformers: language_model is a LlamaForCausalLM
            base, self.head = lm.model, lm.lm_head
        self.layers, self.norm, self.rot, self.embed = base.layers, base.norm, base.rotary_emb, base.embed_tokens
        self.vt, self.proj = inner.vision_tower, inner.multi_modal_projector
        self.clip = getattr(self.vt, "vision_model", self.vt)
        self.feature_layer = model.config.vision_feature_layer
        self.dims = LLMDims.from_config(base.config)
        self._masks: dict = {}

    # ---- visual side ---------------------------------------------------------------------------

    @torch.no_grad()
    def visual(self, pixel_values, spec: MethodSpec, budget: int, gen=None) -> torch.Tensor:
        """Vision tower + projector (+ encoder-side pruning). -> [B, n_visual, hidden]."""
        hs = self.vt(pixel_values, output_hidden_states=True).hidden_states
        feats = self.proj(hs[self.feature_layer][:, 1:])
        if spec.kind != "encoder":
            return feats
        cls = self._cls_scores(hs) if spec.encoder == "cls_attn" else None
        return ENCODERS[spec.encoder](feats, budget, cls_scores=cls, gen=gen)

    def _cls_scores(self, hs) -> torch.Tensor:
        """Head-averaged CLS->patch attention of the layer that produced the consumed features."""
        layer = self.clip.encoder.layers[self.feature_layer]
        a = layer.self_attn
        h = layer.layer_norm1(hs[self.feature_layer - 1])
        b, n, _ = h.shape
        q = a.q_proj(h[:, :1]).view(b, 1, a.num_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(h).view(b, n, a.num_heads, a.head_dim).transpose(1, 2)
        att = ((q @ k.transpose(-1, -2)) * a.scale).float().softmax(-1).mean(1)
        return att[:, 0, 1:]

    def build_embeds(self, feats, prefix_ids, suffix_ids):
        b = feats.shape[0]
        pre = self.embed(prefix_ids).expand(b, -1, -1)
        suf = self.embed(suffix_ids).expand(b, -1, -1)
        return torch.cat([pre, feats.to(pre.dtype), suf], dim=1)

    # ---- attention kernels ---------------------------------------------------------------------

    def _expand(self, t):
        g = self.dims.heads // self.dims.kv_heads
        return t if g == 1 else t.repeat_interleave(g, dim=1)

    def _mask(self, tq, tk, device):
        key = (tq, tk, str(device))
        if key not in self._masks:
            self._masks[key] = torch.ones(tq, tk, dtype=torch.bool, device=device).triu(1 + tk - tq)
        return self._masks[key]

    @contextmanager
    def _t(self, name: str):
        """Accumulate synchronised wall time of a region (only in decomposition passes: it serialises the GPU)."""
        if self.timers is None:
            yield
            return
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if cuda:
            torch.cuda.synchronize()
        self.timers[name] = self.timers.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def _kernel_ctx(self):
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.attn_impl == "flash" else nullcontext()

    def _fused(self, q, k, v, causal):
        tq = q.shape[2]
        if self.attn_impl == "fa2":
            from flash_attn import flash_attn_func

            o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal and tq > 1)
            return o.transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal and tq > 1)

    def _eager(self, q, k, v, causal, keep_rows):
        s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        if causal and q.shape[2] > 1:
            s = s.masked_fill(self._mask(q.shape[2], k.shape[2], q.device), float("-inf"))
        p = s.float().softmax(-1).to(q.dtype)
        return p @ v, (p[:, :, -keep_rows:] if keep_rows else None)

    def _row_probs(self, q_rows, k):
        """Attention probabilities of the last `rows` queries only (no full map needed)."""
        rows, t = q_rows.shape[2], k.shape[2]
        s = (q_rows @ k.transpose(-1, -2)) * (q_rows.shape[-1] ** -0.5)
        s = s.masked_fill(self._mask(rows, t, q_rows.device), float("-inf"))
        return s.float().softmax(-1)

    # ---- prefill / decode ----------------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, embeds, P: int, n_vis: int, S: int, stages: list[tuple[int, int]], scorer: str,
                max_new: int):
        """Run the prompt through all layers, pruning visual tokens at `stages`. -> (logits, State)."""
        d = self.dims
        b, _, _ = embeds.shape
        dev = embeds.device
        st = State()
        stage_at, vis = dict(stages), n_vis
        rows_for_score = 1 if scorer == "last" else S
        x = embeds
        pos = torch.arange(x.shape[1], device=dev).expand(b, -1)
        self._masks.clear()

        for l, layer in enumerate(self.layers):
            t = x.shape[1]
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(b, t, d.heads, d.head_dim).transpose(1, 2)
            k = a.k_proj(h).view(b, t, d.kv_heads, d.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(b, t, d.kv_heads, d.head_dim).transpose(1, 2)
            q, k = _rope(q, k, *self.rot(x, pos))

            kb = k.new_empty(b, d.kv_heads, t + max_new, d.head_dim)
            vb = v.new_empty(b, d.kv_heads, t + max_new, d.head_dim)
            kb[:, :, :t], vb[:, :, :t] = k, v
            st.k.append(kb), st.v.append(vb), st.lens.append(t)

            kk, vv = self._expand(k), self._expand(v)
            want = l in stage_at
            if self.attn_impl == "eager":
                o, w = self._eager(q, kk, vv, True, rows_for_score if want else 0)
            else:
                with self._kernel_ctx():
                    o = self._fused(q, kk, vv, True)
                if want:
                    with self._t("score"):   # extra pass: fused kernels do not return probabilities
                        w = self._row_probs(q[:, :, -rows_for_score:], kk)
            x = x + a.o_proj(o.transpose(1, 2).reshape(b, t, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))

            if want:
                keep = stage_at[l]
                with self._t("select"):
                    score = w.float().mean(dim=(1, 2))[:, P:P + vis]
                    idx = score.topk(keep, dim=1).indices.sort(dim=1).values
                with self._t("gather"):
                    sel = torch.cat([torch.arange(P, device=dev).expand(b, -1), P + idx,
                                     torch.arange(P + vis, t, device=dev).expand(b, -1)], dim=1)
                    x = x.gather(1, sel[..., None].expand(-1, -1, x.shape[-1]))
                    pos = pos.gather(1, sel)
                vis = keep

        st.next_pos = embeds.shape[1]
        return self._logits(x), st

    @torch.no_grad()
    def decode_step(self, tok, st: State):
        d = self.dims
        b = tok.shape[0]
        x = self.embed(tok)[:, None]
        pos = torch.full((b, 1), st.next_pos, device=x.device, dtype=torch.long)
        cos, sin = self.rot(x, pos)
        for l, layer in enumerate(self.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(b, 1, d.heads, d.head_dim).transpose(1, 2)
            k = a.k_proj(h).view(b, 1, d.kv_heads, d.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(b, 1, d.kv_heads, d.head_dim).transpose(1, 2)
            q, k = _rope(q, k, cos, sin)
            n = st.lens[l]
            st.k[l][:, :, n:n + 1], st.v[l][:, :, n:n + 1] = k, v
            st.lens[l] = n + 1
            kk, vv = self._expand(st.k[l][:, :, :n + 1]), self._expand(st.v[l][:, :, :n + 1])
            if self.attn_impl == "eager":
                o, _ = self._eager(q, kk, vv, False, 0)
            else:
                with self._kernel_ctx():
                    o = self._fused(q, kk, vv, False)
            x = x + a.o_proj(o.transpose(1, 2).reshape(b, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        st.next_pos += 1
        return self._logits(x)

    def _logits(self, x):
        return self.head(self.norm(x[:, -1]))

    @torch.no_grad()
    def generate(self, embeds, P, n_vis, S, stages, scorer, max_new, stamp=None):
        """Greedy, fixed-length (EOS ignored so output length is controlled). -> (tokens [B, max_new], State).

        `stamp()` is called right after prefill and after every decode step (used for timing).
        """
        logits, st = self.prefill(embeds, P, n_vis, S, stages, scorer, max_new)
        toks = [logits.argmax(-1)]
        if stamp:
            stamp()
        for _ in range(max_new - 1):
            logits = self.decode_step(toks[-1], st)
            toks.append(logits.argmax(-1))
            if stamp:
                stamp()
        return torch.stack(toks, dim=1), st
