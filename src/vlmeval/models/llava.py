"""LLaVA-1.5 wrapper that injects compressed visual tokens through `inputs_embeds`.

Because we build the embedding sequence ourselves, any compression method that returns a
[keep, D] tensor works unchanged, and the number of visual tokens is exactly what we report.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LogitsProcessor, LogitsProcessorList

from ..methods import METHODS, Context

TOTAL_TOKENS = 576  # 24 x 24 CLIP-L/14-336 patches


class _FirstToken(LogitsProcessor):
    """Timestamps the first decoding step = end of prefill (time to first token)."""

    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            torch.cuda.synchronize()
            self.t = time.perf_counter()
        return scores


def _sub(model, name):
    return getattr(model, name, None) or getattr(model.model, name)


class LlavaRunner:
    def __init__(self, name: str, dtype: str = "float16", attn_implementation: str = "sdpa"):
        self.processor = AutoProcessor.from_pretrained(name)
        self.tok = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_pretrained(
            name, torch_dtype=getattr(torch, dtype), attn_implementation=attn_implementation
        ).cuda().eval()
        self.vt = _sub(self.model, "vision_tower")
        self.proj = _sub(self.model, "multi_modal_projector")
        self.embed = self.model.get_input_embeddings()

    @torch.no_grad()
    def _visual(self, image):
        pix = self.processor.image_processor(image, return_tensors="pt")["pixel_values"]
        pix = pix.to("cuda", self.model.dtype)
        hs = self.vt(pix, output_hidden_states=True).hidden_states
        feats = self.proj(hs[-2][:, 1:])[0]  # LLaVA-1.5 uses the penultimate layer, no CLS
        return feats, self._cls_scores(hs)

    def _cls_scores(self, hs) -> torch.Tensor:
        """Head-averaged CLS->patch attention of the last layer whose output LLaVA consumes."""
        layer = self.vt.vision_model.encoder.layers[-2]
        a = layer.self_attn
        h = layer.layer_norm1(hs[-3])
        b, n, _ = h.shape
        heads, d = a.num_heads, a.head_dim
        q = a.q_proj(h[:, :1]).view(b, 1, heads, d).transpose(1, 2)
        k = a.k_proj(h).view(b, n, heads, d).transpose(1, 2)
        att = ((q @ k.transpose(-1, -2)) * d**-0.5).float().softmax(-1).mean(1)
        return att[0, 0, 1:]

    def _ids(self, text: str, special: bool) -> torch.Tensor:
        ids = self.tok(text, add_special_tokens=special, return_tensors="pt")["input_ids"]
        return ids.to("cuda")

    @torch.no_grad()
    def answer(self, image, question: str, method: str | None, ratio: float,
               max_new_tokens: int = 16, seed: int = 0) -> dict:
        """method=None runs text-only (blind); method='none' keeps all visual tokens."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        parts = [self.embed(self._ids("USER: ", True))]
        n_vis = 0
        if method is not None:
            feats, cls_scores = self._visual(image)
            g = torch.Generator().manual_seed(seed)
            keep = TOTAL_TOKENS if method == "none" else max(1, round(TOTAL_TOKENS * (1 - ratio)))
            feats = METHODS[method](feats, Context(keep=keep, cls_scores=cls_scores, generator=g))
            n_vis = feats.shape[0]
            parts.append(feats[None])
            suffix = f"\n{question} ASSISTANT:"
        else:
            suffix = f"{question} ASSISTANT:"
        parts.append(self.embed(self._ids(suffix, False)))
        embeds = torch.cat(parts, dim=1)

        ft = _FirstToken()
        out = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device="cuda"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([ft]),
            pad_token_id=self.tok.eos_token_id,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return {
            "pred": self.tok.decode(out[0], skip_special_tokens=True).strip(),
            "n_visual_tokens": n_vis,
            "prefill_ms": (ft.t - t0) * 1000,
            "total_ms": (t1 - t0) * 1000,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
