"""Model families for the H1 study: loading, prompt layout, engine class, visual-token geometry and vision FLOPs dims.

  llava        LLaVA-1.5-7B   : 1 tile of 336px CLIP-L/14 -> 576 tokens
  llava_next   LLaVA-NeXT-7B  : AnyRes, `tiles` x 576 tokens (5 -> 2880, the maximum without unpadding). The ViT
                                runs once per tile; `image_newline` tokens are not modelled (<5% of the sequence).
  qwen2_vl     Qwen2-VL-7B    : native-resolution ViT, 672px -> 576 tokens, M-RoPE (no CLS => no CLS-based methods)
  qwen2_5_vl   Qwen2.5-VL-7B  : same, window-attention ViT

Selected through `model.family` in the config (default `llava`); `model.tiles` / `model.image_size` override the
geometry. Visual front ends are fed *synthetic* normalised pixels: latency depends on shapes, not on content.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import flops as fl
from .engine import Engine
from .engine_qwen import NOT_APPLICABLE, QwenEngine
from .methods import MethodSpec

FAMILIES = ("llava", "llava_next", "qwen2_vl", "qwen2_5_vl")
_QWEN = ("qwen2_vl", "qwen2_5_vl")


@dataclass(frozen=True)
class Family:
    name: str
    engine_cls: type
    n_vis: int                      # visual tokens before any pruning (summed over tiles)
    tiles: int
    img: int                        # input image side in pixels (per tile)
    vdims: fl.VisionDims
    llm: fl.LLMDims | None = None
    unsupported: tuple = ()         # encoder methods that cannot run on this family

    def pixels(self, bs: int, gen=None) -> torch.Tensor:
        shape = (bs, self.tiles, 3, self.img, self.img) if self.tiles > 1 else (bs, 3, self.img, self.img)
        return torch.randn(*shape, generator=gen)

    def supports(self, spec: MethodSpec) -> bool:
        return not (spec.kind == "encoder" and spec.encoder in self.unsupported)


def static_family(name: str, model_cfg: dict | None = None) -> Family:
    """Published dimensions, no model needed (used by `--flops-only` and planning)."""
    mc = model_cfg or {}
    if name == "llava":
        return Family(name, Engine, 576, 1, 336, fl.CLIP_L14_336, fl.LLAVA15_7B)
    if name == "llava_next":
        t = int(mc.get("tiles", 5))
        return Family(name, Engine, 576 * t, t, 336, fl.CLIP_L14_336, fl.LLAVA15_7B)
    if name in _QWEN:
        img = int(mc.get("image_size", 672))
        n = (img // 28) ** 2
        base = fl.QWEN25VL_VIT if name == "qwen2_5_vl" else fl.QWEN2VL_VIT
        return Family(name, QwenEngine, n, 1, img, base.resized(n), fl.QWEN2_7B, NOT_APPLICABLE)
    raise ValueError(f"unknown model family {name!r}; choose from {FAMILIES}")


def family_from_model(model, name: str, model_cfg: dict | None = None) -> Family:
    """Same geometry, but with the vision FLOPs dims read from the loaded model's config."""
    mc = model_cfg or {}
    if name in ("llava", "llava_next"):
        vc = model.config.vision_config
        side = vc.image_size // vc.patch_size
        tiles = int(mc.get("tiles", 5)) if name == "llava_next" else 1
        hidden = model.config.text_config.hidden_size
        return Family(name, Engine, side * side * tiles, tiles, vc.image_size,
                      fl.VisionDims.from_config(vc, hidden))
    if name in _QWEN:
        img = int(mc.get("image_size", 672))
        n = (img // 28) ** 2
        vc = model.config.vision_config
        return Family(name, QwenEngine, n, 1, img,
                      fl.VisionDims.from_qwen_config(vc, model.config.text_config.hidden_size, n), None, NOT_APPLICABLE)
    raise ValueError(f"unknown model family {name!r}; choose from {FAMILIES}")


def load_model(name: str, family: str, dtype):
    import transformers as tf

    cls = {"llava": "LlavaForConditionalGeneration", "llava_next": "LlavaNextForConditionalGeneration",
           "qwen2_vl": "Qwen2VLForConditionalGeneration", "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration"}[family]
    cls = getattr(tf, cls)
    try:
        model = cls.from_pretrained(name, dtype=dtype)
    except TypeError:  # transformers < 4.56
        model = cls.from_pretrained(name, torch_dtype=dtype)
    return model.cuda().eval()


def prompt_ids(tok, family: str, question: str, device="cuda"):
    """-> (prefix_ids, suffix_ids): the text around the visual tokens, tokenised once and reused for every sample."""
    if family in _QWEN:
        pre = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>"
        suf = f"<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
        add_pre = False
    else:
        pre, suf, add_pre = "USER: ", f"\n{question} ASSISTANT:", True
    a = tok(pre, add_special_tokens=add_pre, return_tensors="pt")["input_ids"]
    b = tok(suf, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return a.to(device), b.to(device)
