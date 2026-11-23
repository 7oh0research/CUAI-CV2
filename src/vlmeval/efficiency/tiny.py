"""A tiny random-weight LLaVA for CPU smoke tests of the profiling pipeline (no download, no GPU)."""

from __future__ import annotations

import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration


def build_tiny_llava(layers: int = 12, seed: int = 0) -> LlavaForConditionalGeneration:
    torch.manual_seed(seed)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                          image_size=56, patch_size=14)
    tc = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                     num_key_value_heads=4, vocab_size=100)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=3, vision_feature_layer=-2)
    return LlavaForConditionalGeneration(cfg).eval()
