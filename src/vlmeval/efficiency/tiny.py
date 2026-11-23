"""Tiny random-weight VLMs for CPU smoke tests of the profiling pipeline (no download, no GPU)."""

from __future__ import annotations

import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration


def _clip_llama(layers: int):
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                          image_size=56, patch_size=14)
    tc = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                     num_key_value_heads=4, vocab_size=100)
    return vc, tc


def build_tiny_llava(layers: int = 12, seed: int = 0) -> LlavaForConditionalGeneration:
    torch.manual_seed(seed)
    vc, tc = _clip_llama(layers)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=3, vision_feature_layer=-2)
    return LlavaForConditionalGeneration(cfg).eval()


def build_tiny_llava_next(layers: int = 12, seed: int = 0):
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration

    torch.manual_seed(seed)
    vc, tc = _clip_llama(layers)
    cfg = LlavaNextConfig(vision_config=vc, text_config=tc, image_token_index=3, vision_feature_layer=-2,
                          image_grid_pinpoints=[[56, 56], [56, 112], [112, 56], [112, 112]])
    return LlavaNextForConditionalGeneration(cfg).eval()


def build_tiny_qwen(layers: int = 12, seed: int = 0, v25: bool = True):
    """Qwen(2.5)-VL: 112px image -> 8x8 patches -> 4x4 = 16 merged tokens; GQA 4 heads / 2 KV heads."""
    import transformers as tf

    torch.manual_seed(seed)
    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                num_key_value_heads=2, vocab_size=200, max_position_embeddings=512, bos_token_id=1, eos_token_id=2,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]})
    tok = dict(image_token_id=190, video_token_id=191, vision_start_token_id=192, vision_end_token_id=193)
    if v25:
        vis = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14,
                   spatial_merge_size=2, temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1])
        cfg = tf.Qwen2_5_VLConfig(text_config=text, vision_config=vis, **tok)
        return tf.Qwen2_5_VLForConditionalGeneration(cfg).eval()
    vis = dict(depth=2, embed_dim=32, hidden_size=64, mlp_ratio=2, num_heads=2, patch_size=14,
               spatial_merge_size=2, temporal_patch_size=2)
    cfg = tf.Qwen2VLConfig(text_config=text, vision_config=vis, **tok)
    return tf.Qwen2VLForConditionalGeneration(cfg).eval()


def build_tiny(family: str, layers: int = 12):
    """-> (model, model_cfg) for `family` (see families.py); model_cfg carries the geometry the family needs."""
    if family == "llava":
        return build_tiny_llava(layers), {}
    if family == "llava_next":
        return build_tiny_llava_next(layers), {"tiles": 2}
    if family in ("qwen2_vl", "qwen2_5_vl"):
        return build_tiny_qwen(layers, v25=family == "qwen2_5_vl"), {"image_size": 112}
    raise ValueError(family)
