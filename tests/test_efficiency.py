"""Tests for the H1 efficiency harness. Uses a tiny random-weight LLaVA on CPU (no GPU / downloads)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration  # noqa: E402

from vlmeval.efficiency import flops as fl  # noqa: E402
from vlmeval.efficiency.engine import Engine  # noqa: E402
from vlmeval.efficiency.methods import BUILTIN, MethodSpec, Stage, layer_tokens, resolve  # noqa: E402

N_VIS, P, S, LAYERS = 16, 3, 5, 6


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                          image_size=56, patch_size=14)
    tc = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=LAYERS, num_attention_heads=4,
                     num_key_value_heads=4, vocab_size=100)
    return LlavaForConditionalGeneration(LlavaConfig(vision_config=vc, text_config=tc, image_token_index=3,
                                                     vision_feature_layer=-2)).eval()


def _inputs(engine, spec, budget, b=2):
    pix = torch.randn(b, 3, 56, 56, generator=torch.Generator().manual_seed(1))
    feats = engine.visual(pix, spec, budget, gen=torch.Generator().manual_seed(0))
    pre, suf = torch.arange(4, 4 + P)[None], torch.arange(10, 10 + S)[None]
    return engine.build_embeds(feats, pre, suf), feats.shape[1]


def _reference_tokens(model, embeds, n):
    """Greedy decode by re-running the full HF language model on the growing sequence."""
    lm = model.model.language_model
    emb, out = embeds, []
    for _ in range(n):
        h = lm(inputs_embeds=emb).last_hidden_state[:, -1]
        tok = model.lm_head(h).argmax(-1)
        out.append(tok)
        emb = torch.cat([emb, lm.embed_tokens(tok)[:, None]], dim=1)
    return torch.stack(out, 1)


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_engine_matches_hf_without_pruning(model, impl):
    eng = Engine(model, impl)
    embeds, nv = _inputs(eng, BUILTIN["none"], N_VIS)
    toks, st = eng.generate(embeds, P, nv, S, [], "last", max_new=5)
    assert torch.equal(toks, _reference_tokens(model, embeds, 5))
    assert st.lens == [embeds.shape[1] + 4] * LAYERS


def test_eager_and_sdpa_agree_on_pruning(model):
    spec = MethodSpec("t", "inllm", stages=(Stage(1, 1.0), Stage(3, 0.5)))
    _, stages = resolve(spec, 8, N_VIS, LAYERS)
    assert stages == [(1, 8), (3, 4)]
    outs = []
    for impl in ("sdpa", "eager"):
        eng = Engine(model, impl)
        embeds, nv = _inputs(eng, spec, 8)
        outs.append(eng.generate(embeds, P, nv, S, stages, "text", max_new=4))
    assert torch.equal(outs[0][0], outs[1][0])
    # layers <=1 keep all tokens, layers 2-3 keep 8 visual, layers >=4 keep 4 visual (+ 4 decode-cache slots)
    n_text = P + S
    assert outs[0][1].lens == [n_text + N_VIS + 3] * 2 + [n_text + 8 + 3] * 2 + [n_text + 4 + 3] * 2


def test_keeping_everything_equals_no_pruning(model):
    eng = Engine(model, "sdpa")
    embeds, nv = _inputs(eng, BUILTIN["none"], N_VIS)
    base, _ = eng.generate(embeds, P, nv, S, [], "last", max_new=4)
    pruned, _ = eng.generate(embeds, P, nv, S, [(1, N_VIS)], "last", max_new=4)
    assert torch.equal(base, pruned)


def test_pruning_changes_output(model):
    eng = Engine(model, "eager")
    embeds, nv = _inputs(eng, BUILTIN["none"], N_VIS)
    full, _ = eng.prefill(embeds, P, nv, S, [], "last", 1)
    pruned, _ = eng.prefill(embeds, P, nv, S, [(1, 2)], "last", 1)
    assert not torch.allclose(full, pruned)


@pytest.mark.parametrize("name", ["random", "avgpool", "cls_attn"])
def test_encoder_methods_token_count(model, name):
    eng = Engine(model, "sdpa")
    pix = torch.randn(2, 3, 56, 56)
    feats = eng.visual(pix, BUILTIN[name], 4, gen=torch.Generator().manual_seed(0))
    assert feats.shape[1] == resolve(BUILTIN[name], 4, N_VIS, LAYERS)[0] == 4


def test_cls_attn_prefers_high_scores():
    from vlmeval.efficiency.methods import enc_cls_attn
    f = torch.arange(10.0)[None, :, None].expand(1, 10, 2)
    s = torch.zeros(1, 10)
    s[0, [2, 7]] = 1
    assert enc_cls_attn(f, 2, cls_scores=s)[0, :, 0].tolist() == [2.0, 7.0]


def test_layer_tokens_schedule():
    assert layer_tokens(10, 100, [(1, 50), (3, 20)], 6) == [110, 110, 60, 60, 30, 30]


def test_flops_match_published_scale_for_llava15():
    # LLaVA-1.5-7B prefill ~ 576 visual + ~40 text tokens; AnchorPrune reports 8.94 TFLOPs (incl. vision).
    d = fl.LLAVA15_7B
    full = fl.prefill_flops(d, [616] * 32) + fl.vision_flops(fl.CLIP_L14_336)
    assert 8.0e12 < full < 9.6e12
    small = fl.prefill_flops(d, [64 + 40] * 32) + fl.vision_flops(fl.CLIP_L14_336)
    assert small < full / 3


def test_decode_flops_slope_is_exact():
    d = fl.LLAVA15_7B
    a = fl.decode_flops(d, [100] * 32)
    b = fl.decode_flops(d, [101] * 32)
    assert b - a == fl.decode_slope(d)


def test_analysis_detects_inversions_on_synthetic():
    from vlmeval.efficiency.analysis import expand, settings
    from vlmeval.efficiency.synthetic import make_synthetic

    st, _, pairs = settings(expand(make_synthetic(repeats=4)), n_boot=30)
    assert set(st.phase) >= {"prefill", "decode", "e2e", "memory"}
    assert st.tau.between(-1, 1).all()
    # sparsevlm_lite is simulated as slower than its FLOPs suggest, so it must be reversed against fastv/pdrop.
    inv = pairs[pairs.method_a == "sparsevlm_lite"]
    assert {"fastv", "pdrop"} <= set(inv.method_b)
    assert (pairs.flops_ratio != 1).all()


def test_forced_flash_matches_sdpa_or_is_unsupported(model):
    eng_s, eng_f = Engine(model, "sdpa"), Engine(model, "flash")
    embeds, nv = _inputs(eng_s, BUILTIN["none"], N_VIS)
    ref, _ = eng_s.generate(embeds, P, nv, S, [], "last", max_new=3)
    try:
        out, _ = eng_f.generate(embeds, P, nv, S, [], "last", max_new=3)
    except RuntimeError:   # no flash kernel for this device/dtype: must raise, never silently fall back
        return
    assert torch.equal(out, ref)


def test_region_timers_only_in_pruning_stages(model):
    eng = Engine(model, "sdpa")
    embeds, nv = _inputs(eng, BUILTIN["none"], N_VIS)
    eng.timers = {}
    eng.prefill(embeds, P, nv, S, [], "last", 1)
    assert eng.timers == {}
    eng.prefill(embeds, P, nv, S, [(1, 8)], "last", 1)
    assert set(eng.timers) == {"score", "select", "gather"}


def test_kernel_classification():
    from vlmeval.efficiency.kernels import classify, kernel_ok

    assert classify(["aten::linear", "aten::_scaled_dot_product_flash_attention"]) == "sdpa_flash"
    assert classify(["void pytorch_flash::flash_fwd_kernel"]) == "sdpa_flash"
    assert classify(["void flash_fwd_kernel<Flash_fwd_kernel_traits>"]) == "fa2"
    assert classify(["fmha_cutlassF_f16_aligned_64x64_rf_sm80"]) == "sdpa_efficient"
    assert classify(["aten::softmax", "aten::bmm"]) == "eager"
    assert kernel_ok("sdpa", "sdpa_efficient") and not kernel_ok("flash", "sdpa_efficient")   # silent fallback caught
    assert not kernel_ok("eager", "sdpa_flash")


def test_expand_grids_fractional_design():
    from vlmeval.efficiency.profiler import expand_grids

    cfg = {"methods": ["fastv", "random"], "budgets": [64, 32], "batch_sizes": [1, 4], "attn_impls": ["eager", "sdpa"],
           "grids": [{"budgets": [64]}, {"budgets": [32], "attn_impls": ["sdpa"], "batch_sizes": [1]}],
           "shuffle_seed": 1}
    combos = expand_grids(cfg, 576)
    assert len(combos) == len(set(combos))
    assert sum(c[2] == "none" for c in combos) == 4          # once per (backend, batch)
    assert {c[3] for c in combos if c[0] == "eager"} == {576, 64}
    assert {(c[1], c[3]) for c in combos if c[0] == "sdpa" and c[3] == 32} == {(1, 32)}
    assert combos == expand_grids(cfg, 576)                  # deterministic shuffle


def test_permutation_test_separates_conditions():
    import numpy as np
    import pandas as pd

    from vlmeval.efficiency.analysis import SETTING, perm_test_tau

    rng = np.random.default_rng(0)
    rows = []
    for bs, noise in ((1, 0.01), (16, 5.0)):   # bs=16: measured order uncorrelated with FLOPs
        for rep in range(20):
            for i, m in enumerate("abcde"):
                rows.append(dict(model="m", device="d", attn_impl="sdpa", batch_size=bs, budget=64, phase="e2e",
                                 L=32, method=m, rep=f"1-{rep}", flops=float(i), value=i + rng.normal(0, noise)))
    long = pd.DataFrame(rows)
    assert set(SETTING) <= set(long.columns)
    r = perm_test_tau(long, {"batch_size": 1}, {"batch_size": 16}, n_perm=500)
    assert r["tau_a"] > 0.9 and r["tau_b"] < 0.6 and r["p"] < 0.01
