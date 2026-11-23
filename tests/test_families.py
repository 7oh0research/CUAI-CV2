"""New methods (DivPrune, VisionZip, VScan, DART, downsample) and the LLaVA-NeXT / Qwen2(.5)-VL engines, on tiny
random-weight models (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from vlmeval.efficiency import flops as fl  # noqa: E402
from vlmeval.efficiency.families import FAMILIES, family_from_model, static_family  # noqa: E402
from vlmeval.efficiency.methods import BUILTIN, enc_divprune, enc_visionzip, resolve  # noqa: E402
from vlmeval.efficiency.profiler import theoretical  # noqa: E402
from vlmeval.efficiency.tiny import build_tiny  # noqa: E402

LAYERS = 6
_cache: dict = {}


def _setup(fam):
    if fam not in _cache:
        model, mc = build_tiny(fam, LAYERS)
        _cache[fam] = (model, family_from_model(model, fam, mc))
    return _cache[fam]


def _run(fam, name, impl="sdpa", budget=8, max_new=3):
    model, f = _setup(fam)
    spec = BUILTIN[name]
    eng = f.engine_cls(model, impl)
    feats = eng.visual(f.pixels(2, torch.Generator().manual_seed(0)), spec, budget, torch.Generator().manual_seed(0))
    _, stages = resolve(spec, budget, feats.shape[1], LAYERS, f.tiles)
    emb = eng.build_embeds(feats, torch.arange(4, 7)[None], torch.arange(10, 15)[None])
    toks, _ = eng.generate(emb, 3, feats.shape[1], 5, stages, spec.scorer, max_new)
    return feats, toks


@pytest.mark.parametrize("fam", FAMILIES)
@pytest.mark.parametrize("name", [n for n in BUILTIN if n != "none"])
def test_every_method_runs_and_hits_its_token_count(fam, name):
    _, f = _setup(fam)
    if not f.supports(BUILTIN[name]):
        pytest.skip("not applicable to this family")
    feats, toks = _run(fam, name)
    expect, _ = resolve(BUILTIN[name], 8, f.n_vis, LAYERS, f.tiles)
    assert feats.shape[1] == expect and toks.shape == (2, 3)


def test_qwen_rejects_cls_based_methods():
    model, f = _setup("qwen2_5_vl")
    for name in ("cls_attn", "visionzip", "vscan"):
        assert not f.supports(BUILTIN[name])
        with pytest.raises(NotImplementedError):
            f.engine_cls(model, "sdpa").visual(f.pixels(1), BUILTIN[name], 8)


def test_llava_next_runs_the_tower_per_tile():
    _, f = _setup("llava_next")
    assert f.tiles == 2 and f.n_vis == 2 * 16 and f.pixels(3).shape[:2] == (3, 2)


@pytest.mark.parametrize("name", ["dart", "fastv"])
@pytest.mark.parametrize("fam", ["llava", "qwen2_5_vl"])
def test_eager_and_sdpa_agree(fam, name):
    assert torch.equal(_run(fam, name, "eager")[1], _run(fam, name, "sdpa")[1])


def test_divprune_picks_diverse_tokens():
    base = torch.randn(1, 4, 16)
    feats = torch.cat([base + 1e-3 * torch.randn(1, 4, 16) for _ in range(4)], dim=1)   # 4 clusters x 4 copies
    keep = enc_divprune(feats, 4)
    assert keep.shape[1] == 4
    cluster = lambda x: {int(torch.cdist(x[0, i:i + 1], base[0]).argmin()) for i in range(x.shape[1])}  # noqa: E731
    assert len(cluster(keep)) == 4


def test_visionzip_merges_instead_of_dropping():
    feats = torch.randn(1, 64, 16)
    cls = torch.rand(1, 64)
    out = enc_visionzip(feats, 20, cls_scores=cls)
    assert out.shape == (1, 20, 16)


def test_dart_schedule_and_scoring_flops_independent_of_backend():
    dims, vd = fl.LLAVA15_7B, fl.CLIP_L14_336
    a = theoretical(dims, vd, BUILTIN["dart"], 64, 576, 40, "eager", 1)
    b = theoretical(dims, vd, BUILTIN["dart"], 64, 576, 40, "sdpa", 1)
    assert a["prefill_flops"] == b["prefill_flops"] and a["n_visual_final"] == 64
    f = theoretical(dims, vd, BUILTIN["fastv"], 64, 576, 40, "eager", 1)
    assert a["prefill_flops"] > f["prefill_flops"]      # same schedule, plus the pivot-similarity pass


def test_downsample_shrinks_the_vision_tower_too():
    f = static_family("llava")
    full = theoretical(f.llm, f.vdims, BUILTIN["none"], 576, 576, 40, "eager", 1)
    ds = theoretical(f.llm, f.vdims, BUILTIN["downsample"], 64, 576, 40, "eager", 1)
    rnd = theoretical(f.llm, f.vdims, BUILTIN["random"], 64, 576, 40, "eager", 1)
    assert ds["vision_flops"] < full["vision_flops"] * 0.2 and rnd["vision_flops"] == full["vision_flops"]


def test_vscan_schedule_is_two_stage():
    n_in, stages = resolve(BUILTIN["vscan"], 64, 576, 32)
    assert n_in == 96 and stages == [(15, 64)]


def test_qwen_flops_dims_are_sane():
    f = static_family("qwen2_5_vl")
    t = theoretical(f.llm, f.vdims, BUILTIN["none"], 576, 576, 40, "eager", 1)
    assert 6e12 < t["prefill_flops"] < 12e12 and f.vdims.kept_tokens == 576
    f2 = static_family("llava_next")
    full = theoretical(f2.llm, f2.vdims, BUILTIN["none"], 2880, 2880, 40, "eager", 1, f2.tiles)
    one = theoretical(static_family("llava").llm, f2.vdims, BUILTIN["none"], 576, 576, 40, "eager", 1)
    assert full["vision_flops"] == 5 * one["vision_flops"]


@pytest.mark.parametrize("fam", ["qwen2_5_vl", "qwen2_vl"])
def test_qwen_engine_matches_hf_generate(fam):
    model, f = _setup(fam)
    cfg = model.config
    eng = f.engine_cls(model, "eager")
    pix = f.pixels(1, torch.Generator().manual_seed(1))
    feats = eng.visual(pix, BUILTIN["none"], f.n_vis)
    pre, suf = torch.arange(10, 14)[None], torch.arange(20, 25)[None]
    ids = torch.cat([pre, torch.tensor([[cfg.vision_start_token_id]]),
                     torch.full((1, f.n_vis), cfg.image_token_id), suf], dim=1)
    P = pre.shape[1] + 1
    ours, _ = eng.generate(eng.build_embeds(feats, ids[:, :P], suf), P, f.n_vis, suf.shape[1], [], "last", 4)
    patches, grid = eng._patchify(pix)
    with torch.no_grad():
        ref = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=patches,
                             image_grid_thw=grid, mm_token_type_ids=(ids == cfg.image_token_id).int(),
                             max_new_tokens=4, min_new_tokens=4, do_sample=False)
    assert torch.equal(ours, ref[:, ids.shape[1]:])
