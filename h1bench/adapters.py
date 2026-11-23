"""방법별 설정 적용.
원칙: 각 공식 저장소가 제공하는 설정 경로만 사용하고, 토큰 선택 로직은 수정하지 않는다.
키 이름은 공식 저장소 기준이며, step2 smoke test의 실제 남은 토큰 수로 검증한다.
값이 맞지 않으면 configs/methods.yaml 의 config_overrides 로만 조정한다."""
import importlib

N_VIS = 576

def _set(model, key, val):
    setattr(model.config, key, val)
    inner = getattr(model, "model", None)
    if inner is not None:
        setattr(inner, key, val)
        if hasattr(inner, "config"):
            setattr(inner.config, key, val)

def _apply_overrides(model, cfg, R):
    for k, v in (cfg.get("config_overrides") or {}).items():
        if isinstance(v, str) and "{R}" in v:
            v = eval(v.replace("{R}", str(R)))
        _set(model, k, v)

def baseline(model, R, cfg):
    return {"select_fn": None}

def fastv(model, R, cfg):
    # FastV 공식 fork(transformers) 설정 키
    _set(model, "use_fast_v", True)
    _set(model, "fast_v_inplace", True)
    _set(model, "fast_v_sys_length", cfg.get("sys_length", 35))
    _set(model, "fast_v_image_token_length", N_VIS)
    _set(model, "fast_v_attention_rank", R)
    _set(model, "fast_v_agg_layer", cfg.get("K", 2))
    if hasattr(model.model, "reset_fastv"):
        model.model.reset_fastv()
    _apply_overrides(model, cfg, R)
    return {"select_fn": None}

def pdrop(model, R, cfg):
    # 3단계 감소. 마지막 단계 토큰 수가 R이 되도록 단계별 비율 lambda = (R/576)^(1/3) (DECISIONS.md 참조)
    lam = (R / N_VIS) ** (1 / 3)
    _set(model, "layer_list", cfg.get("layer_list", [8, 16, 24]))
    _set(model, "image_token_ratio_list", [lam, lam ** 2, lam ** 3])
    _set(model, "image_token_ratio", lam)
    _apply_overrides(model, cfg, R)
    return {"select_fn": None}

def sparsevlm(model, R, cfg):
    _set(model, "retained_tokens", R)
    _apply_overrides(model, cfg, R)
    return {"select_fn": None}

def visionzip(model, R, cfg):
    from visionzip import visionzip as vz
    ratio = cfg.get("dominant_ratio", 0.84375)  # 공식 설정 54/64, 108/128, 162/192 과 같은 비율
    dom = int(round(R * ratio)); ctx = R - dom
    vz(model, dominant=dom, contextual=ctx)
    return {"select_fn": None, "detail": {"dominant": dom, "contextual": ctx}}

def divprune(model, R, cfg):
    _set(model, "divprune_ratio", R / N_VIS)
    _set(model, "visual_token_num", R)
    _apply_overrides(model, cfg, R)
    return {"select_fn": None}

def cdpruner(model, R, cfg):
    _set(model, "visual_token_num", R)
    _apply_overrides(model, cfg, R)
    return {"select_fn": None}

def anchorprune(model, R, cfg):
    _set(model, "visual_token_num", R)
    _apply_overrides(model, cfg, R)
    return {"select_fn": None}

ADAPTERS = dict(baseline=baseline, fastv=fastv, pdrop=pdrop, sparsevlm=sparsevlm,
                visionzip=visionzip, divprune=divprune, cdpruner=cdpruner, anchorprune=anchorprune)

def resolve_select(spec):
    """configs/methods.yaml 의 select_fn: "module.path:attr" 를 (owner, name)으로 변환."""
    if not spec:
        return None
    mod, attr = spec.split(":")
    owner = importlib.import_module(mod)
    parts = attr.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]
