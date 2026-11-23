"""조합 1개(method, R, attn, bs, rep)를 측정해 results/raw/*.jsonl 에 기록한다.
반복마다 새 프로세스로 실행한다(step 스크립트가 호출)."""
import argparse, json, os, sys, time, warnings, traceback
import yaml, torch
from PIL import Image
from h1bench.common import CudaTimer, wrap_module, wrap_function, GpuMonitor, tensor_bytes
from h1bench.data import eff_subset
from h1bench.adapters import ADAPTERS, resolve_select
from h1bench import flops as F

def build_model(model_path, attn):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    name = get_model_name_from_path(model_path)
    tok, model, proc, _ = load_pretrained_model(model_path, None, name, device_map="cuda:0",
                                                torch_dtype=torch.float16, attn_implementation=attn)
    model.eval()
    return tok, model, proc

def build_batch(tok, model, proc, samples):
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    t0 = time.perf_counter()
    ids, imgs = [], []
    for s in samples:
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + s["question"])
        conv.append_message(conv.roles[1], None)
        ids.append(tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt"))
        imgs.append(Image.open(s["image"]).convert("RGB"))
    pix = process_images(imgs, proc, model.config).to("cuda", torch.float16)
    L = max(len(x) for x in ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.unk_token_id
    inp = torch.full((len(ids), L), pad, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):  # left padding (모든 방법 동일)
        inp[i, L - len(x):] = x; att[i, L - len(x):] = 1
    torch.cuda.synchronize()
    return inp.cuda(), att.cuda(), pix, [img.size for img in imgs], (time.perf_counter() - t0) * 1000, [len(x) for x in ids]

class StepEvents:
    """generate 의 매 토큰 직후 호출된다: 토큰별 시각과 prefill peak 메모리 기록."""
    def __init__(self):
        self.events, self.peak_prefill = [], None
    def __call__(self, input_ids, scores, **kw):
        e = torch.cuda.Event(enable_timing=True); e.record(); self.events.append(e)
        if self.peak_prefill is None:
            torch.cuda.synchronize(); self.peak_prefill = torch.cuda.max_memory_allocated()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

class LayerProbe:
    """디코더 층별 입력 시퀀스 길이(prefill 1회분)와 실제 attention 구현 기록."""
    def __init__(self, layers):
        self.lens, self.active, self.hooks = [None] * len(layers), False, []
        for i, l in enumerate(layers):
            self.hooks.append(l.register_forward_pre_hook(self._mk(i), with_kwargs=True))
        self.impl = []
        for l in layers:
            a = l.self_attn
            cfg = getattr(a, "config", None)
            self.impl.append({"class": type(a).__name__,
                              "config_impl": getattr(cfg, "_attn_implementation", None) if cfg else None})
    def _mk(self, i):
        def f(m, args, kwargs):
            if self.active and self.lens[i] is None:
                h = args[0] if args else kwargs.get("hidden_states")
                self.lens[i] = int(h.shape[1])
        return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method"); ap.add_argument("--R", type=int); ap.add_argument("--attn")
    ap.add_argument("--bs", type=int); ap.add_argument("--rep", type=int)
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg_all = yaml.safe_load(open("configs/methods.yaml"))
    mcfg = cfg_all["methods"][a.method]; grid = cfg_all["grid"]
    n = a.n or grid["n_samples"]; new_tok = grid["new_tokens"]; warm = 0 if a.smoke else grid["warmup"]
    torch.manual_seed(0)
    meta = dict(method=a.method, R=a.R, attn_requested=a.attn, batch_size=a.bs, rep=a.rep)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    mon = GpuMonitor("env/gpu_monitor.csv"); mon.start()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            tok, model, proc = build_model(cfg_all["model_path"], a.attn)
            info = ADAPTERS[a.method](model, a.R, mcfg)
            caught += [str(w.message) for w in wlist]
        torch.cuda.synchronize()
        mem_resident = torch.cuda.memory_allocated()
        timer = CudaTimer()
        vt = model.get_model().get_vision_tower(); proj = model.get_model().mm_projector
        wrap_module(timer, vt, "vision"); wrap_module(timer, proj, "projector")
        sel = resolve_select(mcfg.get("select_fn"))
        if sel: wrap_function(timer, sel[0], sel[1], "select")
        probe = LayerProbe(model.get_model().layers)
        samples = eff_subset(os.environ["MME_ROOT"], grid["n_samples"])[:n]
        batches = [samples[i:i + a.bs] for i in range(0, len(samples), a.bs)]
        if len(batches[-1]) < a.bs: batches = batches[:-1]   # 마지막 불완전 배치는 버림
        warm_b = (batches * 10)[:warm]
        recs = []
        for bi, b in enumerate(warm_b + batches):
            is_warm = bi < len(warm_b)
            inp, att, pix, sizes, t_cpu, txt_lens = build_batch(tok, model, proc, b)
            timer.reset(); se = StepEvents()
            probe.lens = [None] * len(probe.lens); probe.active = True
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            s0 = torch.cuda.Event(enable_timing=True); s0.record()
            with torch.inference_mode(), warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                out = model.generate(inp, attention_mask=att, images=pix, image_sizes=sizes, do_sample=False,
                                     num_beams=1, min_new_tokens=new_tok, max_new_tokens=new_tok,
                                     use_cache=True, stopping_criteria=[se], return_dict_in_generate=True)
                caught += [str(w.message) for w in wl]
            torch.cuda.synchronize(); probe.active = False
            if is_warm: continue
            tt = timer.totals(); ev = se.events
            prefill = s0.elapsed_time(ev[0])
            dec = [ev[i - 1].elapsed_time(ev[i]) for i in range(1, len(ev))]
            text_len = txt_lens[0] - 1
            vis_layers = [None if l is None else l - (inp.shape[1] - 1) for l in probe.lens]
            recs.append(dict(meta, batch_idx=bi - len(warm_b), n_items=len(b), t_cpu_preproc_ms=t_cpu,
                             t_vision_ms=tt.get("vision", 0) + tt.get("projector", 0), t_select_ms=tt.get("select"),
                             prefill_ms=prefill, t_llm_prefill_ms=prefill - tt.get("vision", 0) - tt.get("projector", 0),
                             decode_ms_per_tok=sum(dec) / max(len(dec), 1), n_decode=len(dec),
                             e2e_ms=s0.elapsed_time(ev[-1]) + t_cpu,
                             mem_peak_prefill_GB=se.peak_prefill / 1e9, mem_peak_total_GB=torch.cuda.max_memory_allocated() / 1e9,
                             mem_reserved_peak_GB=torch.cuda.max_memory_reserved() / 1e9,
                             kv_cache_GB=tensor_bytes(getattr(out, "past_key_values", None)) / 1e9,
                             layer_seq_len=probe.lens, layer_visual_tokens=vis_layers, text_len_padded=inp.shape[1] - 1))
        # FLOPs (해석식): 층별 실제 길이 사용, 첫 배치 기준 샘플 1개당
        r0 = recs[0]; lens = [l for l in r0["layer_seq_len"]]
        fl_llm = F.llm_prefill(lens) / a.bs if None not in lens else None
        flops_noselect = (F.vision() + fl_llm) / 1e12 if fl_llm else None
        flops_prof = None
        if a.rep == 0 and not a.smoke:
            from torch.profiler import profile, ProfilerActivity
            inp, att, pix, sizes, _, _ = build_batch(tok, model, proc, batches[0][:1])
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_flops=True) as p, torch.inference_mode():
                model.generate(inp, attention_mask=att, images=pix, image_sizes=sizes, do_sample=False,
                               max_new_tokens=1, use_cache=True)
            flops_prof = sum(e.flops for e in p.key_averages()) / 1e12
        impl_actual = sorted({x["config_impl"] or x["class"] for x in probe.impl})
        fb_msgs = [c for c in caught if "fall" in c.lower() or "eager" in c.lower() or "not support" in c.lower()]
        status = "fallback" if (fb_msgs or any(x != a.attn for x in impl_actual if x)) else "ok"
        summary = dict(meta, status=status, attn_actual="|".join(map(str, impl_actual)), attn_layers=probe.impl,
                       fallback_messages=fb_msgs[:20], mem_resident_GB=mem_resident / 1e9,
                       flops_analytic_noselect_T=flops_noselect, flops_profiler_T=flops_prof,
                       throttled=mon.throttled, adapter_detail=info.get("detail"))
        with open(a.out, "w") as f:
            f.write(json.dumps({"type": "summary", **summary}) + "\n")
            for r in recs: f.write(json.dumps({"type": "batch", **r}) + "\n")
    except torch.cuda.OutOfMemoryError as e:
        _fail(a.out, meta, "oom", e)
    except Exception as e:
        msg = repr(e).lower()
        st = "unsupported" if any(k in msg for k in ["flash", "sdpa", "attn_implementation", "output_attentions", "attention"]) else "failed"
        _fail(a.out, meta, st, e)
    finally:
        mon.stop_flag = True

def _fail(path, meta, status, e):
    with open(path, "w") as f:
        f.write(json.dumps({"type": "summary", **meta, "status": status, "error": repr(e)[:2000],
                            "traceback": traceback.format_exc()[-4000:]}) + "\n")
    print(f"[{status}] {meta} {repr(e)[:300]}", file=sys.stderr)

if __name__ == "__main__":
    main()
