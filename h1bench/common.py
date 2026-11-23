import json, os, time, threading
import torch

class CudaTimer:
    """구간 시간 측정: hook으로 시작/끝 이벤트를 쌓고, 끝에서 한 번에 동기화해 합산한다."""
    def __init__(self):
        self.ev = {}
    def start(self, k):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.ev.setdefault(k, []).append([e, None])
    def stop(self, k):
        e = torch.cuda.Event(enable_timing=True); e.record()
        if k in self.ev and self.ev[k] and self.ev[k][-1][1] is None:
            self.ev[k][-1][1] = e
    def reset(self):
        self.ev = {}
    def totals(self):
        torch.cuda.synchronize()
        return {k: sum(a.elapsed_time(b) for a, b in v if b is not None) for k, v in self.ev.items()}

def wrap_module(timer, module, key, first_call_only=None):
    def pre(m, a, kw=None):
        timer.start(key)
    def post(m, a, o):
        timer.stop(key)
    h1 = module.register_forward_pre_hook(lambda m, a: pre(m, a))
    h2 = module.register_forward_hook(post)
    return [h1, h2]

def wrap_function(timer, owner, name, key):
    """owner.name 함수 호출 구간을 잰다(토큰 선택 함수 등)."""
    fn = getattr(owner, name)
    def w(*a, **kw):
        timer.start(key)
        out = fn(*a, **kw)
        timer.stop(key)
        return out
    setattr(owner, name, w)
    return lambda: setattr(owner, name, fn)

class GpuMonitor(threading.Thread):
    def __init__(self, path, interval=1.0):
        super().__init__(daemon=True)
        self.path, self.interval, self.stop_flag, self.throttled = path, interval, False, False
    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            return
        bad = 0
        for attr in ["nvmlClocksThrottleReasonSwThermalSlowdown", "nvmlClocksThrottleReasonHwThermalSlowdown",
                     "nvmlClocksThrottleReasonSwPowerCap", "nvmlClocksThrottleReasonHwSlowdown"]:
            bad |= getattr(pynvml, attr, 0)
        new = not os.path.exists(self.path)
        with open(self.path, "a") as f:
            if new:
                f.write("time,temp_C,sm_clock_MHz,mem_clock_MHz,power_W,throttle_mask\n")
            while not self.stop_flag:
                try:
                    t = pynvml.nvmlDeviceGetTemperature(h, 0)
                    sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    mc = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                    pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
                    tr = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
                    if tr & bad:
                        self.throttled = True
                    f.write(f"{time.time():.1f},{t},{sm},{mc},{pw:.1f},{tr}\n"); f.flush()
                except Exception:
                    pass
                time.sleep(self.interval)

def tensor_bytes(obj):
    """KV cache 등 중첩 구조 안의 텐서 바이트 합."""
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        return tensor_bytes(obj.key_cache) + tensor_bytes(obj.value_cache)
    if hasattr(obj, "layers"):  # 최신 transformers Cache
        s = 0
        for l in obj.layers:
            s += tensor_bytes(getattr(l, "keys", None)) + tensor_bytes(getattr(l, "values", None))
        return s
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(o) for o in obj)
    return 0

def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
