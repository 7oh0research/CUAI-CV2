"""Record host/GPU details without changing GPU clocks or requiring root privileges."""
import json
import os
import subprocess


def sh(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return repr(exc)

info = {
    "python": sh("python --version"),
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    "gpu_name": None,
    "cuda_available": False,
    "torch": None,
    "cuda_torch": None,
    "transformers": None,
    "flash_attn": False,
}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_torch"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, total_mem_GB=round(p.total_memory / 1e9, 2),
                    compute_capability=f"{p.major}.{p.minor}")
except Exception as exc:
    info["torch_error"] = repr(exc)
try:
    import transformers
    info["transformers"] = transformers.__version__
except Exception as exc:
    info["transformers_error"] = repr(exc)
try:
    import flash_attn
    info["flash_attn"] = True
    info["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
except Exception as exc:
    info["flash_attn_error"] = repr(exc)

os.makedirs("env", exist_ok=True)
with open("env/env.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
if not info["cuda_available"]:
    raise SystemExit("[ERROR] PyTorch cannot access CUDA. Check NVIDIA driver, GPU visibility and PyTorch CUDA build.")
