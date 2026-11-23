import json, os, subprocess, torch
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return repr(e)
p = torch.cuda.get_device_properties(0)
info = dict(gpu_name=p.name, total_mem_GB=p.total_memory / 1e9, compute_capability=f"{p.major}.{p.minor}",
            driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
            vbios=sh("nvidia-smi --query-gpu=vbios_version --format=csv,noheader"),
            max_clocks=sh("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv,noheader"),
            power_limit=sh("nvidia-smi --query-gpu=power.limit --format=csv,noheader"),
            gpu_processes=sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
            cuda_torch=torch.version.cuda, torch=torch.__version__)
for m in ["transformers", "flash_attn", "accelerate"]:
    try: info[m] = __import__(m).__version__
    except Exception: info[m] = None
r = sh("nvidia-smi -lgc 2520,2520 2>&1"); info["clock_lock"] = r
info["clock_locked"] = "error" not in r.lower() and "insufficient" not in r.lower() and r != ""
os.makedirs("env", exist_ok=True)
json.dump(info, open("env/env.json", "w"), indent=2)
print(json.dumps(info, indent=2))
if "4090" not in p.name or p.total_memory < 40e9:
    with open("DECISIONS.md", "a") as f:
        f.write(f"- GPU 실측값 / {p.name}, {p.total_memory/1e9:.1f} GB / 사용자 제공값(RTX 4090 48GB)과 다름, 실측값 기준 진행\n")
