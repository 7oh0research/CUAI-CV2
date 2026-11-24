"""RTX 5090에서 PyTorch가 실제로 동작하는지 검증한다. 실패하면 0이 아닌 코드로 종료."""
import sys, json, platform
import torch
info = {"python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
if not torch.cuda.is_available():
    print(json.dumps(info, indent=2)); sys.exit("CUDA를 사용할 수 없음")
cap = torch.cuda.get_device_capability(0)
info.update(gpu=torch.cuda.get_device_name(0), capability=f"sm_{cap[0]}{cap[1]}",
            arch_list=torch.cuda.get_arch_list(),
            mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
need = f"sm_{cap[0]}{cap[1]}"
if need not in info["arch_list"] and f"compute_{cap[0]}{cap[1]}" not in info["arch_list"]:
    print(json.dumps(info, indent=2))
    sys.exit(f"설치된 PyTorch에 {need} 커널이 없음. cu128 이상 wheel로 다시 설치하세요.")
# 실제 커널 실행 (fp16 matmul + SDPA)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = (a @ a).float().sum().item()
q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
info["matmul_ok"] = bool(b == b); info["sdpa_ok"] = bool(torch.isfinite(o).all().item())
import transformers; info["transformers"] = transformers.__version__
print(json.dumps(info, indent=2))
if not (info["matmul_ok"] and info["sdpa_ok"]): sys.exit("GPU 연산 검증 실패")
print("OK")
