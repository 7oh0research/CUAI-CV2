"""FLOPs 해석식 (FLOPs = 2 x MAC). 정의는 flops/README.md."""
D, M, L = 4096, 11008, 32           # LLaMA-2-7B (Vicuna-7B)
VD, VM, VL, VN = 1024, 4096, 24, 577 # CLIP ViT-L/14-336 (LLaVA는 마지막에서 두 번째 층 출력 사용 -> 23층)
VOCAB = 32000

def llm_layer(n):
    return 2 * n * (4 * D * D + 3 * D * M) + 2 * 2 * n * n * D

def llm_prefill(layer_tokens):
    """layer_tokens: 층별 전체 시퀀스 길이(텍스트+비전) 리스트(길이 32)."""
    return sum(llm_layer(n) for n in layer_tokens) + 2 * layer_tokens[-1] * D * VOCAB * 0 + 2 * D * VOCAB

def vision():
    per = 2 * VN * (4 * VD * VD + 2 * VD * VM) + 2 * 2 * VN * VN * VD
    proj = 2 * 576 * (VD * D + D * D)
    return (VL - 1) * per + proj
