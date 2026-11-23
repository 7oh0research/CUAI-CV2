# FLOPs 정의
- FLOPs = 2 x MAC. 샘플 1개, prefill 1회 기준.
- LLM 층 하나(시퀀스 길이 n): 2n(4d^2 + 3dm) + 4n^2 d,  d=4096, m=11008, 32층.
- 층별 n 은 실제 측정한 층 입력 길이(layer_seq_len) 사용. 층 안에서 토큰이 줄어드는 방법은 해당 층 입력 길이 기준.
- Vision: CLIP ViT-L/14-336, 577 토큰, 23층 + projector(2층 MLP).
- lm_head 는 마지막 토큰 1개만 계산.
- flops_analytic_noselect_T: selection 연산 제외. flops_analytic_T: 현재는 noselect 와 같고, selection FLOPs 는 profiler 값과의 차이로 확인.
- flops_profiler_T: torch.profiler(with_flops=True) 합계, max_new_tokens=1. profiler 는 matmul/conv 계열만 집계한다.
