# H1 실험: 이론 효율(FLOPs)과 실측 효율(latency/메모리)의 괴리

## 가설

> 시각 토큰 효율화 방법들을 FLOPs(또는 토큰 수)로 매긴 순위는 실측 latency/메모리 순위와 일치하지 않으며,
> 배치 크기, 출력 길이, attention 구현(FlashAttention 계열 sdpa vs eager), 단계(prefill vs decode)에 따라 순위가 체계적으로 뒤바뀐다.

## 선행 연구와 Gap

| 논문 | 관찰 | 한계 |
| --- | --- | --- |
| Wen et al., ACL Findings'25 (Sec 7.1, Table 7) | LLaVA-NeXT-7B: SparseVLM은 FastV보다 FLOPs가 2.8% 높은데 latency는 26.8% 길다. 원인: attention map이 필요한 방법은 FlashAttention 불가, 다층 pruning 오버헤드 | 방법 3개, 단일 설정의 사례 수준 |
| AnchorPrune (arXiv 2607.07033, 부록 Table S8) | LLaVA-1.5-7B, 32 tokens: SparseVLM decode 31.4 ms/token, 메모리 18.1 GB로 baseline(28.0 / 14.5 GB)보다 나쁨 | 자기 방법의 효율 비교이며 평가 관행 비판이 목적이 아님 |
| VisionDrop (AAAI'26, Table 4) | RTX 3090에서 FLOPs 6.0x 감소 vs 실측 2.7x 가속 | 이 차이를 문제로 다루지 않음 |

**Gap:** 배치 크기와 출력 길이를 바꾸며 FLOPs 순위와 latency 순위가 체계적으로 뒤바뀌는지(순위 상관)를 본 연구는 없다. prefill과 decode는 별도 축으로 측정한다.

## 실험 설계

- 모델: LLaVA-1.5-7B, fp16, **양자화 없음** (양자화 커널이 latency를 왜곡). 24GB+ GPU 권장.
- 방법 (모두 training-free):
  - 기준: `none` (576 tokens)
  - 인코더 측: `random`, `avgpool`, `cls_attn` (CLS attention top-k) — attention 구현 불변
  - LLM 내부: `fastv` (K=1 이후), `pdrop` (7/15/23층, PyramidDrop 스타일), `sparsevlm_lite` (2/6/15층, 텍스트 query 기반 점수)
  - **주의:** LLM 내부 방법은 공식 코드가 아닌 최소 재구현이다 (`sparsevlm_lite`에는 recycling/merging 없음). 공식 구현으로 재현하려면 `custom_methods`로 스케줄을 조정하거나 교체한다.
- 축: 방법 x budget(192/144/64) x 배치(1/4/16) x attention(sdpa/eager), 출력 64 tokens (L=1/16/64는 per-step 기록에서 파생).
- 측정: `vision_ms`, `prefill_ms`, 스텝별 `decode_ms`, `peak_mem_mb`, `kv_mb`. warmup 3회, 반복 10회, CUDA event 사용, OOM은 기록 후 건너뜀, 중단 후 재개 지원.
- 엔진: HF 모델 가중치만 사용하는 자체 추론 루프(`efficiency/engine.py`). 같은 코드 경로에서 sdpa/eager만 바꾸고, LLM 내부 pruning과 KV 길이를 직접 제어한다. pruning 없을 때 HF 결과와 일치함을 테스트로 확인했다.
- 이론 FLOPs: 해석적 계산(`efficiency/flops.py`). dense attention, LM head는 마지막 위치만, 비전 FLOPs는 projector 포함. FLOPs는 attn 구현과 무관하지만 latency는 무관하지 않다. 이것이 H1의 핵심 관찰 대상이다. LLaVA-1.5 전체 prefill 약 8.2~8.5T (AnchorPrune 보고 8.94T). sdpa 모드의 LLM 내부 방법은 별도 scoring pass의 FLOPs도 포함한다.

## 지표

설정(모델, attn, 배치, budget, phase, L)마다:
- FLOPs 순위 vs 실측 순위의 Kendall tau (반복 bootstrap 신뢰구간), Spearman
- **신뢰할 만한 순위 역전 쌍:** FLOPs 차이 >5%, 실측 순서가 bootstrap의 95% 이상에서 반대, 실측 차이 >2%. `method_a`는 FLOPs가 더 적은데 실측이 더 느리거나 큰 방법.
- `slower_than_baseline`: 압축했는데 `none`보다 느린/큰 경우
- realization = 실측 속도향상 / FLOPs 속도향상
- phase: prefill, decode, e2e(L), memory

decode처럼 FLOPs 차이가 거의 없는 구간에서는 tau가 동률에 좌우되므로 `flops_spread`와 역전 쌍 수를 함께 본다.

## 실행

```bash
python scripts/profile_efficiency.py --flops-only                          # GPU 없이 FLOPs 표
python scripts/profile_efficiency.py --config configs/h1.yaml              # GPU, 재개 가능
python scripts/analyze_h1.py --records outputs/h1/records.jsonl --out outputs/h1/report
```

파이프라인 검증용: `--tiny`(랜덤 가중치 소형 LLaVA, CPU, 시간 값은 의미 없음), `analyze_h1.py --synthetic`(시뮬레이션 latency).

산출물: `report.md`, `settings.csv`, `methods.csv`, `inversions.csv`, `tau_heatmap.png`, `flops_vs_latency.png`.

## 현재 상태와 한계

- 개발 머신(8GB GPU)에서는 7B 실험을 못 돌려 **실측 결과는 아직 없다.** 파이프라인은 소형 모델/합성 데이터로만 검증했고, CUDA 타이밍 경로는 GPU 머신에서 처음 실행된다.
- 엔진은 transformers 5.24에서 테스트했다.
- LLM 내부 방법의 공식 구현과 latency 차이가 있을 수 있으므로, 본 연구 단계에서 공식 코드로 재확인한다.
