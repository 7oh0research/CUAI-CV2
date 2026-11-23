# H1 실험 프로토콜: 이론 효율 vs 실측 효율 (사전 등록)

> 이 문서는 **결과를 보기 전에** 고정하는 프로토콜이다. 지지/반증 기준(τ 임계값), 요인 설계, 주장-근거 대응을
> 먼저 못 박아 두어, 결과에 맞춰 기준을 바꾸는 일을 막는다. 하네스 구현 설명은
> [`h1_experiment.md`](h1_experiment.md), 원격 GPU(Vast.ai RTX 5090) 설정은 [`remote_gpu.md`](remote_gpu.md).

## 0. 가설과 판정 기준

**H1.** 이론 효율(토큰 수, FLOPs)과 실측 효율(latency, throughput, memory)은 괴리된다. FLOPs 기준 방법 순위와
실측 기준 순위는 일치하지 않으며, 이 불일치는 실행 조건(attention 커널, 배치 크기, 출력 길이)에 따라 체계적으로 달라진다.

| 판정 | 기준 (setting = model × device × backend × batch × budget × phase × L) |
| --- | --- |
| **지지** | 대부분의 setting에서 Kendall τ의 bootstrap 95% CI **상한 < 0.6**, 그리고 배치/출력 길이/백엔드를 바꿀 때 τ 또는 순위가 바뀐다(순열 검정 p < 0.05) |
| **반증** | **모든** setting에서 τ CI **하한 ≥ 0.9** → H1 기각, "FLOPs는 충분한 proxy"로 보고 |
| 중간 | 위 둘 모두 아님 → 조건부 결과로 보고(어느 조건에서 깨지는지를 기술) |

- 임계값은 `analysis.H1_SUPPORT_TAU = 0.6`, `H1_REFUTE_TAU = 0.9`로 코드에 고정되어 있고, `report.md` 상단에 판정 입력값
  (support/refute 비율, 신뢰할 수 있는 순위 역전 수)이 자동 출력된다.
- τ는 방법 간 FLOPs가 거의 같을 때(예: decode) 동률 잡음에 지배된다. 그래서 **신뢰할 수 있는 역전 쌍**
  (FLOPs 차이 > 5%, bootstrap 95% 이상에서 실측 순서 반대, 실측 차이 > 2%)을 τ와 함께 보고한다.

## 1. 실험 요인과 수준

| 요인 | 수준 | 비고 |
| --- | --- | --- |
| 모델 | **LLaVA-1.5-7B** (576 tok) | LLaVA-NeXT-7B(2880 tok), Qwen2.5-VL-7B는 **미구현**, §6 참조 |
| 방법 | none, random, avgpool, cls_attn(인코더측), fastv, pdrop, sparsevlm_lite (LLM 내부) | 모두 **재구현**(공식 코드 아님). DivPrune, DART, VisionZip, VScan, downsample은 미구현 |
| 유지 토큰 | 192 / 128 / 64 / 32 (576 기준) | NeXT는 같은 비율로 960/640/320/160 |
| Attention 백엔드 | `eager`, `sdpa`, `flash`(torch FlashAttention **강제**, 불가 시 `unsupported` 기록), `fa2`(flash-attn 패키지, 선택) | 조용한 fallback 금지 |
| 배치 크기 | 1, 4, 16, 32 | 한계 초과는 `oom`으로 기록(그 자체가 capacity 결과) |
| 출력 길이 L | 1(prefill 위주), 32, 128, 512 | EOS 무시 고정 길이. 512 토큰 한 번 생성에서 L=1/32/128/512를 모두 유도 |
| GPU | RTX 5090 32GB | **"GPU 2종"은 미충족**: 두 번째 GPU(예: Vast.ai 3090/4090/A100)는 선택 사항, §6 |

### 1.1 Fractional design (`configs/h1.yaml`)

전체 factorial은 3 백엔드 × 4 배치 × 25(= 6 방법 × 4 budget + none) = **300 config**. 아래 설계로 **180 config**로 줄인다.
**배치 × 출력 길이 × 백엔드 교차는 완전 포함**한다(출력 길이는 한 번의 512-토큰 실행에서 파생되므로 추가 비용이 없다).

| Tier | 방법 | budget | 백엔드 | 배치 | config 수 |
| --- | --- | --- | --- | --- | --- |
| A (핵심) | 전부(6 + none) | {192, 64} | eager, sdpa, flash | 1, 4, 16, 32 | 156 |
| B (budget 축) | 전부 | {128, 32} | sdpa | 1, 16 | 24 |

### 1.2 측정 pass와 총 실행 횟수

| pass | max_new | warm-up | 측정 반복 | 목적 |
| --- | --- | --- | --- | --- |
| prefill | 1 | 10 | 50 | prefill/vision 시간의 촘촘한 CI |
| decode | 512 | 3 | 10 | decode ms/token, L ≤ 512의 e2e |

- config당 측정 60회 → **총 측정 10,800회**(+ warm-up 2,340회). config당 진단 1회(커널 trace, 오버헤드 분해, L별 peak memory 4회).
- **예상 GPU 시간 ≈ 11시간** (`python scripts/plan_h1.py`; 5090 실측 전 가정값이므로 **추정치**, 파일럿 후 `--step-ms`, `--tflops`로 재보정).
  Vast.ai 시간당 단가 × 1.5배 여유를 예산으로 잡을 것.
- 용량 추정(추정치): LLaVA-7B fp16 KV ≈ 0.52 MB/토큰. 배치 32, 시각 576 + 출력 512 → KV ≈ 19 GB + 가중치 14 GB > 32 GB이므로
  **vanilla는 bs=32에서 OOM**, 압축 방법은 적재 가능할 것으로 예상(capacity 효과, 실측으로 확인).

## 2. 측정 지표

| 구분 | 지표 | 구현 |
| --- | --- | --- |
| 이론 | 토큰 수 감소율, 해석적 FLOPs(attention + FFN + 점수 계산 pass, 공식은 `efficiency/flops.py`) | `profiler.theoretical` |
| 이론 | 각 논문이 보고한 FLOPs | **수동 기입 필요**: `docs/h1_experiment.md` 표에 논문별 값을 추가 |
| 실측 | vision / prefill(점수 계산·선택 포함) / decode ms·token / e2e latency | CUDA event (`Clock`) |
| 실측 | throughput(samples/s, tokens/s), peak memory(L별), KV cache 크기 | `run_once`, `diagnostics` |
| 오버헤드 | 점수 계산(`ovh_score_ms`), top-k 선택(`ovh_select_ms`), 재배열(`ovh_gather_ms`) | `Engine._t` (동기화 pass, 측정 pass와 분리) |
| 검증 | 실제 실행된 attention 커널 family | `kernels.trace_kernels` (torch.profiler) |
| 환경 | torch / CUDA / transformers / flash-attn / git commit, SM 클럭·온도·전력 | `env.collect_env`, `gpu_state` |
| 정확도 | POPE / MME / TextVQA, ±1%p operating point | **미구현**, §6 |

## 3. 측정 프로토콜(재현성)

- 같은 코드베이스·라이브러리 버전(보고서 상단에 자동 기록), fp16, **양자화 금지**.
- CUDA event + `torch.cuda.synchronize()`로 시간 측정, warm-up 후 반복, 중앙값과 95% CI는 분석 단계에서 계산.
- 입력은 모든 방법에 같은 시드의 픽셀(지연은 shape에 의존하고 내용에는 의존하지 않음) → **실제 이미지 ~20장으로 sanity check를 별도 수행**할 것.
- **GPU 클럭 고정은 컨테이너에서 보통 불가** → 대신 config마다 SM 클럭/온도/전력을 기록(보고서 §8 throttling audit)하고,
  `shuffle_seed`로 config 순서를 무작위화해 방법이 열 상태와 교락되지 않게 한다.
- 커널 검증: config마다 짧은 trace로 실제 attention 커널 family(`fa2` / `sdpa_flash` / `sdpa_efficient` / `sdpa_math` / `eager`)를 분류하고,
  요청한 백엔드와 불일치하면 `kernel_ok=False`로 표시한다(보고서 §6). 이것이 "SparseVLM은 FlashAttention과 호환된다"는 주장을 직접 검증하는 근거다.
  단, 비전 타워는 HF 구현(SDPA)을 그대로 쓰므로 `attn_impl`은 **LLM 부분에만** 적용된다.
- 전용 프로세스: 다른 프로세스를 끄고(`nvidia-smi`로 확인), 실행 중 재개 가능(`records.jsonl`의 key 기준).

## 4. 분석 계획 (→ `scripts/analyze_h1.py`, `report.md`)

| # | 분석 | 구현 |
| --- | --- | --- |
| 1 | setting별 FLOPs 순위 vs 실측 순위의 Kendall τ(+ bootstrap CI), 신뢰할 수 있는 역전 쌍의 수·비율 | `analysis.settings`, `summarize` |
| 2 | Realization Ratio = 실측 speedup / FLOPs speedup (방법·조건별) | `report.md` §2 |
| 3 | 배치·출력 길이에 따른 τ, RR 변화 (히트맵/곡선) | `tau_heatmap.png`, §1 표 |
| 4 | Break-even: 압축 방법이 vanilla보다 느리거나 메모리가 더 큰 조건 (AnchorPrune의 SparseVLM decode 사례 재현·일반화) | `report.md` §4 |
| 5 | 백엔드 효과: 같은 방법을 eager/sdpa/flash로 실행, attention 점수 의존 방법의 불이익 분리 (`score` 오버헤드) | §2, §7 |
| 6 | 조건 간 τ 차이의 순열 검정 (batch 1 vs 32, L 1 vs 512, 백엔드 간) | `analysis.perm_test_tau`, `report.md` §5 |

주의: 반복 측정이 같은 방법을 공유하므로 per-repeat τ는 pseudo-replication이다. p값은 선별용으로 보고하고 CI와 함께 제시한다.

## 5. 결과 템플릿

### 표

| 표 | 내용 | 출처 |
| --- | --- | --- |
| T1 | 요인/수준, config·실행 횟수, GPU 시간 | §1 |
| T2 | 방법별 FLOPs(자체 계산 vs 논문 보고), 토큰 수 | `--flops-only` + 수동 입력 |
| T3 | (backend, batch, phase)별 mean τ, 역전 비율, support/refute 비율 | `report.md` §1 (`summary`) |
| T4 | 방법 × budget: FLOPs speedup, 실측 speedup, RR (prefill / decode / e2e L=128) | `report.md` §2 |
| T5 | 가장 빈번한 신뢰 역전 쌍, vanilla보다 느린 방법 | `inversions.csv`, §3-4 |
| T6 | 커널 검증(요청 vs 실행 family) | `report.md` §6 |
| T7 | 오버헤드 분해(score / select / gather) | `report.md` §7 |
| T8 | 순열 검정(조건 간 τ 차이) | `permutation_tests.csv` |
| T9 | 환경·throttling 감사 | `report.md` 상단, §8 |

### 그림

| 그림 | 내용 | 상태 |
| --- | --- | --- |
| F1 | FLOPs vs latency 산점도(방법별 점, vanilla 기준선) | 자동 (`flops_vs_latency.png`) |
| F2 | τ 히트맵(배치 × 출력 길이, 백엔드별) | 자동 (`tau_heatmap.png`) |
| F3 | Realization Ratio vs 배치 / vs L 곡선 | `methods.csv`로 작성 |
| F4 | break-even 지도(방법 × 조건, vanilla 대비 느림 여부) | `inversions.csv`로 작성 |
| F5 | 백엔드별 prefill 누적 막대(forward / score / select / gather) | `settings.csv`, 레코드 |
| F6 | Capacity: 배치별 peak memory와 OOM 경계 | `methods.csv` (phase=memory) |

## 6. 구현 상태와 타당성 위협

| 항목 | 상태 | 대응 |
| --- | --- | --- |
| LLaVA-1.5-7B 엔진 | 구현, HF 출력과 토큰 단위 일치 테스트 통과 | |
| LLaVA-NeXT, Qwen2/2.5-VL | **미구현** (AnyRes 특징은 같은 LLaMA 루프에 연결 가능, Qwen은 M-RoPE·GQA 등 별도 엔진 필요) | 이후 tier. 모델 간 일반화 주장은 그 전까지 보류 |
| DivPrune, DART, VisionZip, VScan, image downsample | **미구현** | 이후 추가, 현재 6개 방법으로는 "계열이 고르게 섞였다"고 주장하지 말 것 |
| 방법 구현 품질 | fastv / pdrop / sparsevlm_lite는 **최소 재구현**이며 공식 코드가 아님 | 공식 구현과 소규모 subset에서 대조 검증, 논문에는 재구현임을 명시 |
| 정확도(POPE/MME/TextVQA, ±1%p) | **미구현**: 효율 하네스는 정확도를 측정하지 않음 | 별도 단계로 수행하고 ±1%p 이내 operating point끼리만 효율 비교 |
| 하드웨어 일반성 | **GPU 1종(5090)** | 선택적으로 Vast.ai의 두 번째 GPU(3090/4090/A100)에서 tier A 일부 반복 |
| `fa2` (flash-attn) | Blackwell(sm_120) 지원 불확실 | `--preflight` 후 가능할 때만 `attn_impls`에 추가 |
| 비독립 근거 | Wen et al.과 DART는 같은 그룹 | 선행 연구로 **하나의 근거**로 취급 |
| 랜덤 픽셀 | 내용 무관 가정 | 실제 이미지 ~20장 sanity check |

## 7. 주장-근거 대응표

| # | 논문에 쓸 Claim | 필요한 Evidence | 산출물 | 반증 시 |
| --- | --- | --- | --- | --- |
| C1 | FLOPs 순위는 실측 순위를 신뢰할 수 있게 예측하지 못한다 | setting 대부분에서 τ CI 상한 < 0.6, 신뢰 역전 쌍 | T3, T5, F2 | 모든 setting에서 τ ≥ 0.9이면 기각, "FLOPs는 충분한 proxy"로 서술 |
| C2 | 이론 속도 향상의 일부만 실측으로 실현된다 (Realization Ratio < 1) | 방법·조건별 RR 분포 | T4, F3 | RR ≈ 1이면 C2 기각 |
| C3 | 불일치의 크기는 배치 크기·출력 길이에 따라 체계적으로 변한다 | τ/RR의 batch·L 곡선, 순열 검정 | T8, F2, F3 | p ≥ 0.05이면 "조건 의존성 증거 없음" |
| C4 | 일부 방법은 특정 조건에서 vanilla보다 느리거나 메모리를 더 쓴다 (break-even) | vanilla 대비 느림 조건 목록, SparseVLM decode 재현 | T5, F4 | 모든 조건에서 vanilla보다 빠르면 기각 |
| C5 | attention 점수 의존 방법은 fused 커널에서 추가 pass 비용을 낸다 | eager vs sdpa vs flash 비교, `score` 오버헤드 | T6, T7, F5 | 오버헤드 무시 가능하면 기각 |
| C6 | "FlashAttention 호환" 주장은 실행 커널 기준으로 검증되어야 한다 | 요청 vs 실제 커널 family, `unsupported` 기록 | T6 | 모두 일치하면 해당 주장은 지지됨(재구현 한계 명시) |
| C7 | 압축은 vanilla가 OOM 나는 배치에서 실행을 가능하게 한다 | OOM 경계, L별 peak memory | F6 | 경계 차이 없으면 기각 |

## 8. 단계별 일정

| 단계 | 내용 | 산출물 | 비용(추정) |
| --- | --- | --- | --- |
| S0 (로컬) | 코드/테스트 점검, `--flops-only`, `plan_h1.py` | 설정·계획 확정 | 0 |
| S1 | Vast.ai 5090 인스턴스, `setup_remote.sh`, `--preflight` | 환경 보고 | ~0.5 h |
| S2 | **파일럿** `configs/h1_pilot.yaml` (36 config) | 커널 검증, OOM 경계, 시간 보정 | ~0.3 h |
| S3 | `plan_h1.py` 재보정, config 조정 | 최종 예산 | 0 |
| S4 | **본 실험** `configs/h1.yaml` (180 config, 재개 가능) | `records.jsonl` | ~11 h |
| S5 | `analyze_h1.py` → 보고서, 판정 | `report.md`, CSV, 그림 | 0 |
| S6 | 실제 이미지 sanity check, 공식 구현 대조 | 타당성 부록 | ~1 h |
| S7 (선택) | 두 번째 GPU 반복, LLaVA-NeXT/Qwen, 추가 방법, 정확도 | 일반화 결과 | 별도 |

## 9. 실행 명령

```bash
python scripts/profile_efficiency.py --preflight
python scripts/plan_h1.py --config configs/h1.yaml
python scripts/profile_efficiency.py --config configs/h1_pilot.yaml
python scripts/analyze_h1.py --records outputs/h1_pilot/records.jsonl --out outputs/h1_pilot/report
python scripts/profile_efficiency.py --config configs/h1.yaml        # 중단되면 같은 명령으로 재개
python scripts/analyze_h1.py --records outputs/h1/records.jsonl --out outputs/h1/report
```

측정 코드 골격(타이밍 하네스, 커널 로깅)은 새로 쓰지 않고 실제 구현을 그대로 쓴다:
`src/vlmeval/efficiency/profiler.py`(`Clock`, `run_once`, `diagnostics`, `profile_grid`),
`engine.py`(백엔드·영역 타이머), `kernels.py`(커널 trace), `env.py`(환경·GPU 상태).
