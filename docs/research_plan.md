# 연구 계획: VLM 시각 토큰 효율화 평가 지표 진단 및 제안

백본: **LLaVA-1.5-7B** (`llava-hf/llava-1.5-7b-hf`, fp16, 양자화 없음). 학습 없음, 공개 체크포인트 + training-free 방법만 사용.

## 연구 질문

VLM 시각 토큰 효율화 연구에서 현재 쓰이는 평가 방식은 방법 간 실제 우열을 올바르게 드러내는가?

기여: (1) 문제 발견 (2) 원인 분석 (3) 새 지표 + 메타 평가.

## 범위

대상은 "효율화 방법의 평가 지표" (토큰 수, FLOPs, 정확도 유지율, 평균 점수). 벤치마크 채점 방식 자체는 범위 밖.

## 단계와 코드 대응

| Phase | 내용 | 위치 |
|---|---|---|
| 1 | 효율화 논문 15~20편 평가 관행 조사 | `docs/literature_survey.csv`, `docs/survey_guide.md` |
| 2 | 측정 환경 (샘플 단위 로그) | `src/vlmeval/{models,methods,data}`, `scripts/run_experiment.py`, `configs/` |
| 3 | 가설 H1~H5 검정 | `src/vlmeval/analysis/hypotheses.py` |
| 4 | 새 지표 설계 (사전 고정) | `docs/metric_spec.md`, `src/vlmeval/analysis/metrics.py` |
| 5 | 메타 평가 | `src/vlmeval/analysis/stats.py`, `scripts/analyze.py` |

## 가설

| ID | 문제 | 확인 방법 | 코드 |
|---|---|---|---|
| H1 | 이론 효율(토큰/FLOPs)과 실측 지연시간의 괴리 | 이론 감소율 vs 실측 감소율, 순위 비교 | `h1_*` |
| H2 | 평균 유지율이 특정 과제의 붕괴를 가림 | 벤치마크별 유지율, macro vs worst 순위 | `h2_*` |
| H3 | 이미지 없이 풀리는 문항이 압축 효과를 부풀림 | blind 실행, 시각 의존 문항만의 유지율 | `h3_*` |
| H4 | 정확도는 같아도 답이 바뀜 | flip rate, CAR | `h4_*` |
| H5 | 한 압축률만 보면 결론이 달라짐 | 압축률별 순위 역전 | `h5_*` |

전초 단계에서는 2~3개만 깊게 본다. 순위 역전이 가장 뚜렷한 가설이 본 연구의 중심.

## 비교 방법 (모두 projector 이후, LLM 입력 직전에 적용)

| 이름 | 설명 |
|---|---|
| `none` | 576토큰 전부 (기준선) |
| `blind` | 이미지 없음 (H3용) |
| `random` | 무작위 토큰 선택 (단순 베이스라인) |
| `avgpool` | 패치 격자 평균 풀링 (단순 베이스라인) |
| `cls_attn` | 인코더 CLS attention top-k (VisionZip 계열 단순화) |

압축률(제거 비율): 0.5 / 0.75 / 0.9. 

**한계 (논문에 명시할 것):** 현재는 LLM 입력 전 압축만 구현. FastV처럼 LLM 중간 레이어에서 attention으로 가지치기하는 방법은 FlashAttention과 충돌해 H1에 중요한 사례이므로 본 연구 확장 항목으로 둔다 (`methods/`에 추가하고 `LlavaRunner`에 레이어 훅 필요).

## 샘플 단위 로그 (핵심 설계 원칙)

`outputs/<run>/records.jsonl`, 컬럼은 `src/vlmeval/schema.py`. 점수만 저장하지 않는다. 새 지표는 전부 이 데이터에서 계산되므로 지표를 바꿔도 재실험이 필요 없다.

## 통계 원칙

- 모든 수치에 bootstrap 95% CI (샘플 단위 재표집, 방법 간 pairing 유지).
- 방법 간 비교는 paired 검정 (McNemar exact, `stats.mcnemar_exact`).
- 소규모(벤치마크당 500) 결과는 잡음일 수 있다. 본 연구에서 전체 벤치마크로 재확인한다.
- 순위의 신뢰성: `rank_stability` (bootstrap 순위 일관성).

## 1주 일정

| 일차 | 작업 | 산출물 |
|---|---|---|
| 1~2 | 문헌 조사 + 환경 구축, 추론 재현 | `literature_survey.csv`, smoke test 통과 |
| 3~4 | `run_experiment.py` 전체 실행 | `records.jsonl` |
| 5~6 | `analyze.py`, 가설 2~3개 검정 | `report.md`, 순위 역전 사례 |
| 7 | 지표 초안 확정, 본 연구 계획 | 주제 선정 보고서 |

## 실행 순서

```bash
pip install -e ".[infer,dev]"
pytest                                                  # 분석 코드 검증 (GPU 불필요)
python scripts/analyze.py --synthetic --out outputs/synthetic_report   # 파이프라인 점검
python scripts/run_experiment.py --dry-run              # 데이터셋 컬럼 확인 (GPU 불필요)
python scripts/run_experiment.py --config configs/pilot.yaml
python scripts/analyze.py --records outputs/pilot/records.jsonl --out outputs/pilot/report
```

## 최초 GPU 실행 시 확인할 것

- `data/loaders.py`의 HF 데이터셋 id와 컬럼명 (lmms-lab 미러 기준, 미검증)
- `none` 방법의 baseline 정확도가 공개 수치와 대략 일치하는지 (POPE ~85, TextVQA ~58). 크게 다르면 프롬프트 템플릿 점검.
- 지연시간 측정은 batch size 1, 동일 GPU, 다른 프로세스 없음, 24GB 이상 권장.
- GQA 어댑터는 미구현 (TODO).

## H1 상세 실험 설계

H1(이론 효율 vs 실측 효율)의 구체 설계, 선행 연구(Wen et al., AnchorPrune, VisionDrop), 실행 방법은 [h1_experiment.md](h1_experiment.md) 참고. 코드: `src/vlmeval/efficiency/`.

