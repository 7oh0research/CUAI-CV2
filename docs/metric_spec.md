# 지표 명세 (사전 고정)

결과를 보기 **전에** 정의를 고정한다. 결과를 본 뒤 지표를 끼워 맞추는 것을 막기 위한 문서이며, 변경 시 날짜와 사유를 아래 변경 이력에 남긴다.

## 좋은 지표의 조건

1. **타당성** - 측정하려는 것을 반영하는가
2. **신뢰성** - 샘플을 다시 뽑아도 순위가 안정적인가 (`rank_stability`)
3. **변별력** - 방법 간 차이를 구분하는가 (bootstrap CI가 겹치지 않는 쌍의 수)
4. **단순성** - 샘플 단위 로그만으로 계산 가능

## 정의

표기: 구성 c = (방법, 압축률), 샘플 i, 기준선(전체 토큰) 정답 `b_i`, 구성 c 정답 `y_i^c`, blind 정답 `z_i`.

| 지표 | 정의 | 겨냥 가설 | 구현 |
|---|---|---|---|
| `retention_pooled` (기존) | `mean(y^c) / mean(b)` | - | `metrics.retention_pooled` |
| `retention_macro` (기존) | 벤치마크별 retention의 평균 | - | `metrics.retention_macro` |
| `worst_retention` | 벤치마크별 retention의 최솟값 | H2 | `metrics.worst_retention` |
| `vd_retention` | `V = {i : b_i=1, z_i=0}` 위에서 `mean(y^c)` | H3 | `metrics.vd_retention` |
| `flip_rate` | 기준선과 예측 문자열이 다른 샘플 비율 | H4 | `metrics.flip_rate` |
| `car` (retained-correct rate) | `sum(b_i * y_i^c) / sum(b_i)` (기준선이 맞힌 문항 중 계속 맞히는 비율, 우연히 맞힌 것은 상쇄 안 됨) | H4 | `metrics.car` |
| `latency_reduction` | `1 - median(실측 ms) / median(기준선 ms)` | H1 | `metrics.latency_reduction` |
| `pareto_auc` | x=`latency_reduction`, y=`retention_pooled` 곡선의 [0,1] 구간 면적. (0,1)에서 시작, 방법의 최고 속도 이후는 0 | H1, H5 | `metrics.pareto_auc` |

비고: `vd_retention`은 기준선이 이미 맞힌 문항만 대상이므로 상한이 1이고, 기준선 정확도로 나누지 않는다.

## 메타 평가 기준 (새 지표가 필요하다는 증거)

| 항목 | 기준 | 구현 |
|---|---|---|
| 순위 역전 | 기존 지표와 새 지표의 순위가 다른 쌍을 보고하고, 어느 쪽이 "참"에 가까운지 외부 기준(세부 과제, 전체 벤치마크, 소규모 사람 평가)으로 판정 | `stats.reversed_pairs` |
| 상관 | Kendall tau가 너무 높으면 불필요, 너무 낮으면 타당성 의심 | `stats.kendall` |
| 안정성 | bootstrap 순위 일관성 (mean tau, top-1 안정성, 쌍별 일관성) | `stats.rank_stability` |

## 변경 이력

- 초안 (전초 단계 시작 시점)
