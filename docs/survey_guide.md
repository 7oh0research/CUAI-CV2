# Phase 1: 평가 관행 조사 가이드

`literature_survey.csv`에 논문당 한 행. 목표는 15~20편. 이 표 자체가 논문의 한 섹션이 되고, "의심스러운 관행"이 H1~H5의 근거가 된다.

| 컬럼 | 기록 방법 |
|---|---|
| `efficiency_metric` | 토큰 수 / FLOPs / 지연시간 / 메모리 중 보고한 것 전부 |
| `reports_measured_latency` | 실측 시간을 보고했는가 (yes/no). FLOPs만 있으면 no |
| `gpu`, `batch_size` | 논문 본문/부록에 명시된 값, 없으면 `unspecified` |
| `aggregation` | 평균 정확도 / 평균 유지율 / 벤치마크별만 등 |
| `flash_attention_compatible` | attention 점수가 필요한 방법이면 no (명시 없으면 `unclear`) |
| `uses_blind_baseline` | 이미지 없는 기준선을 보고했는가 |
| `reports_ci_or_seeds` | 신뢰구간/다중 시드 보고 여부 |

판단이 애매하면 `unclear`로 두고 `notes`에 근거(섹션/표 번호)를 남긴다. 논문을 읽지 않고 기억으로 채우지 말 것.

집계 예: "조사한 N편 중 M편만 실측 지연시간을 보고했다" 형태의 문장을 만들 수 있어야 한다.
