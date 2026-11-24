# H3 Blind Test (Vanilla LLaVA-1.5-7B)

가설 H3: "이미지 없이도 풀리는 문항이 압축 효과를 부풀린다."
이 저장소는 압축을 적용하기 전에 먼저 **vanilla 모델에서 blind 문항이 실제로 많이 존재하는지** 확인한다.
blind 문항이 거의 없으면 H3는 의미가 약하고, 많으면 H3 본 실험(압축 방법 비교)으로 넘어갈 근거가 된다.

## 입력 조건
| 조건 | 입력 | 목적 |
|---|---|---|
| `image` | 원본 이미지 + 질문 | 기준 정확도 |
| `blank` | 같은 크기의 회색 이미지 + 질문 | 시각 토큰 수는 같고 정보만 없음 |
| `noimage` | 질문만 (이미지 토큰 없음) | 순수 언어 사전지식 |

## 실행 (Vast.ai, RTX 5090)
```bash
bash setup.sh                 # 시스템 패키지, venv, PyTorch(cu128), 의존성 설치와 GPU 검증
source .venv/bin/activate
python scripts/inspect_datasets.py     # 데이터셋 필드 구조 확인 (처음 한 번)
bash run_all.sh quick         # 벤치마크당 200문항 빠른 점검 (약 10~20분)
bash run_all.sh full          # 전체 실행
```
`tmux new -s h3` 안에서 실행하면 접속이 끊겨도 계속 돈다. 다시 실행하면 완료된 문항은 건너뛴다(resume).

## 결과
- `results/<run>/raw/*.jsonl`: 문항별 원본 출력, 파싱된 답, 정답 여부
- `results/<run>/summary.csv`: 벤치마크별 핵심 지표
- `results/<run>/per_category.csv`: 하위 범주별 지표
- `results/<run>/blind_solvable_ids/<bench>.txt`: blind 문항 id 목록 (H3 본 실험에서 필터로 사용)
- `results/<run>/report.md`, `fig_accuracy.png`: 자동 보고서

## 핵심 지표
- `acc_image`, `acc_blank`, `acc_noimage`: 조건별 정확도
- `chance`: 문항별 무작위 정답 확률의 평균 (yes/no 0.5, 객관식 1/선택지 수, TextVQA 0)
- `majority`: 가장 흔한 정답만 고를 때의 정확도
- `LPR` (Language Prior Ratio) = (acc_noimage - chance) / (acc_image - chance)
  : 우연 수준을 넘는 정확도 중 이미지 없이도 얻어지는 비율
- `blind_solvable`: noimage와 blank **두 조건 모두에서** 맞힌 문항 비율 (우연히 맞힌 경우를 줄이기 위함)
- `p_value`: noimage 정확도가 우연 수준보다 높은지에 대한 단측 검정 (Poisson-binomial 정규 근사)
- `yes_rate_noimage`: yes/no 과제에서 이미지 없이 "yes"라고 답한 비율 (응답 편향 확인)

## H3 판정 규칙 (configs/default.yaml에 사전 고정)
벤치마크마다 `p_value < alpha` 이고 `LPR >= lpr_threshold` 이면 "blind 문항 유의미"로 판정한다.
하나 이상의 벤치마크에서 성립하면 H3 본 실험을 진행할 근거가 있다고 본다.
**판정 기준은 결과를 보기 전에 정하고 바꾸지 않는다.**

## 주의
- 숫자는 모두 이 저장소의 실행 로그에서 계산된다. 논문 보고값을 섞지 않는다.
- MMBench는 circular evaluation을 쓰지 않는 단순 정확도다(선택지 순서 편향이 남아 있을 수 있음).
- TextVQA는 LLaVA 원 논문과 달리 OCR 토큰 힌트를 넣지 않는다.
- POPE 등에서 이미지 없이 "yes"만 답해도 50%가 나오므로 chance, majority와 함께 해석한다.
