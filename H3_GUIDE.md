# H3-only 실험 가이드

## 연구 범위

- H1(이론 FLOPs와 실측 Latency의 괴리)은 선행 연구 수치와 문헌 분석으로 정리하고, 이 실행 프로필에서는 H1 전체 벤치마크를 반복하지 않습니다.
- H3(시각 의존도 착시)만 직접 평가합니다.
- 모델: LLaVA-1.5-7B
- 방법: Vanilla, FastV, SparseVLM, VisionZip
- 압축 방법은 64 visual tokens 설정을 사용합니다.
- Attention은 SDPA로 고정합니다.
- Blind 단계에서는 Vanilla에 이미지 없이 질문만 입력합니다.
- Visual 단계에서는 같은 문항에 실제 이미지를 입력합니다.
- Score A는 전체 문항 정확도, Score B는 Blind 단계에서 맞힌 문항을 제외한 정확도입니다.

## 실행

프로젝트 루트에서:

```bash
bash start.sh --profile h3
```

필요한 Conda 환경이 없으면 H3에 필요한 네 환경만 설치합니다. 첫 실행에서 MME 디렉터리, POPE annotation(JSONL/CSV/JSON), POPE 이미지 디렉터리 경로를 물어봅니다. 이후 값은 `.env`에 저장됩니다.

직접 설정하려면 `.env.example`을 참고해 `.env`에 아래 세 값을 설정합니다.

```bash
MME_ROOT=/path/to/MME_Benchmark_release_version
POPE_ROOT=/path/to/pope/images
POPE_FILE=/path/to/pope_test.jsonl
```

POPE annotation은 각 row에 `image`(또는 `image_path`/`image_id`/`filename`), `question`, `answer`(또는 `label`/`ground_truth`) 필드가 있어야 합니다. 이미지 파일은 POPE_ROOT 아래에서 찾을 수 있어야 합니다.

## 결과

- `results/h3/blind_baseline.jsonl`: 이미지 없는 Vanilla 예측
- `results/h3/*_visual.jsonl`: Vanilla 및 압축 방법별 이미지 입력 예측
- `results/h3/blind_predictions.csv`: Blind 분류 결과
- `results/h3/h3_scores.csv`: 데이터셋별 Score A / Score B 및 yes-class precision/recall/F1
- `results/h3/h3_report.json`: 실험 요약
- `logs/h3/`: 단계별 실행 로그

중단 후 같은 명령을 다시 실행하면 이미 기록된 문항 ID는 건너뛰고 이어서 진행합니다. 다만 완성된 방법별 JSONL이 존재하면 해당 방법 전체를 건너뜁니다. 일부 결과를 의도적으로 다시 만들려면 해당 결과 JSONL을 백업한 뒤 삭제하세요.

## 시간 관련 주의

2시간은 목표 예산이지 보장 시간이 아닙니다. 이 구현은 호환성을 우선해 이미지 추론을 기본 Batch Size 1로 실행합니다. 전체 MME와 POPE를 모두 사용하면 GPU, 디스크 캐시, 모델 로딩 및 실제 생성 속도에 따라 2시간을 넘길 수 있습니다. 2시간 제한이 엄격하면 전체 데이터 실행 전에 샘플 수를 제한한 예비 실행으로 처리량을 측정한 뒤, 표본 수와 대여 시간을 결정해야 합니다.

## 해석상 주의

Blind-Solvable은 이미지 없이 모델이 맞힌 문항을 뜻합니다. Score B는 이 집합을 제외한 문항의 정확도입니다. MME 공식 점수 산출 방식과 완전히 동일한 점수라고 주장하지 말고, H3 분석용 문항 수준 accuracy로 보고하세요. 공식 MME 점수가 필요하면 별도의 공식 scorer도 병행하세요.
