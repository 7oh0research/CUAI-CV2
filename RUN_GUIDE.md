# 실행 가이드 (Linux / Ubuntu / Debian)

## 0. 사전 조건

- NVIDIA GPU 및 정상 동작하는 NVIDIA Driver (`nvidia-smi`가 동작해야 함)
- Ubuntu/Debian, RHEL 계열, Arch Linux (패키지 관리자 자동 감지)
- MME Benchmark 데이터셋을 내려받아 압축 해제한 상태
- 인터넷 연결 및 충분한 디스크 공간 (Conda 환경과 모델 가중치가 여러 개 생성/캐시됨)

이 저장소는 GPU Driver를 설치하거나 클라우드 GPU를 할당하지 않습니다. `nvidia-smi`가 동작하지 않으면 먼저 머신 제공자의 NVIDIA Driver 설치/활성화 절차를 완료하세요.

## 1. 최초 설치 (아무 개발 환경도 없는 Linux)

저장소 압축을 이미 해제했다면 다시 압축을 풀 필요가 없습니다. Vast.ai 인스턴스에서 저장소 폴더로 이동한 뒤 아래 한 줄을 실행합니다.

```bash
cd /path/to/h1_bench
bash setup.sh
```

`setup.sh`가 `.env`를 자동 생성한 뒤 전체 설치를 진행합니다. MME 데이터셋 경로는 설치 후 `.env`의 `MME_ROOT`에 입력합니다. 예를 들어 데이터셋을 저장소 안의 `data/`에 압축 해제했다면 다음처럼 지정합니다.

```bash
MME_ROOT=/workspace/h1_bench/data/MME_Benchmark_release_version
```

이 스크립트는 다음을 순서대로 수행합니다.

1. Ubuntu/Debian 시스템 라이브러리 및 빌드 도구 설치 (`sudo` 필요)
2. Miniforge 설치 (`$HOME/miniforge3`, root 권한 불필요)
3. NVIDIA Driver/GPU 및 MME 경로 사전 검사
4. 비교 알고리즘 공식 저장소 clone 및 commit 기록
5. Python 3.10 + PyTorch 2.7.1 + CUDA 12.8 (RTX 5090 / sm_120 compatible) 기반의 알고리즘별 독립 Conda 환경 생성
6. LLaVA 및 benchmark 의존성 설치
7. CUDA compiler가 호환되는 경우 FlashAttention 2 설치 시도
8. CUDA 및 Python 환경 검사

설치 도중 SSH 세션이 끊기거나 실패하면 오류를 확인한 뒤 같은 명령을 다시 실행할 수 있습니다. 이미 존재하는 Conda 환경과 clone 디렉터리는 가능한 한 재사용합니다.

## 2. 환경 검사

```bash
bash step1_env_check.sh
```

확인 파일:
- `env/env.json`: 대표 환경의 GPU, CUDA, PyTorch 정보
- `env/h1_*.json`: 알고리즘별 패키지 및 FlashAttention 설치 여부
- `env/env_flags.json`: FlashAttention 설치 환경 개수
- `env/methods.json`: clone한 공식 저장소의 commit 정보

## 3. Smoke Test

```bash
bash step2_smoke_test.sh
```

확인 파일: `results/smoke/*.jsonl`, `results/smoke/smoke_report.txt`

`MISMATCH`가 나온 방법은 해당 알고리즘의 공식 구현과 `h1bench/adapters.py`, `configs/methods.yaml` 설정을 대조해야 합니다. Smoke Test는 실제 알고리즘의 토큰 감소 동작을 확인하기 위한 단계이며, 단순히 패키지가 import된다고 측정이 유효한 것은 아닙니다.

## 4. 통합 실행 (권장)

처음부터 설치까지 한 번에 진행하려면 아래 명령을 실행합니다.

```bash
bash start.sh
```

`start.sh`는 Conda/LLaVA 환경이 없으면 `setup.sh`를 먼저 실행하고, `.env`에 MME 경로가 비어 있으면 데이터셋 경로를 한 번 입력받은 뒤 전체 파이프라인을 시작합니다. 이미 설치와 `.env` 설정이 끝났다면 아래 명령으로 실험만 실행할 수 있습니다.

```bash
bash run_all.sh
```

`run_all.sh`는 `run_pipeline.sh`를 호출하며 다음 순서로 진행합니다.

1. GPU/CUDA/PyTorch/Conda 및 데이터셋 사전 검사
2. 각 알고리즘의 Smoke Test 및 최종 visual-token 수 검증
3. 현재 Vast.ai GPU에서 측정한 Smoke Test 실행 시간을 기반으로 전체 소요 시간 추정
4. R=32, batch size=1 기준 실험 및 MME 예측 답변 생성
5. batch-size sweep (4, 8, 16)
6. R sweep (64, 128, 192)
7. CSV 및 그래프 취합, 최종 JSON 보고서 생성
8. 공식 MME scorer가 제공되면 점수 계산 시도

전체 실험 시작 전 예상 소요 시간과 GPU 대여 비용이 발생할 수 있다는 안내가 나오며, `y`를 입력해야 진행합니다. 확인 없이 실행하려면 `bash start.sh --yes` 또는 `bash run_all.sh --yes`를 사용합니다.

선택 옵션:

```bash
bash run_pipeline.sh --dry-run       # 실행 단계만 확인
bash run_pipeline.sh --skip-smoke    # Smoke Test 생략 (권장하지 않음)
bash run_pipeline.sh --yes           # 전체 실험 확인 프롬프트 생략
```

Smoke Test에서 토큰 수 불일치 또는 알고리즘 오류가 발견되면 전체 실험을 중단합니다. 원인을 수정한 뒤 다시 실행하세요.

## 5. 결과 및 재실행

- `results/raw/`: 조합별 원시 측정 결과
- `results/mme/<method>/`: 방법별 MME 예측 답변 및 `.complete` 완료 마커
- `results/summary.csv`: 실험 요약
- `results/ratios.csv`: baseline 대비 비율
- `results/layer_tokens.csv`: 층별 visual-token 수
- `results/attn_impl_actual.csv`: 실제 Attention 구현
- `results/figs/`: 그래프
- `results/final_report.json`: 최종 결과 인덱스 및 상태
- `logs/pipeline.log`: 통합 실행 로그
- `logs/runtime_runs.tsv`: 개별 실험 프로세스 실행 시간

기존 결과 파일이 있으면 해당 조합은 건너뛰는 Resume 동작을 합니다. 중단 후 같은 명령을 다시 실행할 수 있습니다. 완료 마커가 없는 MME 결과는 다시 생성합니다.

공식 MME 점수 계산은 별도 배포된 `calculation.py`가 필요합니다. 기본 경로 `$MME_ROOT/../eval_tool/calculation.py`에서 자동 탐색하며, 다른 경로라면 `.env`에 `MME_EVAL_TOOL=/absolute/path/to/calculation.py`를 지정합니다. 평가 도구가 없으면 예측 파일까지만 생성하고 공식 점수 계산은 건너뜁니다.

## 6. FlashAttention 관련

FlashAttention은 필수가 아닙니다. `nvcc`가 없거나 CUDA/PyTorch 조합이 맞지 않으면 설치를 건너뛰며 `eager`, `sdpa` 측정은 계속할 수 있습니다. FlashAttention을 별도로 시도하려면 호환 CUDA Toolkit을 준비한 뒤 아래처럼 실행합니다.

```bash
bash setup/install_flash_attn.sh h1_llava
```

설치 여부는 `env/h1_llava.json` 등을 확인하세요. 각 환경별 설치 상태가 다를 수 있습니다.

## 7. 선택 사항: AnchorPrune

AnchorPrune 저장소는 현재 자동 clone 대상이 아닙니다. 별도 checkout 경로를 `.env`의 `ANCHORPRUNE_PATH`에 지정한 뒤 아래를 다시 실행하세요.

```bash
bash setup/clone_methods.sh
bash setup/make_envs.sh
bash step1_env_check.sh
```

## 8. 유의 사항

- `configs/methods.yaml`의 `n_samples`, `warmup`, `new_tokens`, batch size 및 R sweep은 실험 설정입니다. GPU 메모리가 부족하면 batch size를 낮추거나 설정을 조정하세요.
- 일반적인 데스크톱 RTX 4090은 24GB VRAM입니다. README의 과거 실험 장비 표기와 실제 장비 사양이 다를 수 있으므로 `env/env.json`의 실측 VRAM을 기준으로 판단하세요.
- 모델 가중치(`liuhaotian/llava-v1.5-7b`)는 첫 실행 시 Hugging Face에서 다운로드되며, 저장 공간과 네트워크가 필요합니다.
- MME 예측 답변은 `step3_main_R32_bs1.sh`에서 생성합니다. 공식 점수 계산은 별도 scorer가 탐지되면 통합 Pipeline이 실행하고, 없으면 예측 파일 생성까지만 진행합니다.
- FlashAttention 미설치 환경의 `flash_attention_2` 조건은 `unsupported_env`로 기록됩니다.

## H3-only fast workflow

For the revised research scope, use:

```bash
bash start.sh --profile h3
```

This runs only H3 (Vanilla blind inference, then visual inference for Vanilla/FastV/SparseVLM/VisionZip, followed by Score A/Score B aggregation). H1 is not rerun. Configure `MME_ROOT`, `POPE_ROOT`, and `POPE_FILE` in `.env`. Full details: `H3_GUIDE.md`.
