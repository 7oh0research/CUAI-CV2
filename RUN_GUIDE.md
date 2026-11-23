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
5. Python 3.10 + PyTorch 2.1.2 + CUDA 12.1 기반의 알고리즘별 독립 Conda 환경 생성
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

## 4. 전체 측정

```bash
bash step3_main_R32_bs1.sh
bash step4_batch_sweep.sh
bash step5_R_sweep.sh
bash step6_aggregate.sh
```

또는 처음부터 순서대로 실행하려면:

```bash
bash run_all.sh
```

기존 결과 파일이 있으면 해당 조합은 건너뛰는 resume 동작을 합니다. 결과를 처음부터 다시 측정하려면 `results/raw/`의 해당 결과 파일을 직접 백업/삭제하세요.

| 단계 | 주요 결과 |
|---|---|
| 1 | `env/env.json`, `env/h1_*.json` |
| 2 | `results/smoke/*.jsonl`, `results/smoke/smoke_report.txt` |
| 3~5 | `results/raw/*.jsonl`, `logs/*.log` |
| 6 | `results/summary.csv`, `results/ratios.csv`, `results/figs/` |

## 5. FlashAttention 관련

FlashAttention은 필수가 아닙니다. `nvcc`가 없거나 CUDA/PyTorch 조합이 맞지 않으면 설치를 건너뛰며 `eager`, `sdpa` 측정은 계속할 수 있습니다. FlashAttention을 별도로 시도하려면 호환 CUDA Toolkit을 준비한 뒤 아래처럼 실행합니다.

```bash
bash setup/install_flash_attn.sh h1_llava
```

설치 여부는 `env/h1_llava.json` 등을 확인하세요. 각 환경별 설치 상태가 다를 수 있습니다.

## 6. 선택 사항: AnchorPrune

AnchorPrune 저장소는 현재 자동 clone 대상이 아닙니다. 별도 checkout 경로를 `.env`의 `ANCHORPRUNE_PATH`에 지정한 뒤 아래를 다시 실행하세요.

```bash
bash setup/clone_methods.sh
bash setup/make_envs.sh
bash step1_env_check.sh
```

## 7. 유의 사항

- `configs/methods.yaml`의 `n_samples`, `warmup`, `new_tokens`, batch size 및 R sweep은 실험 설정입니다. GPU 메모리가 부족하면 batch size를 낮추거나 설정을 조정하세요.
- 일반적인 데스크톱 RTX 4090은 24GB VRAM입니다. README의 과거 실험 장비 표기와 실제 장비 사양이 다를 수 있으므로 `env/env.json`의 실측 VRAM을 기준으로 판단하세요.
- 모델 가중치(`liuhaotian/llava-v1.5-7b`)는 첫 실행 시 Hugging Face에서 다운로드되며, 저장 공간과 네트워크가 필요합니다.
- MME 점수 생성은 `step3_main_R32_bs1.sh` 실행 후 안내되는 공식 MME evaluation tool로 별도 수행합니다.
- FlashAttention 미설치 환경의 `flash_attention_2` 조건은 `unsupported_env`로 기록됩니다.
