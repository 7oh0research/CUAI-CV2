# CUAI-CV2 RTX 5090 Fresh Server Bootstrap

## 중요: Vast.ai 디스크

현재 저장소의 H3 설치기는 H3 방법별로 `h1_llava`, `h1_fastv`, `h1_sparsevlm`, `h1_visionzip` 네 개의 독립 Conda 환경을 만듭니다. 이는 FastV의 Transformers fork 등 방법별 의존성 충돌을 피하기 위한 구조입니다. 따라서 단일 환경으로 임의 통합하지 않았습니다.

RTX 5090 모델 파일(약 14GB), 네 환경, 데이터셋, Git working tree를 고려하면 30GB는 안전한 용량이 아닙니다. 설치기는 저장 공간이 40GiB 미만이면 프로젝트 설치를 시작하기 전에 중단합니다. 따라서 30GB 인스턴스에서는 설치가 진행되지 않습니다. 권장 디스크는 60GB(전체 데이터셋까지 저장하면 70GB)입니다. 30GB 인스턴스를 이미 만들었다면 실행 전에 Vast.ai 볼륨을 늘리세요.

## Vast.ai 초기 인스턴스 전제

- Linux x86_64
- RTX 5090이 `nvidia-smi`에 표시됨
- NVIDIA 드라이버가 컨테이너에 전달됨
- Ubuntu/Debian 계열 권장 (저장소 OS 패키지 설치기는 apt/dnf/yum/pacman 지원)
- 작업 경로 기본값: `/workspace/CUAI-CV2`

컨테이너 안에서 NVIDIA 드라이버나 전체 CUDA Toolkit을 설치하지 않습니다. PyTorch CUDA 12.8 wheel을 사용합니다.

## 설치

ZIP을 서버에 업로드하고 압축을 푼 뒤:

```bash
cd cuai_cv2_5090_bootstrap
bash install.sh
```

다른 경로를 사용하려면:

```bash
CUAI_ROOT=/workspace/CUAI-CV2 CONDA_PREFIX_ROOT=/root/miniforge3 bash install.sh
```

설치기는 저장소 `main`을 shallow clone하고, 저장소의 오래된 PyTorch 버전 검증 문자열을 PyTorch 2.9.1 / Torchvision 0.24.1로 갱신한 뒤 H3 설치 절차를 실행합니다. 설치기는 FlashAttention을 설치하지 않습니다(SDPA 경로).

## 검증

```bash
bash verify.sh
```

성공 기준: 네 환경 모두 Python 3.10, PyTorch 2.9.1, Torchvision 0.24.1, NumPy 1.26.4, CUDA 12.8, `sm_120`, 실제 CUDA matrix multiply.

## 데이터셋 설정과 실행

저장소 H3 가이드는 MME root, POPE 이미지 root, POPE annotation file 경로를 요구합니다. 데이터 파일을 먼저 서버에 업로드/다운로드하고 경로를 확인한 뒤:

```bash
cd /workspace/CUAI-CV2
cp .env.example .env
nano .env
```

`.env`에 실제 경로를 기록합니다:

```bash
MME_ROOT=/absolute/path/to/MME_Benchmark_release_version
POPE_ROOT=/absolute/path/to/pope/images
POPE_FILE=/absolute/path/to/pope_test.jsonl
```

실험 실행:

```bash
bash run_h3.sh
```

결과는 `results/h3/`, 로그는 `logs/h3/`에 기록됩니다. 실행은 저장소의 H3 구현을 따릅니다. H3 가이드에 따르면 기본 이미지 추론은 Batch Size 1이며, 전체 데이터 실행은 2시간을 넘을 수 있습니다.

## 주의

- `bash setup_h3.sh`를 별도로 다시 실행할 필요는 없습니다. `install.sh`가 올바른 버전으로 패치한 후 호출합니다.
- `h1_llava`와 `h1_llava_py310`을 혼용하지 않습니다. 새 설치는 저장소가 기대하는 네 환경 이름을 그대로 사용합니다.
- 30GB에서는 설치 후 모델/데이터 다운로드 단계에서 공간이 부족할 수 있습니다. 디스크 여유 공간을 수시로 확인하세요.
