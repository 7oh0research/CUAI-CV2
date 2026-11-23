# h1_bench: H1 효율 실측 (LLaVA-1.5-7B)

이 저장소는 LLaVA-1.5-7B 기반으로 여러 visual-token pruning 방법의 추론 효율을 측정합니다.

## 처음 설치하는 Linux 머신

NVIDIA GPU와 정상 동작하는 NVIDIA Driver가 필요합니다. Ubuntu/Debian, RHEL 계열(dnf/yum), Arch Linux(pacman)에서 아래 명령을 실행할 수 있습니다.

```bash
unzip h1_bench.zip
cd h1_bench
cp .env.example .env
# .env를 열어 MME_ROOT를 실제 MME 데이터셋 경로로 수정
bash setup/setup_all.sh
```

설치 후 검증 및 실행 순서는 [RUN_GUIDE.md](RUN_GUIDE.md)를 참고하세요.

- Python: 3.10
- PyTorch: 2.1.2 / torchvision 0.16.2
- PyTorch CUDA runtime: 12.1
- Transformers: 4.37.2
- FlashAttention 2: CUDA compiler가 호환될 때 선택적으로 설치

각 알고리즘은 독립 Conda 환경을 사용합니다. 공식 알고리즘 저장소의 코드는 `third_party/`에 clone되며, 측정 결과와 로그는 `results/`, `logs/`, 환경 정보는 `env/`에 생성됩니다.

**주의:** 설치 스크립트는 NVIDIA Driver를 설치하지 않습니다. GPU Driver는 머신/클라우드 제공자의 드라이버 설치 절차로 먼저 준비해야 합니다. CUDA toolkit이 없더라도 PyTorch CUDA 실행은 가능할 수 있지만, FlashAttention 소스 빌드는 생략됩니다.
