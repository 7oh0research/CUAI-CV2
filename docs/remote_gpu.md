# 원격 GPU 실행 가이드 (Vast.ai, RTX 5090 32GB, SSH)

## 인스턴스 선택

- GPU: RTX 5090 1장 (32GB). 7B fp16 가중치 약 14GB + 배치 16 / eager attention도 32GB에 들어간다. OOM이면 기록하고 건너뛰도록 되어 있다.
- 이미지: PyTorch 이미지 중 **CUDA 12.8 이상** 선택 (5090은 sm_120이라 cu128+ 빌드 필요). `setup_remote.sh`가 필요하면 cu128 torch를 설치하고 preflight로 검증한다.
- 디스크: 60GB 이상 (모델 다운로드 약 14GB + 캐시).
- 권장: 인터럽터블이 아닌 on-demand. 중간 회수되면 측정이 끊기지만 `records.jsonl`에서 재개된다.
- 호스트 선택 시 단독 GPU, 높은 신뢰도, 전력 제한이 낮지 않은 것을 고른다. 공유 호스트는 latency 노이즈가 커진다.

## 실행

```bash
ssh -p <PORT> root@<HOST>
git clone https://github.com/7oh0research/CUAI-CV2.git && cd CUAI-CV2
bash scripts/setup_remote.sh          # venv, torch(cu128), 설치, preflight, 테스트
tmux new -s h1
source .venv/bin/activate
python scripts/profile_efficiency.py --config configs/h1.yaml    # SSH가 끊겨도 tmux에서 계속, 중단 시 재실행하면 이어서
```

모델은 공개 체크포인트(`llava-hf/llava-1.5-7b-hf`)라 토큰이 필요 없다.

## 결과 회수

```bash
# 로컬에서
rsync -avz -e "ssh -p <PORT>" root@<HOST>:CUAI-CV2/outputs/h1/ outputs/h1/
python scripts/analyze_h1.py --records outputs/h1/records.jsonl --out outputs/h1/report
```

분석은 CPU만 쓰므로 로컬에서 하고, 인스턴스는 `records.jsonl`을 받은 뒤 바로 삭제해 비용을 줄인다. `outputs/`는 git-ignore이므로 커밋하지 말고 별도로 보관한다.

## 측정 신뢰도 체크리스트

- `records.jsonl`의 `device`에 GPU 이름이 기록된다. 한 번의 H1 실행은 같은 인스턴스에서 끝낸다 (GPU가 바뀌면 비교가 깨진다).
- 측정 중 다른 프로세스 금지. `nvidia-smi`로 시작 전 사용률 0%인지 확인.
- 클럭 변동이 의심되면 `nvidia-smi -q -d CLOCK,PERFORMANCE`로 throttling 확인 (컨테이너에서는 클럭 고정 권한이 없을 수 있음).
- 첫 실행에서 `--tiny` 대신 `configs/h1.yaml`의 `repeats`를 3으로 줄인 스모크 실행을 먼저 하고, 전체 grid는 그 뒤에 돌린다.
- 5090은 Blackwell이라 sdpa가 FlashAttention 계열 커널을 쓰는지는 torch 버전에 따라 다르다. 보고서에 torch/transformers/CUDA 버전과 GPU 이름을 함께 적는다.
