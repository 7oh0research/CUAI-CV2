# CUAI-CV2 H3 experiment

H3 scope: **Vanilla, FastV, SparseVLM, VisionZip** on LLaVA-1.5-7B. Default visual token budget: 64.

## Install
Use an Ubuntu Vast.ai instance with RTX 5090 32GB and at least 60GB disk (70GB recommended with benchmark data). Then:
```bash
unzip CUAI-CV2_H3_Repository.zip
cd CUAI-CV2_H3
bash setup/install.sh
cp .env.example .env
nano .env
```
The install creates isolated environments `h3_vanilla`, `h3_fastv`, `h3_sparsevlm`, `h3_visionzip` because official upstreams pin conflicting Transformers versions.

## Data
POPE 공식 JSON/JSONL 파일은 다음 변환기를 사용할 수 있습니다.
```bash
python scripts/convert_pope.py --input /path/to/coco_pope_random.json --image-root /path/to/COCO/val2014 --split random --output /workspace/datasets/pope_random.jsonl
```
MME는 14개 subtask의 공식 질문/정답/이미지 ID를 보존하여 동일 manifest 형식으로 변환하세요. MME 공식 점수는 공식 `eval_tool/calculation.py` 결과 포맷이 필요하므로, 이 프로젝트의 일반 accuracy 요약을 공식 MME score로 표기하지 않습니다.

Create JSONL manifest, one row per benchmark question:
```json
{"question_id":"pope_random_000001","dataset":"pope","image":"/workspace/datasets/POPE/image.jpg","question":"Is there a dog in the image? Answer yes or no.","ground_truth":"yes","category":"random"}
```
Required keys: `question_id`, `question`, `ground_truth`; image path is required for image conditions. Preserve official MME/POPE IDs and answers. Configure `H3_MANIFEST`, `HF_MODEL_PATH`, `LLAVA_MODEL_PATH` in `.env`. HF path is for Vanilla/FastV; original LLaVA checkpoint path is for SparseVLM/VisionZip.

## Run
Smoke test first:
```bash
LIMIT=10 bash scripts/run_all.sh
```
Full run:
```bash
bash scripts/run_all.sh
```
Outputs: `results/raw/*.jsonl`, `results/processed/blind_solvable_ids.json`, `results/reports/h3_metrics.csv`.

Blind-solvable IDs are determined once using Vanilla Blind and excluded identically during visual-dependent analysis. POPE yes/no accuracy is normalized. For publication-grade MME scores, run the official MME evaluator on its required format; included generic accuracy is not an MME score.

## Method implementations
FastV: official FastV HF Transformers fork and `fastv_config` (K=2; pruning ratio 0.8889 to retain about 64/576 image tokens); SparseVLM: official v1.5 branch LLaVA implementation with `RETAIN_TOKN`; VisionZip: official `visionzip(model, dominant=54, contextual=10)` integration (64 total). The three upstreams use different model loaders and dependencies, so isolated environments are intentional.

## Validation caveat
Static syntax checks only are possible in the package-building environment. No network access or RTX 5090 is available here; install and end-to-end CUDA smoke tests must be performed on Vast.ai before using results.
