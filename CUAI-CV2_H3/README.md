# CUAI-CV2 | H3 Visual Token Compression Benchmark

H3-only reproducibility repository for comparing **Vanilla, FastV, SparseVLM, and VisionZip** on LLaVA-1.5-7B. The central analysis is whether language-only (blind) solvability inflates apparent performance of visual-token compression methods.

> This repository separates the H3 experiment protocol and result analysis from the upstream implementations. The upstream method repositories are cloned by `setup/clone_methods.sh`; their licenses and original instructions remain authoritative. Do not describe a method as reproduced until its adapter and parity checks have been validated.

## Experiment matrix

| Run | Image | Method | Purpose |
|---|---|---|---|
| blind | removed | Vanilla LLaVA | Identify blind-solvable questions |
| original | original | Vanilla LLaVA | Full-image reference |
| compressed | original | FastV | Token reduction baseline |
| compressed | original | SparseVLM | Token reduction baseline |
| compressed | original | VisionZip | Token reduction baseline |

Use the **same question IDs, prompt template, model checkpoint, decoding settings, and benchmark split** across runs. Freeze the blind-solvable set using Vanilla blind predictions before comparing compression methods. Never recompute this set per method.

## Repository layout

```text
configs/                 experiment, model, dataset settings
src/h3bench/             JSONL contracts, blind filtering, metrics, report helpers
scripts/                 CLI entry points
setup/                   host checks, environment setup, upstream source checkout
third_party/             cloned upstream method repositories (not committed)
results/raw/              immutable per-question predictions
results/processed/        joined analysis tables
results/reports/          figures and summaries
logs/                     run logs
```

## Hardware and software

- Recommended GPU: NVIDIA RTX 5090 32GB (keep this GPU; no need to downgrade to RTX 4090)
- Linux x86_64, NVIDIA driver compatible with CUDA 12.8
- Python 3.10
- PyTorch 2.9.1 / torchvision 0.24.1 CUDA 12.8 wheel set (verify on host)
- LLaVA-1.5-7B
- Storage: **60 GB minimum recommended**, 70 GB+ if storing both benchmarks and model/cache copies. A 30 GB volume is not a safe target for the full setup.

## Quick start

```bash
# 1) Unzip, enter repository
cd CUAI-CV2_H3

# 2) Check host and create the H3 environments
bash setup/install.sh

# 3) Clone upstream implementations
bash setup/clone_methods.sh

# 4) Verify all four environments and GPU
bash scripts/verify_environment.sh

# 5) Edit .env and configs/datasets.yaml, then run the experiment stages
cp .env.example .env
# Set MME_ROOT, POPE_ROOT, POPE_FILE, MODEL_PATH

bash scripts/run_blind.sh
bash scripts/run_original.sh
bash scripts/run_compression.sh

# 6) Analyze the predictions
bash scripts/analyze.sh
```

The run scripts intentionally stop with a clear message until the corresponding method adapter command is configured. This prevents silently substituting a different algorithm or claiming results from a method that was not actually executed. See `docs/IMPLEMENTATION_STATUS.md` and `docs/ADAPTER_CONTRACT.md`.

## H3 analysis definitions

- `blind_solvable`: Vanilla blind prediction matches the benchmark ground truth under the benchmark's official answer-normalization rule.
- `visual_dependent`: complement of `blind_solvable` among valid blind-evaluated question IDs.
- `accuracy_all`: accuracy over all valid question IDs.
- `accuracy_visual_dependent`: accuracy over the frozen visual-dependent subset.
- `blind_solvable_ratio`: number of blind-solvable questions divided by valid blind-evaluated questions.

Report both the absolute accuracy and the change after excluding blind-solvable items. Do not compare methods on different subsets. For MME, retain its official category scoring procedure; do not reduce its official score to a generic exact-match accuracy without documenting that change. POPE can use the official evaluator / yes-no normalization.

## Output schema

Each prediction JSONL row must contain at least:

```json
{"question_id":"...","dataset":"pope","method":"vanilla","condition":"blind","prediction":"yes","ground_truth":"yes","is_correct":true}
```

Optional fields: `image`, `prompt`, `raw_response`, `latency_sec`, `input_visual_tokens`, `retained_visual_tokens`, `seed`, `model_id`, `run_id`.

Raw predictions are append-only. Never overwrite a completed run; use a new `run_id`.

## Reproducibility rules

1. Record Git commit hashes for this repository and each upstream implementation.
2. Record checkpoint identifier / revision and dataset version.
3. Keep token budget and method-specific parameters in `configs/experiment.yaml`.
4. Use deterministic question ordering and save question IDs.
5. Save raw generations before evaluation.
6. Validate blind labels manually on a small sample before launching full runs.
7. Report failures, skipped IDs, and invalid outputs; do not silently drop them.

## License

The orchestration code in this repository is provided under the MIT License. Third-party implementations and model/data assets retain their own licenses and terms. See `THIRD_PARTY.md`.
