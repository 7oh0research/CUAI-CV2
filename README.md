# CUAI-CV2
in CUAI, 26-2, team-project

## Research topic

**Analysis paper:** diagnosing problems in how visual-token efficiency methods for VLMs are evaluated, and proposing better metrics, with **LLaVA-1.5-7B** as the backbone. Training-free; uses public checkpoints only.

Core question: do current efficiency metrics (token count, FLOPs, average accuracy retention) rank methods correctly? We test five hypotheses (H1-H5), propose new metrics, and validate them with a meta-evaluation. See [docs/research_plan.md](docs/research_plan.md).

## Structure

| Path | Purpose |
|------|---------|
| `src/vlmeval/` | Package: model wrapper, compression methods, data adapters, analysis |
| `src/vlmeval/efficiency/` | **H1** harness: analytic FLOPs, manual LLaVA engine (sdpa/eager, in-LLM pruning), profiler, rank analysis, report |
| `scripts/profile_efficiency.py` | H1: measure latency/memory over method x budget x batch x attention (GPU) |
| `scripts/analyze_h1.py` | H1: FLOPs-rank vs latency-rank analysis, report + plots (CPU) |
| `scripts/run_experiment.py` | Run (benchmark x method x ratio) grid, log per-sample JSONL (GPU) |
| `scripts/analyze.py` | Metrics, CIs, hypothesis tests, meta-evaluation, report (CPU) |
| `configs/` | Experiment configs (`h1.yaml`, `h1_tiny.yaml`, `pilot.yaml`) |
| `tests/` | Unit tests for analysis, methods and the H1 engine (no GPU needed) |
| `docs/` | Research plan, [H1 experiment](docs/h1_experiment.md), frozen metric spec, literature survey template |
| `data/` | Datasets (git-ignored) |
| `notebooks/` | Exploration |

## Getting started

```bash
git clone https://github.com/7oh0research/CUAI-CV2.git
cd CUAI-CV2
pip install -e ".[dev]"                 # analysis only (CPU)
pytest
python scripts/analyze.py --synthetic --out outputs/synthetic_report

# On the GPU machine (24GB+ recommended, fp16, unquantized):
pip install -e ".[infer,dev]"
python scripts/run_experiment.py --config configs/pilot.yaml --dry-run
python scripts/run_experiment.py --config configs/pilot.yaml
```

### H1 (FLOPs vs measured latency)

```bash
python scripts/profile_efficiency.py --flops-only                          # analytic FLOPs table, no GPU
python scripts/analyze_h1.py --synthetic --out outputs/h1_synthetic        # pipeline check on simulated latencies
python scripts/profile_efficiency.py --config configs/h1_tiny.yaml --tiny  # CPU smoke test (timings meaningless)

# GPU machine (24GB+, fp16, unquantized):
python scripts/profile_efficiency.py --config configs/h1.yaml              # resumable
python scripts/analyze_h1.py --records outputs/h1/records.jsonl --out outputs/h1/report
```

See [docs/h1_experiment.md](docs/h1_experiment.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
