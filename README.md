# CUAI-CV2
in CUAI, 26-2, team-project

## Research topic

**Analysis paper:** diagnosing problems in how visual-token efficiency methods for VLMs are evaluated, and proposing better metrics, with **LLaVA-1.5-7B** as the backbone. Training-free; uses public checkpoints only.

Core question: do current efficiency metrics (token count, FLOPs, average accuracy retention) rank methods correctly? We test five hypotheses (H1-H5), propose new metrics, and validate them with a meta-evaluation. See [docs/research_plan.md](docs/research_plan.md).

## Structure

| Path | Purpose |
|------|---------|
| `src/vlmeval/` | Package: model wrapper, compression methods, data adapters, analysis |
| `scripts/run_experiment.py` | Run (benchmark x method x ratio) grid, log per-sample JSONL (GPU) |
| `scripts/analyze.py` | Metrics, CIs, hypothesis tests, meta-evaluation, report (CPU) |
| `configs/` | Experiment configs (`pilot.yaml`) |
| `tests/` | Unit tests for analysis and methods (no GPU needed) |
| `docs/` | Research plan, frozen metric spec, literature survey template |
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
python scripts/analyze.py --records outputs/pilot/records.jsonl --out outputs/pilot/report
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
