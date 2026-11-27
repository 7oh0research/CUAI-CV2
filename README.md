# vtc-prune-bench

Compare 7 visual-token pruning methods on 13 benchmarks with LLaVA-1.5-7B, and test whether
blind (language-prior) items change the method ranking. Design: DESIGN.md. Methods: METHODS.md.

```bash
bash setup.sh && source .venv/bin/activate
python scripts/inspect_datasets.py          # 1. fix vtcb/data.py if any benchmark FAILS
bash run_all.sh quick                       # 2. sanity_check.py must print ALL PASS
bash run_all.sh full                        # 3. tmux recommended; resumable
```
Partial runs: `python scripts/run.py full --benches pope mme --methods fastv --budgets 64`.
Report: `results/<mode>/report.md`, tables in `tables/`, blind ids in `blind_ids/`.

Status: written without GPU access. Only scoring and analysis were tested (synthetic data).
Dataset field names, the hooks in vtcb/model.py, and transformers==4.51.3 compatibility are
UNVERIFIED until scripts/inspect_datasets.py and scripts/sanity_check.py pass.
