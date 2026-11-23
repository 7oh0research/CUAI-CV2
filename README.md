# H1/H3 VLM Token Compression Benchmark

This repository retains the original full H1 benchmark workflow and adds an H3-only workflow.

## Recommended: H3-only run

H1 is treated as prior-work/literature analysis; the H3 profile performs direct experiments for visual-dependence bias using LLaVA-1.5-7B, Vanilla, FastV, SparseVLM, and VisionZip at a 64-token budget.

```bash
bash start.sh --profile h3
```

See [H3_GUIDE.md](H3_GUIDE.md) for dataset layout, setup, metrics, outputs, and runtime caveats.

## Original full workflow

The previous H1/full benchmark remains available:

```bash
bash start.sh
# or
bash run_pipeline.sh
```

See [RUN_GUIDE.md](RUN_GUIDE.md) for the original workflow.
