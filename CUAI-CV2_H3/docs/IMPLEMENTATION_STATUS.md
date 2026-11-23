# Implementation status

Included and ready:
- H3-only repository structure and experiment configuration for Vanilla, FastV, SparseVLM, VisionZip.
- Host preflight and four isolated Python 3.10/CUDA 12.8 environments.
- Upstream checkout script and commit-hash capture.
- JSONL input/output helpers and blind-solvable analysis with a frozen Vanilla blind set.
- Run-script guardrails and documented adapter contract.

Not yet validated / intentionally not claimed complete:
- End-to-end model loading and inference adapters for the four methods.
- MME official scoring integration and POPE official evaluator integration.
- A real Vast.ai RTX 5090 installation and full benchmark run.

Before reporting experimental results, implement and validate the adapters against the exact pinned upstream commits. This is necessary for scientific validity, not merely packaging.
