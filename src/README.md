# src

`vlmeval/` package:

- `schema.py`, `io.py` - per-sample record schema and JSONL I/O
- `models/llava.py` - LLaVA-1.5 wrapper; compressed tokens are injected via `inputs_embeds`, timing via first-token hook (GPU)
- `methods/` - `none`, `random`, `avgpool`, `cls_attn`
- `data/` - benchmark adapters (POPE, TextVQA, MME; GQA TODO) and scoring
- `analysis/` - `panel` (aligned arrays), `metrics`, `stats` (bootstrap, McNemar, rank stability), `hypotheses` (H1-H5)
- `synthetic.py` - fake records for GPU-free pipeline checks

To add a compression method: write `fn(feats[N,D], Context) -> [keep,D]` in `methods/__init__.py`, register it in `METHODS`, and list it in the config.
