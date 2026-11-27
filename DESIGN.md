# Experiment design (pre-registered)

## Question
Do language-prior ("blind") items change the ranking of visual-token pruning methods?
Inspired by VTC-Bench (filter items where downsampling already succeeds), we add a second
filter: items vanilla LLaVA solves with NO visual information (blank and noimage both correct).

## Factors
- Model: LLaVA-1.5-7B (576 visual tokens). One model; extend in v2.
- Methods (7): random, pool2d (= downsampling), VisionZip, DivPrune  [pre-LLM]
               FastV, PyramidDrop, DART                           [in-LLM]
- Budgets: 144 / 64 / 36 retained image tokens (75 / 89 / 94% reduction)
- Benchmarks (13): POPE, MME, TextVQA, ScienceQA-IMG, MMBench-EN, GQA, MMStar, AI2D,
  SEED-Bench(image), ChartQA, OCRBench, RealWorldQA, VizWiz
- Controls: vanilla image / blank (mean-color image) / noimage

## Subsets per (benchmark, budget)
all | nonblind | difficult@b (pool2d_b wrong) | difficult_nonblind@b

## Hypotheses and pre-registered tests
- H3-main: ranking on `all` differs from `nonblind`.
  Evidence = Kendall tau (bootstrap CI) and number of method pairs whose paired-bootstrap
  95% CI is significant on `all` and significant with the opposite sign on `nonblind`.
  Only benchmarks with blind_frac >= 0.10 count. Claim "ranking changes" only if
  >= 1 reversed pair in >= 2 benchmarks. Otherwise report "ranking robust to blind items".
- H3-spread: retention gap between methods widens on `nonblind` (spread_nonblind > spread_all).
- H2: mean retention over benchmarks is reported with the vanilla-noimage floor from the
  previous study (~58%) as reference.
- H4: answer flip rate vs vanilla >> |accuracy delta| (gain and loss cancel).
- VTC-Bench replication: does pool2d look strong on `all` but weak on `difficult`?

## Not in scope
Latency/FLOPs (in-LLM methods are mask-based here), circular MMBench, official
GPT-judged scores, multi-model generalization.

## Cost estimate (RTX 5090, rough, NOT measured)
Conditions = 3 controls + 7x3 = 24 generations per item. ~24k items total
-> ~580k generations. With short answers expect roughly 0.1-0.2 s each => 16-32 GPU-hours.
Measure from quick run (`sec` column) before committing.
