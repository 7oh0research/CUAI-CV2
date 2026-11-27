# Method implementations and deviations from the papers

All methods are re-implemented in `vtcb/model.py` for ONE model so the comparison is
controlled. They are faithful in selection logic but not byte-identical to official code.
Check each against the official repo before publishing; list any deviation in the paper.

| method | where | selection | deviation |
|---|---|---|---|
| random | pre | seeded uniform sample, order kept | - |
| pool2d | pre | 24x24 grid adaptive avg-pool to sqrt(K)^2 | needs square budget |
| VisionZip | pre | CLS-attention top-D dominant (layer -2) + C contextual tokens by similarity merging | D/C = 85/15 split; uniform target selection; no fine-tuning |
| DivPrune | pre | greedy max-min cosine diversity | features before projector |
| FastV | in-LLM | after layer 2 keep top-K by last-token attention | mask-based (see below) |
| PyramidDrop | in-LLM | layers 2/10/18 keep K, K/2, K/4 by last-token attention | stage ratio fixed by config, not tuned |
| DART | in-LLM | after layer 2: 8 pivots by key-norm proxy (hidden norm), drop tokens most similar to pivots | uses hidden states, not keys |

Mask-based in-LLM pruning: pruned image tokens are masked as keys from the pruning layer on
(prefill and decode). Equivalent in output to dropping them with original position ids.
Requires eager attention. Do not measure speed with this repo.

Budget definition: tokens kept right after the first pruning point. PyramidDrop therefore
uses fewer tokens on average than FastV at the same budget; `kept@layer` is logged.
