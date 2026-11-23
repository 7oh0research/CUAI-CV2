# Method adapter contract (required before full runs)

This package intentionally does not contain copied/modified third-party algorithm code. Each method must be connected to its pinned upstream implementation and tested against its documented behavior before claiming results.

## Required adapter interface

A validated adapter must expose a command that accepts:

- dataset name and input question file
- condition (`blind`, `original`, `compressed`)
- model checkpoint and prompt template
- method name and method-specific config
- output JSONL path

Every output row must preserve the source `question_id` and include `dataset`, `method`, `condition`, `prediction`, `ground_truth`, `is_correct`, and `run_id`. Compressed methods should additionally record original and retained visual token counts where available.

## Blind condition

Use the same LLaVA-1.5-7B language backbone and decoding settings as other conditions, but remove the image from the model input. Do not replace the VLM with a different standalone language model. Freeze the blind-solvable question IDs from this Vanilla run and reuse the exact set for all methods.

## Algorithm validation checklist

- Vanilla: no token pruning; verify the visual token count matches the uncompressed LLaVA image path.
- FastV: verify the selected pruning layer and keep/prune ratio against the paper and pinned implementation.
- SparseVLM: verify text-guided scoring, pruning, and token recycling are active; log actual retained tokens.
- VisionZip: verify dominant/contextual token selection and the configured token budget; log actual retained tokens.
- For every method, compare a small deterministic batch with the upstream example / expected behavior before full MME and POPE runs.

## Why the run scripts stop

The upstream projects evolve independently and may have incompatible LLaVA forks, Transformers versions, CLI arguments, and evaluation scripts. A generic guessed command would risk silently running Vanilla for a purported compression experiment. The run scripts therefore fail closed until the adapter is explicitly implemented and validated.
