# close_eval

This directory contains the refactored closed-loop evaluator for ThinkVLN/StreamVLN/ThinkVLNActor.

## Overview

`close_eval.py` is a compatibility facade. Core logic is split into small modules:

- `close_eval.py`: facade + legacy exports + script entrypoint.
- `close_eval_cli.py`: CLI parsing and top-level orchestration (`eval`, `evaluate`).
- `close_eval_runner.py`: `VLNEvaluator` and environment rollout loops.
- `close_eval_models.py`: model loading and `build_nav_model(...)`.
- `close_eval_utils.py`: pure helper functions and ladder summary utilities.
- `close_eval_dist.py`: distributed helpers and scalar all-reduce.

## Main Entry Points

- Script mode:
  - `python thinkvln/eval/close_eval.py --help`
- Programmatic mode:
  - `from thinkvln.eval.close_eval import eval, evaluate, VLNEvaluator`

## Typical Usage

### Legacy closed-loop eval

```bash
python thinkvln/eval/close_eval.py \
  --model_type thinkvln_actor \
  --model_path /path/to/checkpoint \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split val_unseen \
  --output_path ./results/env_eval \
  --ladder_mode legacy
```

### Ladder eval (subtask / oracle_switch / both)

```bash
python thinkvln/eval/close_eval.py \
  --model_type thinkvln_actor \
  --model_path /path/to/checkpoint \
  --ladder_mode both \
  --summary_full_path /path/to/summary_full.jsonl \
  --output_path ./results/env_eval
```

## Outputs

- `result.json`: per-episode legacy results plus final aggregate line.
- `subtask_closed_loop_rank{rank}.jsonl`: per-subtask details for ladder subtask mode.
- `oracle_switch_rank{rank}.jsonl`: per-episode details for ladder oracle-switch mode.
- `ladder_summary.json`: aggregated ladder metrics (written by rank 0).

## Notes

- Ladder modes require:
  - `--model_type thinkvln_actor`
  - `--summary_full_path` set to valid JSONL metadata.
- The facade keeps old imports stable, so existing code importing from `thinkvln.eval.close_eval` should continue to work.
