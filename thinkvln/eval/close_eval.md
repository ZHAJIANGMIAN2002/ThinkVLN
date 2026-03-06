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

### Subtask closed-loop eval (main pipeline)

```bash
python thinkvln/eval/close_eval.py \
  --model_type thinkvln_actor \
  --model_path /path/to/checkpoint \
  --ladder_mode subtask \
  --summary_full_path /path/to/summary_full.jsonl \
  --output_path ./results/env_eval
```

## Memory Model Evaluation Flow (ThinkVLNActor)

This is the main process used in `ladder_mode=subtask` for the memory model:

1. Episode setup
   - `VLNEvaluator` loads one episode and parses `summary_full` metadata (`actions`, `subtask_sequence`, `plan`).
   - `ThinkVLNActorNavigationModel.reset_episode_state(...)` clears internal memory-bank state.

2. Subtask start state
   - Evaluator replays the simulator to the subtask start frame to get the correct environment state.
   - Replay is only for simulator state alignment, not for feeding historical images to the model.

3. Step-by-step closed-loop rollout
   - At each step, evaluator calls `predict_action_with_progress_and_done(...)` with current observation and subtask id.
   - Inside the navigation model, the current frame is appended to an internal memory bank:
     - `memory_bank_images`
     - `memory_bank_subtasks`
   - If this is the first frame of a subtask, its bank index is marked as a subtask-start anchor.

4. Memory sampling for model input
   - The wrapper samples history indices from the bank using `select_memory_frame_indices(...)`.
   - History is capped by `--memory_num_history_images`.
   - Current frame is always included; sampled history + current frame form the multi-image input.
   - Prompt includes instruction, current subgoal, previous progress, and a memory hint when history exists.

5. Prediction and state update
   - Model returns `action_logits`, `progress_preds`, and optionally `done_preds`.
   - Wrapper selects the action, computes/clamps progress, derives done, and updates `prev_progress`.
   - Evaluator steps the environment, checks distance-to-subgoal success, and records metrics.

Key behavior: memory context comes from the model wrapper’s own online memory bank during rollout, not from replayed observation history.

## Outputs

- `subtask_closed_loop_rank{rank}.jsonl`: per-subtask detailed rollout records.
- `ladder_summary.json`: aggregated subtask closed-loop metrics (written by rank 0).

## Notes

- Subtask closed-loop requires:
  - `--model_type thinkvln_actor`
  - `--summary_full_path` set to valid JSONL metadata.
- The facade keeps old imports stable, so existing code importing from `thinkvln.eval.close_eval` should continue to work.
