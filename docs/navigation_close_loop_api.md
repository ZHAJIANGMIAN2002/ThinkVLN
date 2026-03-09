# Navigation Model and Close-Loop Evaluator API

This note documents the API boundary between:

- `ThinkVLNActorNavigationModel` in `thinkvln/models/navigation_model.py`
- `VLNEvaluator` in `thinkvln/eval/close_eval_runner.py`

The goal is to make responsibilities explicit, especially for `--ladder_mode subtask`.

## Scope

This doc focuses on ThinkVLNActor subtask closed-loop evaluation.  
It does not describe StreamVLN-specific internals.

## Responsibility Split

### `ThinkVLNActorNavigationModel` owns

- model forward inference and output parsing
- query token construction and model input packing
- memory-bank state (`memory_bank_images`, `memory_bank_subtasks`, subtask anchors)
- progress/done post-processing
- action selection from action logits

### `VLNEvaluator` owns

- Habitat env lifecycle (`config_env`, reset, step, close)
- episode sharding/sampling and `summary_full` parsing
- GT replay for positions/subtask alignment
- replay-to-subtask-start and memory priming calls
- rollout loop termination logic (distance threshold, budget, episode_over)
- metrics aggregation and result file writing

### Non-goals per class

- Navigation model does not compute geodesic success metrics.
- Evaluator does not inspect raw model logits or do model-side prompt/token logic.

## API Contract: Navigation Side

### Base interface

`NavigationModel` defines:

- `eval() -> None`
- `predict_action(observation, instruction, plan=None, prev_subtask=None, **kwargs) -> Tuple[int, Optional[str]]`

Action IDs are evaluator-wide:

- `0=stop`
- `1=forward`
- `2=turn_left`
- `3=turn_right`

### ThinkVLNActor methods used by close-loop

- `reset_episode_state(episode_key: Optional[str] = None) -> None`  
  Clear per-episode memory/progress/subtask state.

- `record_memory_observation(observation: Image.Image, subtask_id: int, episode_key: Optional[str] = None) -> None`  
  Append a replay frame into memory bank without running inference.

- `predict_action_with_progress_and_done(...) -> Tuple[int, float, bool]`  
  Inputs:
  - `observation: Image.Image` (current RGB+map image)
  - `instruction: str`
  - `subgoal: str`
  - `episode_key: Optional[str]`
  - `subtask_id: Optional[int]`
  Outputs:
  - `action_id: int`
  - `pred_progress: float` in `[0, 1]`
  - `done: bool`

## API Contract: Evaluator Side

### Main entry

- `eval_subtask_closed_loop(idx: int, summary_full: Dict[str, Dict[str, Any]]) -> Dict[str, float]`

Returns additive scalar stats for distributed all-reduce, including:

- episode counters (`episodes_total`, `episodes_evaluated`, `episodes_missing_meta`, `episodes_malformed`)
- subtask counters (`subtasks_total`, `subtasks_success`)
- progress metrics (`progress_abs_error_sum`, `progress_count`)
- steps-to-subgoal accumulation (`steps_success_sum`, `steps_success_count`)

### Key internal helpers

- `_replay_gt_positions(...)`  
  Replay GT actions to get per-step positions for subgoal distance checks.

- `_replay_to_frame_with_memory(...)`  
  Replay from episode start to subtask start frame and prime model memory via
  `record_memory_observation` for frames in `[0, start_frame)`.

- `_safe_geodesic_distance(...)`  
  Robust distance with Euclidean fallback.

## Data Contracts Between Classes

### Evaluator -> NavigationModel

Per rollout step, evaluator sends:

- image from `prepare_image_with_map(observations["rgb"], env.get_metrics())`
- `instruction` for episode
- `subgoal_text` from parsed plan
- `episode_key`
- `subtask_id`

### NavigationModel -> Evaluator

Returns:

- action ID for `env.step(action)`
- predicted progress for MAE logging
- done flag (currently optional for evaluator control, but available)

## Subtask Closed-Loop Sequence

1. evaluator parses episode metadata (`actions`, `subtask_sequence`, `plan`)
2. evaluator computes subtask spans and GT positions
3. for each subtask span:
   - reset nav episode state
   - replay to subtask start with memory priming (`_replay_to_frame_with_memory`)
   - rollout online with `predict_action_with_progress_and_done`
   - check distance-to-subgoal success and step budget
   - write per-subtask JSONL detail
4. evaluator returns scalar stats for global reduction

## Extension Rules

If you change one side, keep this boundary stable:

- If adding a new closed-loop strategy in evaluator, do not directly mutate model internals beyond exposed methods (`reset_episode_state`, `record_memory_observation`, predict APIs).
- If changing model memory logic, keep evaluator-facing signatures unchanged unless updating both sides together.
- If adding another navigation wrapper for subtask mode, it should provide equivalent behavior for:
  - per-episode reset
  - replay-memory ingest
  - action/progress inference

