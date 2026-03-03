# Actor Closed-Loop Ladder (Extend `close_eval.py`)

## Summary
Implement the two missing actor closed-loop rungs directly in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py), using `summary_full.jsonl` as the subtask GT source and supporting **ThinkVLNActor only** in this pass.

The ladder outputs will be:

1. **Subtask closed-loop (Actor core metric)**  
   Metrics: `subtask_success_rate`, `steps_to_subgoal` (success-only mean), `progress_mae`.
2. **Oracle-switch closed-loop (Actor upper bound in full episode)**  
   Metrics: `sr`, `spl` (episode-level).

Legacy env eval behavior for non-actor models remains intact.

---

## Public API / Interface Changes

1. Update [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py) CLI:
- `--model_type`: add `thinkvln_actor` to choices.
- `--ladder_mode`: `legacy | subtask | oracle_switch | both` (default `legacy`).
- `--summary_full_path`: required for `subtask/oracle_switch/both`.
- `--base_model_path`: optional, used when `model_path` is LoRA adapter.
- `--subgoal_success_distance`: default `0.5`.
- `--subtask_step_budget_factor`: default `2.0`.

2. Add actor navigation wrapper in [navigation_model.py](/mnt/swx/ThinkVLN/thinkvln/models/navigation_model.py):
- `ThinkVLNActorNavigationModel` with:
  - `predict_action(...)`
  - `predict_action_with_progress(...)`

3. Keep existing `NavigationModel` abstraction unchanged for compatibility; actor-specific progress method is additive.

---

## Implementation Plan

1. Add subtask GT loader and helper utilities in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py):
- `load_summary_full(path) -> Dict[episode_key, meta]`
- `episode_key = f"{scene_id}_{episode_id}"` mapping helper
- `build_subtask_spans(subtask_sequence)` returning `(subtask_idx, start_frame, end_frame)`
- `timeline_progress(step_idx, gt_subtask_steps)` for progress target
- `step_budget = max(1, ceil(gt_subtask_steps * factor))`

2. Add `ThinkVLNActorNavigationModel` in [navigation_model.py](/mnt/swx/ThinkVLN/thinkvln/models/navigation_model.py):
- Build actor prompt using fixed subgoal text (plan step).
- Processor path mirrors action collator logic: image+text prompt, append alternating action/progress query token IDs.
- Forward pass uses actor heads (via model forward in action mode) and extracts first-step action/progress.
- Progress is clipped to `[0, 1]` for metric computation.
- Closed-loop execution uses only first predicted action each step.

3. Integrate actor model loading in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py):
- Reuse loader logic from [evaluate.py](/mnt/swx/ThinkVLN/thinkvln/engine/evaluate.py) style for LoRA/full checkpoints.
- Instantiate `ThinkVLNActorNavigationModel` when `model_type=thinkvln_actor`.

4. Implement rung 1: subtask closed-loop evaluator in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py):
- For each assigned episode:
  - Fetch GT metadata from `summary_full`.
  - Precompute GT positions by replaying GT actions once on env (for subgoal positions).
  - For each subtask span:
    - Reset episode.
    - Replay GT actions to subtask start frame.
    - Fix subgoal text for the whole rollout.
    - Roll out actor until success/fail:
      - Success: geodesic distance to subgoal position `<= 0.5m`.
      - Fail: exceeds subtask step budget (`2x` GT subtask steps) or episode termination.
      - Record per-step progress absolute error vs timeline GT target.
    - Log per-subtask result line.
- Aggregate:
  - `subtask_success_rate = successful_subtasks / total_subtasks`
  - `steps_to_subgoal = mean(steps on successful subtasks only)`
  - `progress_mae = mean(abs_error across all evaluated subtask steps)`

5. Implement rung 2: oracle-switch closed-loop evaluator in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py):
- Full-episode rollout.
- Oracle switching uses **GT frame boundaries** from `subtask_sequence` by current step index.
- At each step:
  - Determine oracle current subtask from GT sequence index.
  - Use corresponding plan step as fixed subgoal input for that step.
  - Actor predicts action; env steps.
- Aggregate episode metrics:
  - `sr` mean
  - `spl` mean

6. Distributed aggregation updates in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py):
- Replace tensor-cat logic for ladder modes with summed counters/statistics via `all_reduce`.
- Keep legacy gather behavior for `legacy` mode unchanged.

7. Output artifacts in [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py):
- Per-rank JSONL detail files:
  - `subtask_closed_loop_rank{rank}.jsonl`
  - `oracle_switch_rank{rank}.jsonl`
- Rank-0 summary file:
  - `ladder_summary.json` with two sections:
    - `subtask_closed_loop`
    - `oracle_switch_closed_loop`
- Do not append ladder results into legacy `result.json` to avoid mixed schema.

8. Validation and guardrails:
- If ladder mode enabled and `summary_full_path` missing/unreadable: fail fast with explicit error.
- If episode missing in summary mapping: skip with counter and warning.
- If plan/subtask sequence malformed: skip episode and continue.

---

## Tests and Scenarios

1. Add pure-unit tests in [test_close_eval_ladder.py](/mnt/swx/ThinkVLN/thinkvln/tests/test_close_eval_ladder.py):
- Subtask span extraction from `subtask_sequence`.
- Timeline progress target behavior (`gt_steps=0`, normal case).
- Step budget calculation with factor `2.0`.
- Oracle subtask lookup by step index against GT boundaries.
- Aggregation math:
  - success rate
  - success-only steps mean
  - progress MAE

2. Add wrapper behavior test (mocked model/processor) in [test_close_eval_ladder.py](/mnt/swx/ThinkVLN/thinkvln/tests/test_close_eval_ladder.py):
- `predict_action_with_progress` returns first-step action.
- Progress clipping to `[0,1]` is applied before metric use.

3. Smoke checks (non-Habitat):
- `python -m py_compile` for modified files.
- Run new unit tests under `conda activate vln`.

---

## Assumptions and Defaults (Locked)

1. Environment for execution is `conda activate vln`.
2. Implementation location is [close_eval.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval.py) (not a new eval file).
3. First pass supports `ThinkVLNActor` only for ladder modes.
4. GT source is `summary_full.jsonl`.
5. Subtask success rule is distance-to-subgoal with threshold `0.5m`.
6. Oracle-switch rule is GT frame-boundary switching.
7. Action execution is strict closed-loop: first predicted action each step.
8. Progress MAE target is GT subtask timeline; predicted progress is clipped to `[0,1]`.
9. Subtask step budget is `2x` GT subtask steps.
10. Oscillation/loop metric is removed from ladder outputs.
11. Oracle-switch summary reports `SR/SPL` only.
