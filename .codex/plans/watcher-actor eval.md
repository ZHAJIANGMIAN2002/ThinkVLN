# Two-System Watcher-Actor Eval Plan

## Summary
- Build a new standalone package under [thinkvln/eval/two_system_eval](/mnt/swx/ThinkVLN/thinkvln/eval/two_system_eval), separate from `close_eval`, but reuse small helpers from [thinkvln/eval](/mnt/swx/ThinkVLN/thinkvln/eval).
- Drive the whole evaluator from one YAML file, similar to [config/eval_config.yaml](/mnt/swx/ThinkVLN/config/eval_config.yaml). The runtime entry should be `python thinkvln/eval/two_system_eval.py --config config/two_system_eval.yaml`.
- Use one unified watcher JSON schema everywhere:
  - `{"memory":"...","done":bool,"subtask":"..."}`

## Public Interfaces
- Add a config file at [config/two_system_eval.yaml](/mnt/swx/ThinkVLN/config/two_system_eval.yaml) with fixed sections:
  - `actor`: `model_type`, `model_path`, `base_model_path`, `device`, `memory_num_history_images`, `done_threshold`
  - `watcher`: `backend`, `model_name`, `model_path`, `base_model_path`, `api_base_url`, `api_key_env`, `reasoning_effort`, `request_timeout`, `max_retries`
  - `env`: `habitat_config_path`, `eval_split`, `sample_rate`, `target_episode_key`
  - `rollout`: `max_steps_per_wakeup`, `progress_threshold`, `episode_step_cap`, `forbidden_actions`
  - `output`: `output_dir`, `save_trace_jsonl`, `save_summary_json`, `save_step_debug_html`
  - `runtime`: `log_level`, `world_size`, `dist_url`, `dist_timeout_minutes`
- Keep CLI minimal:
  - required public arg: `--config`
  - allow only hidden/infra args needed for `torchrun` such as `--local_rank`
  - do not expose the watcher/actor/env settings as long public flags
- Define one watcher return type in code:
  - `memory: str`
  - `done: bool`
  - `subtask: str`
  - `raw_response: dict`
  - `wakeup_reason: str`
- Extend actor wrappers in [navigation_model.py](/mnt/swx/ThinkVLN/thinkvln/models/navigation_model.py):
  - `predict_action_with_progress_and_done(..., hint: Optional[str] = None)`
  - append watcher memory as `hint` to the actor prompt only when non-empty

## Implementation Changes
- Add a simple runner that owns the episode plan state:
  - parse instruction and plan
  - initialize `done_steps=[]`, `active_step=plan_steps[0]`, `pending_steps=plan_steps[1:]`
  - reset env and call watcher init with first observation
  - pass watcher `memory` as actor `hint`, and watcher `subtask` as actor subgoal
  - execute actor until wakeup on any of: `max_steps_per_wakeup`, progress above threshold, stop action, actor `done`, or `env.episode_over`
  - send the rollout slice plus previous memory and `Done / Active / Pending` to watcher
  - if watcher `done=false`, keep the same active step
  - if watcher `done=true`, promote the next pending step
  - if watcher `done=true` and pending is empty, end the episode immediately
- Implement watcher backends behind one interface:
  - `ApiWatcherBackend` using OpenAI-compatible chat completions
  - `LocalWatcherBackend` using in-process multimodal generation with existing Qwen3VL loading/inference utilities and optional LoRA base-model resolution
- Keep watcher prompting stage-specific internally, but parse the same schema on both calls:
  - init prompt: first observation + instruction + full plan
  - update prompt: `Done / Active / Pending` + prior memory + rollout images + rollout actions
- Add runner-owned global memory for every simulator step:
  - RGB image reference
  - action
  - position and rotation
  - actor progress and done
  - active plan step
  - watcher hint used
  - rollout / watcher boundary markers
  - optional predicted waypoint if available from actor type
- Write outputs under the configured output dir:
  - per-rank episode trace JSONL
  - rank-0 aggregate summary JSON
  - optional debug HTML if enabled in config

## Config Design
- Follow the nested YAML style already used by [config/eval_config.yaml](/mnt/swx/ThinkVLN/config/eval_config.yaml), not flat argparse-style names.
- The new config should be complete enough that normal runs need no extra runtime flags beyond `--config`.
- Default values to lock in for v1:
  - `watcher.backend: api`
  - `rollout.max_steps_per_wakeup: 10`
  - `rollout.progress_threshold: 0.85`
  - `rollout.forbidden_actions: [0]` during normal subtask rollouts
  - `output.save_trace_jsonl: true`
  - `output.save_summary_json: true`
- Keep actor and watcher configuration separate in YAML even if both may use Qwen-family models, so swapping watcher backend does not affect actor loading.

## Test Plan
- Unit tests for config loading:
  - full YAML parses into actor/watcher/env/rollout/output/runtime sections
  - `--config` is sufficient for normal execution
  - missing required config keys fail with clear errors
- Unit tests for watcher prompt builders:
  - both init and update prompts parse only `memory/done/subtask`
  - update prompt includes `Done / Active / Pending`, prior memory, rollout actions, and rollout images
- Unit tests for actor hint threading:
  - hint is appended only when present
  - no-hint behavior stays unchanged
- Backend tests:
  - API watcher request payload shape, retry handling, and JSON parsing
  - local watcher model loading, LoRA base-model resolution, and JSON parsing
- Runner tests with fake env, fake actor, and fake watcher:
  - init flow
  - each wakeup trigger
  - handoff on `done=true`
  - refinement on `done=false`
  - final completion when pending is empty
  - global memory and trace contents
- CLI smoke tests:
  - `python thinkvln/eval/two_system_eval.py --config ...`
  - distributed launch keeps only infra args outside YAML

## Assumptions
- The new package stays intentionally simpler than `close_eval`; no compatibility facade or extra ladder modes are added in v1.
- Watcher memory is always cumulative and stored in the single `memory` field.
- Watcher `subtask` is always the next actor-facing text, whether continuing the current step or promoting a new one.
- The evaluator uses YAML as the single source of truth for runtime settings; long CLI override lists are out of scope for this implementation.
