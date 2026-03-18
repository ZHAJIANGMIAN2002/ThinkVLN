# Watcher Annotation Prompts

Human-readable summary of the effective prompts in [watcher_openai_annotation.py](/mnt/swx/ThinkVLN/thinkvln/datagen/generation/watcher_openai_annotation.py).

There are two stages:
- `memory_start`
- rollout update: `done`, `next_subtask`, `memory_end`

## 1. `memory_start`

Output:

```json
{"memory_start":"..."}
```

Key points:
- Format is fixed: `traj summary; current state; neutral status`
- Keep only past progress that still matters at the pivot
- Make the pivot state explicit
- Third fragment must stay neutral, for example `active step in progress`
- Do not write `ready for next step` or `task complete`
- No frame-by-frame narration, no instruction restatement, no extra explanation

Input summary:
- Sampled RGB images from episode start to pivot
- Images are in time order
- First image is episode start
- Last image is the pivot

Example:

```text
Left the dining area and entered the hall; at the bathroom entrance facing inward; active step in progress
```

## 2. Rollout Update

Output:

```json
{"done":true,"next_subtask":"...","memory_end":"..."}
```

Key points:
- No map is available; judge only from `memory_start`, rollout images, rollout actions, and plan state
- `done=true` only if the current step is complete and the rollout end is a valid starting point for the next step
- Do not hand off early just because the current step looks mostly complete
- `next_subtask` must be short, imperative, and directly usable
- `memory_end` keeps the same structure as `memory_start`: `traj summary; current state; task status`
- First two fragments should feel like a direct update of `memory_start`
- Third fragment can now decide the state: `step ongoing`, `ready for next step`, or `task complete`

Transition checks:
- `Turn`: switch only after heading is aligned for the next move
- `Region Transition`: switch only after clearly crossing into the next region
- `Visual Approach`: switch only when the target or stop point is immediate
- `General Cruise`: switch only at the real trigger point, such as doorway, corner, or intersection
- `Stop`: switch only when the robot is already settled in the stop position

Input summary:
- Plan state split into `Done`, `Active`, `Pending`
- `memory_start`
- Rollout actions
- Sampled rollout RGB images

Example:

```text
Cleared the dining area and reached the hall entrance; aligned with the bathroom approach; ready for next step
```
