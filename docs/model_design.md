# ThinkVLN Model Design

Current design summary aligned with:
- [watcher_openai_annotation_prompts.md](/mnt/swx/ThinkVLN/docs/watcher_openai_annotation_prompts.md)
- [thinkvln_actor.py](/mnt/swx/ThinkVLN/thinkvln/models/thinkvln_actor.py)

## 1. System Overview

ThinkVLN is organized as a two-level navigation stack:

- `Actor`: low-level executor that predicts the next action, scalar progress, and binary done signal from current visual context
- `Watcher`: high-level state updater that maintains compact memory and decides whether the current subtask should continue or hand off to the next one

The current codebase is strongest on:
- a trainable multimodal `Actor`
- a prompt-defined `Watcher` annotation format for supervision and data construction

## 2. Actor Design

### Base model

- Backbone: `ThinkVLNForConditionalGeneration`
- Actor wrapper: `ThinkVLNActor`
- Visual-language base is inherited from the underlying ThinkVLN / Qwen3-VL style model

### Core prediction targets

The actor predicts three control signals:

- `action_logits`: 4-way discrete action classification
  - `forward`
  - `turn_left`
  - `turn_right`
  - `stop`
- `progress_preds`: scalar subtask progress
- `done_preds`: binary completion probability for the current step/subtask state

### Query-token interface

The actor uses learnable query-token pairs instead of action-token search.

- `num_query_tokens = K`
- Actual appended query length is `2 * K`
- Query tokens are interleaved:
  - `action_0, progress_0, action_1, progress_1, ...`

At inference/training time:
- even query positions feed the action head
- odd query positions feed the progress head
- the first progress query also feeds the done head

### Head architecture

The current actor head stack is:

1. shared projector
   - `Linear -> LayerNorm -> GELU -> Dropout`
2. action head
   - OpenVLA-OFT style `MLPResNet`
   - output shape: `[batch, K, 4]`
3. progress head
   - OpenVLA-OFT style `MLPResNet`
   - output shape: `[batch, K]`
   - current training/inference mainly uses the first scalar
4. done head
   - binary linear head on the first projected progress token
   - output: `done_logits`, `done_preds = sigmoid(done_logits)`

### Training modes

`ThinkVLNActor.forward()` supports two modes:

- Action mode
  - triggered when `action_labels` is provided
  - computes `action_loss + progress_loss + done_loss`
  - no LM loss
- CoT mode
  - triggered when `action_labels` is absent
  - computes only language-model loss
  - no action/progress/done prediction loss

This keeps one model usable for both control prediction and reasoning-style supervision.

### Losses

- action: cross-entropy
- progress: MSE by default, optional Huber
- done: BCE-with-logits

Weighted sum:

```text
total_loss =
    action_loss_weight * action_loss +
    progress_loss_weight * progress_loss +
    done_loss_weight * done_loss
```

### Runtime behavior

At runtime the navigation stack mainly consumes:

- the first action logit vector
- the first scalar progress prediction
- the first done probability

This makes the actor an efficient local controller, while leaving step-transition judgment to the watcher layer.

## 3. Watcher Design

The watcher side is currently defined as a compact annotation and supervision interface rather than a standalone trained network in this file set.

### Stage A: `memory_start`

Output format:

```json
{"memory_start":"..."}
```

Fixed structure:

```text
traj summary; current state; neutral status
```

Requirements:
- summarize only past progress that still matters
- make the pivot state explicit
- keep the third fragment neutral
- do not decide handoff yet

Typical neutral status:
- `active step in progress`
- `still on current step`
- `approach still ongoing`

### Stage B: rollout update

Output format:

```json
{"done":true,"next_subtask":"...","memory_end":"..."}
```

Definitions:
- `done`: whether the current active step has reached a real handoff point
- `next_subtask`: short imperative subtask text
- `memory_end`: updated watcher memory after the rollout

`memory_end` keeps the same update-friendly structure:

```text
traj summary; current state; task status
```

The first two fragments should feel like a direct continuation of `memory_start`.
The third fragment is now allowed to decide state:
- `step ongoing`
- `ready for next step`
- `task complete`

### Watcher decision rule

Current watcher logic does not rely on a map.
It judges only from:
- `memory_start`
- rollout RGB images
- rollout actions
- plan state split into `Done / Active / Pending`

`done=true` requires both:
- the current active step is naturally complete
- the rollout end is already a valid start point for the next step

This avoids early handoff cases such as:
- not yet at the intersection before a turn
- not yet fully through a doorway
- not yet aligned after a turn

### Transition logic used by watcher

- `Turn`
  - hand off only after the new heading is aligned for the next move
- `Region Transition`
  - hand off only after clearly crossing into the next region
- `Visual Approach`
  - hand off only when the target/stop point is immediate
- `General Cruise`
  - hand off only at the actual structural trigger point
- `Stop`
  - hand off only when already settled in the stop position

## 4. Actor-Watcher Interface

The current system boundary is:

- Actor handles short-horizon control prediction
- Watcher handles subtask-level memory and handoff judgment

A clean interface looks like:

### Actor inputs

- current image sequence / visual context
- text prompt or subgoal
- appended action/progress query tokens

### Actor outputs

- `action_logits`
- `progress_preds`
- `done_preds`

### Watcher inputs

- pre-pivot history images for `memory_start`
- rollout images and rollout actions
- current plan state
- previous memory

### Watcher outputs

- `memory_start`
- `done`
- `next_subtask`
- `memory_end`

## 5. Data and Supervision

Current training/supervision signals in the codebase are:

### Actor-side labels

- `action_labels`
- `progress_labels`
- `done_labels`

### Watcher-side annotations

- `memory_start`
- `done`
- `next_subtask`
- `memory_end`

This means the project already has:
- direct control supervision for the actor
- structured high-level supervision for watcher memory and handoff decisions

## 6. Current Design Takeaways

- The actor is no longer just `action + progress`; it is now `action + progress + done`
- The watcher is no longer described by old `PROCEED / RESUME / FAIL` labels in the current annotation design
- The latest watcher format is memory-centric and update-oriented:
  - `memory_start` is neutral
  - `memory_end` is a direct update with explicit task status
- The main division of labor is:
  - actor for local execution
  - watcher for memory compression and subtask transition
