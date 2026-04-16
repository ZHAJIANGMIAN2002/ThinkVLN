# ThinkVLN Model Design

This document is a practical introduction to the current system design. It is written to match the style of `docs/repo_handoff_guide.md`: concrete modules first, then data flow, then how the actor and watcher cooperate in training and evaluation.

## 1. System overview

The current repository is organized around a two-system navigation stack:

1. a fast **actor** that predicts low-level navigation actions from the current observation, instruction, and compressed trajectory context;
2. a slower **watcher** that reviews rollout spans, updates compact memory, and decides whether the current subtask should hand off to the next one.

In practice:

- the actor is trained on the active `streamvln/` codepath and StreamVLN-style subtask supervision;
- the watcher is trained as a large Qwen vision-language model on watcher rollout data and annotations.


The high-level reason for this split is simple:

- the actor needs to run frequently and cheaply inside Habitat closed-loop evaluation;
- the watcher needs a broader, slower judgment over a rollout span: what progress has already been made, whether the active step is truly complete, and what text hint should guide the next actor segment.

## 2. Main modules

### Actor stack

The actor side is split between the active StreamVLN actor baseline and the ThinkVLN evaluation wrappers that consume it.

Primary files:

- `streamvln/model/stream_video_vln.py`: main StreamVLN model path.
- `streamvln/dataset/streamvln_actor_dataset.py`: actor sample construction, prompt format, memory layout, and 4-step action supervision.
- `streamvln/streamvln_train.py`: baseline StreamVLN training path.
- `scripts/train_streamvln_actor.py`: current training launcher and config adapter for StreamVLN actor experiments.
- `config/streamvln_actor_train*.yaml`: active actor experiment configs.

Actor-facing evaluation wrappers:

- `thinkvln/models/navigation_model.py`: unified navigation wrapper used by closed-loop evaluation.
- `thinkvln/eval/close_eval_models.py`: builds the eval-time navigation model from config/CLI args.
- `thinkvln/eval/close_eval_runner.py`: Habitat runtime loop for subtask closed-loop evaluation.

### Watcher stack

The watcher side is implemented inside `thinkvln/` as a rollout-level supervision and decision system.

Primary files:

- `thinkvln/dataset/watcher_sft_dataset.py`: watcher prompt contract, rollout-sample loading, and target JSON format.
- `thinkvln/engine/watcher_sft_trainer.py`: watcher SFT trainer.
- `thinkvln/datagen/generation/watcher_rollout_generation.py`: generates rollout spans for watcher training.
- `thinkvln/datagen/generation/watcher_openai_annotation.py`: API-based watcher annotation pipeline.
- `thinkvln/datagen/generation/watcher_openai_annotation_deploy.py`: deployment-oriented annotation path.
- `thinkvln/datagen/generation/watcher_manual_done_annotation.py`: manual done-label workflow.
- `thinkvln/datagen/generation/watcher_manual_switch_annotation.py`: manual switch / handoff workflow.
- `thinkvln/datagen/generation/watcher_merge_dataset.py`: merges watcher supervision sources.
- `scripts/train_watcher.sh`: watcher training launcher.
- `config/watcher_sft.yaml`, `config/watcher_sft_fullv2_streamv2.yaml`: current watcher training configs.

### Shared offline data pipeline

These modules build the trajectory summaries and subtask structures that both actor and watcher depend on:

- `scripts/generate_summary_full.py`: builds `summary_full.jsonl`.
- `thinkvln/datagen/generation/subtask_split.py`: splits trajectories into subtask segments.
- `thinkvln/datagen/generation/subtask_determination.py`: generates subtask determination outputs.
- `scripts/run_subtask_summary_pipeline.sh`: practical summary/subtask pipeline launcher.
- `scripts/build_streamvln_actor_dataset.py`: materializes StreamVLN actor training records from one or more summary files, with optional watcher hints.

### Evaluation stack

There are two important evaluation paths in the current system:

- **subtask evaluation**:
  - `thinkvln/eval/close_eval.py`
  - `thinkvln/eval/close_eval_cli.py`
  - `thinkvln/eval/close_eval_runner.py`
  - `scripts/run_close_eval_subtask.sh`
  - `scripts/summarize_subtask_success.py`
- **two-system evaluation**:
  - `thinkvln/eval/two_system_eval.py`
  - `config/two_system_eval*.yaml`

## 3. Actor design

The current actor is best understood as a StreamVLN-derived low-level controller with extra subtask-oriented supervision.

Core behavior:

- input: current RGB observation, instruction, current subtask text, optional watcher hint, and selected historical observations;
- output: a short action chunk, typically the next 4 actions;
- optional supervision heads or metadata: progress and done labels for subtask-aware evaluation and handoff logic.

The actor prompt builder in `streamvln/dataset/streamvln_actor_dataset.py` makes the design intent explicit. A typical actor sample can include:

- `Instruction`
- `Current subtask`
- `Next subtask`
- `Subtask start observation`
- `Watcher hint`
- `Historical observations`
- `Recent actions`
- `Steps in current subtask`

The actor is therefore not a plain end-to-end navigation policy. It is a subtask-conditioned controller with optional watcher-provided text memory and sparse visual memory.

Important implementation details:

- supervision is chunked into the next 4 actions;
- the dataset supports a special `<next>` token to mark subtask transitions;
- the data builder can inject watcher memory text from a separate watcher hint file;
- historical observations are selected with anchor-aware layouts rather than naive full-history replay.

The result is an actor that is still operationally close to StreamVLN, but trained in a way that makes it compatible with watcher-guided hierarchical control.

## 4. Watcher design

The watcher is the slower supervisory model. Its job is not to emit primitive actions every step. Its job is to judge progress over a rollout span and produce compact state that the actor can use on the next segment.

The watcher prompt and output contract are defined in `thinkvln/dataset/watcher_sft_dataset.py`.

The watcher consumes:

- the plan state split into `done`, `active`, and `pending` steps;
- a prior compressed memory string, `memory_start`;
- rollout actions over a short segment;
- sampled rollout images from that segment.

The watcher returns structured JSON with three fields:

- `memory_end`: updated compact cumulative memory;
- `done`: whether the current active step has reached a natural handoff;
- `next_subtask`: the next actor-facing subtask string, either a continuation of the current step or the promoted next step.

This is the central watcher design choice: the watcher is a rollout-level state updater and handoff judge, not merely a classifier. It performs three coupled functions:

1. compress trajectory history into reusable text memory;
2. decide whether the active subtask is genuinely complete;
3. rewrite the actor-facing subtask text for the next rollout segment.

That combination is what lets the two-system loop stay lightweight while still being more structured than a single flat actor policy.

## 5. Training data sources

### Actor training data

The actor training data is based on StreamVLN-style trajectory summaries and materialized subtask records.

Main sources:

- `data/trajectory_data/R2R_back/summary_full.jsonl`
- `data/trajectory_data/ScaleVLN_back/summary_full*.jsonl`
- materialized actor datasets such as `data/trajectory_data/streamvln_actor_train.jsonl`

The materialization step is handled by `scripts/build_streamvln_actor_dataset.py`, which can combine:

- one or more summary files such as R2R and ScaleVLN;
- the corresponding image roots;
- optional watcher hint files;
- memory-layout settings such as history length, anchor counts, and sliding-window mode.

Actor records typically contain:

- `episode_key`
- `frame_idx`
- `subtask`
- `action_labels`
- image paths and history image paths
- optional `watcher_hint`
- optional `next_subtask`

Conceptually, the actor training data answers this question:

> Given the current observation, subtask, recent context, and optional watcher memory, what short low-level action chunk should the actor produce next?

### Watcher training data

The watcher training data is built from rollout bundles rather than directly from raw trajectory summaries.

Main sources in the checked-in configs:

- rollout bundle root:
  `results/watcher_rollout_train_full_v2`
- rollout manifest:
  `results/watcher_rollout_train_full_v2/manifest/watcher_rollout_manifest.jsonl`
- annotation file:
  `results/watcher_rollout_train_full_v2/watcher_openai_annotations.deploy.full.jsonl`
- base trajectory summary:
  `data/trajectory_data/R2R_back/summary_full.jsonl`

The watcher dataset loader combines:

- manifest rows produced by rollout generation;
- annotation rows produced by OpenAI-style or manual watcher labeling;
- summary metadata such as instruction and plan steps from `summary_full.jsonl`.

Each watcher sample is built around:

- `instruction`
- `plan_steps`
- `done_steps`
- `active_step`
- `pending_steps`
- `memory_start`
- `rollout_actions`
- `rollout_image_paths`
- target `memory_end`
- target `done`
- target `next_subtask`

Conceptually, the watcher training data answers this question:

> After observing this rollout segment, how should the system update memory, and is the current subtask ready to hand off?

## 6. Training pipeline

The current end-to-end training pipeline has two connected but separate tracks.

### Stage A: build trajectory summaries and subtask structure

Starting from raw trajectory or simulator outputs, the repo builds summary files and subtask labels.

Typical pieces:

- `scripts/generate_summary_full.py`
- `thinkvln/datagen/generation/subtask_split.py`
- `thinkvln/datagen/generation/subtask_determination.py`
- `scripts/run_subtask_summary_pipeline.sh`

This stage produces `summary_full.jsonl`-style files that are the base source for later actor and watcher workflows.

### Stage B: materialize actor training records

The next step is to transform summary trajectories into StreamVLN actor examples.

Typical entrypoint:

- `scripts/build_streamvln_actor_dataset.py`

This stage:

- chooses the underlying summary datasets such as R2R and ScaleVLN;
- resolves image paths;
- injects optional watcher hints;
- selects sparse history frames;
- writes a materialized actor dataset JSONL for efficient training.

### Stage C: train the actor

The actor is trained through the active StreamVLN path.

Typical entrypoints:

- `scripts/train_streamvln_actor.py`
- `scripts/train_streamvln_actor.sh`
- `config/streamvln_actor_train*.yaml`

The actor config defines:

- base model path, currently `model_weights/streamvln` in the checked-in StreamVLN actor config;
- LoRA settings;
- history length and future-step horizon;
- optional watcher memory ratio;
- training hyperparameters and output paths.

### Stage D: generate watcher rollouts

Once an actor is available, the system generates rollout segments specifically for watcher supervision.

Typical entrypoints:

- `thinkvln/datagen/generation/watcher_rollout_generation.py`
- `scripts/run_watcher_rollout_train_full.sh`

This stage runs the actor in Habitat, collects rollout spans around subtask pivots, stores rollout frames, and writes manifest rows for later annotation.

### Stage E: annotate watcher data

The rollout bundle is then annotated so that watcher targets become supervised learning data.

Typical entrypoints:

- `thinkvln/datagen/generation/watcher_openai_annotation.py`
- `thinkvln/datagen/generation/watcher_openai_annotation_deploy.py`
- `thinkvln/datagen/generation/watcher_manual_done_annotation.py`
- `thinkvln/datagen/generation/watcher_manual_switch_annotation.py`
- `scripts/run_watcher_openai_annotation*.sh`
- `scripts/run_watcher_manual_annotation.sh`
- `scripts/run_watcher_manual_switch_web.sh`

This stage produces target labels for:

- updated watcher memory;
- handoff decision (`done`);
- next actor-facing subtask text.

### Stage F: train the watcher

The watcher is then fine-tuned with the rollout-based SFT dataset.

Typical entrypoints:

- `thinkvln/engine/watcher_sft_trainer.py`
- `scripts/train_watcher.sh`
- `config/watcher_sft.yaml`
- `config/watcher_sft_fullv2_streamv2.yaml`

In the checked-in config, the watcher is trained from a local Qwen3-VL-8 checkpoint with LoRA enabled and rollout-image inputs sampled by `image_stride`.

## 7. Online cooperation pipeline

At runtime, the actor and watcher cooperate as a staged loop rather than a single monolithic policy.

### Initialization

At the start of an episode:

- the environment provides the instruction and the full plan;
- the watcher starts with an empty or initial memory;
- the active step is the first plan step;
- the actor receives the current subtask text plus the first observation.

### Fast actor loop

The actor runs for a short local segment:

- it predicts low-level actions from the current observation, current subtask, sparse history, and optional watcher hint;
- it executes those actions in the environment;
- the system accumulates rollout images and action traces for the current segment.

This is the fast control path and is meant to be called frequently.

### Slow watcher wakeup

After some number of steps, or when a progress threshold / wakeup rule is hit, the watcher is called:

- it receives the rollout segment, prior memory, and current plan state;
- it rewrites memory into a compact cumulative state;
- it decides whether the current active step has reached a natural handoff;
- it emits the next actor-facing subtask text.

### Handoff logic

If `done=false`:

- the active step remains the same;
- the watcher subtask text usually refines or recovers the current step;
- the actor continues on the same plan step but with updated watcher memory.

If `done=true`:

- the current active step moves into `done_steps`;
- the next pending step becomes the new active step;
- the actor continues with the promoted step and updated watcher memory.

This separation is the core system design:

- the actor handles local movement;
- the watcher handles long-horizon subtask state transitions and memory compression.

## 8. Evaluation

The current repo uses two evaluation styles that answer different questions.

### A. Subtask evaluation

Subtask evaluation measures how well the actor handles the current active step in closed loop.

Primary files:

- `thinkvln/eval/close_eval.py`
- `thinkvln/eval/close_eval_cli.py`
- `thinkvln/eval/close_eval_runner.py`
- `thinkvln/eval/close_eval_utils.py`
- `scripts/run_close_eval_subtask.sh`
- `scripts/summarize_subtask_success.py`

Important characteristics:

- `close_eval_cli.py` currently exposes `ladder_mode=subtask`;
- the evaluation requires `summary_full.jsonl`;
- the navigation model must implement `predict_action_with_progress_and_done(...)`;
- the output summary includes subtask success, progress error, and done-related metrics.

This evaluation is mainly about the actor’s ability to execute a subtask correctly under closed-loop control.

In other words, subtask evaluation answers:

> Can the current actor finish the right local step within budget, and are its progress / done signals aligned with the ground-truth subtask structure?

### B. Two-system evaluation

Two-system evaluation measures the joint behavior of actor and watcher as a coordinated hierarchical system.

Primary files:

- `thinkvln/eval/two_system_eval.py`
- `config/two_system_eval*.yaml`
- `thinkvln/tests/test_two_system_eval.py`

The config structure in `two_system_eval.py` makes the design explicit:

- `actor`: low-level policy and memory settings;
- `watcher`: backend, model path, and image stride;
- `env`: Habitat config and summary source;
- `rollout`: wakeup frequency, progress threshold, watcher budget, and episode cap;
- `output`: trace and summary artifact settings.

This evaluation asks a different question:

> Does the combined system manage handoffs correctly, preserve useful memory, and achieve better episode-level behavior than the actor alone?

Operationally, this path evaluates:

- actor execution quality inside each rollout span;
- watcher handoff quality;
- watcher memory usefulness;
- end-to-end episode success under repeated actor-watcher alternation.

## 9. Practical repo map

If someone needs to understand or extend the current system quickly, the shortest reading path is:

1. `docs/repo_handoff_guide.md`
2. `docs/model_design.md`
3. `streamvln/dataset/streamvln_actor_dataset.py`
4. `thinkvln/dataset/watcher_sft_dataset.py`
5. `scripts/build_streamvln_actor_dataset.py`
6. `scripts/train_streamvln_actor.py`
7. `scripts/run_watcher_rollout_train_full.sh`
8. `thinkvln/engine/watcher_sft_trainer.py`
9. `thinkvln/eval/close_eval_cli.py`
10. `thinkvln/eval/two_system_eval.py`

That sequence covers:

- how actor samples are built;
- how watcher samples are built;
- how training data is produced;
- how the actor and watcher are trained;
- how the two evaluation modes map onto the actual runtime system.
