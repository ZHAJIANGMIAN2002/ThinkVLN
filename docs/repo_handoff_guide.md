# ThinkVLN Handoff Guide

This document is a practical map of the repository for new collaborators. It focuses on what each major folder is for, where the main entry points are, which parts are third-party, and how the current two-system design fits into related VLA/VLN work.

## 1. Repository at a glance

The repo currently has three layers:

1. `thinkvln/`: the main active codebase.
2. `streamvln/`: an active StreamVLN-based actor baseline, fine-tuned with subtask-level data.
3. `third_party/`: external code snapshots used for reference, dependency patching, or comparison.

In practice, most current work should start from `thinkvln/`, `config/`, and `scripts/`.

## 2. Top-level folders

### Core code

| Folder | Role | Key files / notes |
| --- | --- | --- |
| `thinkvln/` | Main codebase for actor training, watcher training, closed-loop eval, datagen, and utilities. | Start here for most development. |
| `streamvln/` | Active StreamVLN actor baseline codepath, including subtask-level training/eval and shared prompt/history logic. | Still imported by some ThinkVLN code. |
| `scripts/` | Convenience launchers, dataset builders, rollout pipelines, deploy helpers, and smoke scripts. | Good place to find “how this was actually run”. |
| `config/` | YAML configs for actor training, watcher training, eval, AR/FM experiments, and two-system evaluation. | Best place to see current experiment settings. |

### Data, weights, and outputs

| Folder | Role | Notes |
| --- | --- | --- |
| `data/` | Datasets and derived training data. | Includes trajectory data, CoT data, Habitat datasets, watcher rollout data. |
| `model_weights/` | Local base-model checkpoints. | Contains Qwen3-VL and other local weights. |
| `checkpoints/` | Saved model checkpoints from experiments. | Includes StreamVLN LoRA checkpoints and other saved runs. |
| `outputs/` | Main training outputs. | Actor, watcher, LoRA, and other run outputs. |
| `results/` | Evaluation artifacts, debug outputs, rollout manifests, annotation outputs, and summaries. | Important when tracing watcher datagen or eval outputs. |


### Docs and misc

| Folder | Role | Notes |
| --- | --- | --- |
| `docs/` | Internal notes and design writeups. | Good repo-specific context; not always fully up to date. |
| `llava/` | Local LLaVA-related code tree. | Supports model and training compatibility work. |
| `trl/` | Local TRL code tree. | Local trainer support code used by some training workflows. |

## 3. Main active package: `thinkvln/`

### `thinkvln/models/`

This folder defines the main model wrappers and architectures.

Important files:

- `thinkvln/models/thinkvln_model.py`: base ThinkVLN model.
- `thinkvln/models/thinkvln_actor.py`: actor model with action / progress / done heads.
- `thinkvln/models/navigation_model.py`: unified navigation wrapper interface used by evaluation code.
- `thinkvln/models/thinkvln_ar_model.py`: autoregressive variant.
- `thinkvln/models/thinkvln_fm_actor.py`, `fm_components.py`, `fm_actor_config.py`: flow-matching / waypoint-related path.
- `thinkvln/models/actor_config.py`, `thinkvln_config.py`: model configuration helpers.

Most closed-loop navigation behavior is consumed through `navigation_model.py`, not by calling raw model classes directly.

### `thinkvln/engine/`

This is where training and model construction entry points live.

Important files:

- `thinkvln/engine/sft_trainer.py`: main actor SFT trainer.
- `thinkvln/engine/watcher_sft_trainer.py`: watcher SFT trainer.
- `thinkvln/engine/ar_trainer.py`: AR training path.
- `thinkvln/engine/sft_trainer_fm.py`: FM actor training path.
- `thinkvln/engine/inference.py`: loading and inference utilities.

If someone wants to understand how training arguments are flattened from YAML into Hugging Face `TrainingArguments`, `sft_trainer.py` and `watcher_sft_trainer.py` are the first files to read.

### `thinkvln/eval/`

This folder contains the closed-loop evaluation stack.

Important files:

- `thinkvln/eval/close_eval.py`: public entry facade.
- `thinkvln/eval/close_eval_cli.py`: argument parsing and eval orchestration.
- `thinkvln/eval/close_eval_runner.py`: closed-loop evaluator runtime.
- `thinkvln/eval/close_eval_models.py`: model builder for eval.
- `thinkvln/eval/two_system_eval.py`: current actor + watcher two-system evaluation pipeline.
- `thinkvln/eval/streamvln_debug_eval.py`: debug-oriented StreamVLN-style eval path.

For the current architecture handoff, `two_system_eval.py` is the single most important file in this folder.

### `thinkvln/dataset/`

This folder defines the dataset contracts used for training and some evaluation-time prompt building.

Important files:

- `thinkvln/dataset/dataset.py`: mixed action/CoT dataset for actor training.
- `thinkvln/dataset/watcher_sft_dataset.py`: watcher rollout dataset and watcher prompt format.
- `thinkvln/dataset/ar_dataset.py`: AR dataset.
- `thinkvln/dataset/fm_waypoint_dataset.py`: FM / waypoint dataset path.

Two especially important details:

- `dataset.py` explains the actor-side sample structure: instruction, current plan step, action labels, and optional CoT samples.
- `watcher_sft_dataset.py` defines the watcher prompt, output JSON format, and how rollout spans are converted into watcher training samples.

### `thinkvln/datagen/`

This folder is for offline data construction and annotation pipelines.

Typical responsibilities:

- generate chain-of-thought data
- generate subtask labels or summaries
- process rollout traces
- support watcher annotation workflows

This folder matters whenever a result in `results/` or `data/` was produced by an offline generation pipeline rather than by model training directly.

### `thinkvln/tools/`

Utility code that supports data loading, profiling, frame extraction, local apps, and development workflows.

Important files:

- `thinkvln/tools/dataset_utils.py`: shared helpers for prompt memory selection, image loading, progress labels, and sample processing.
- `thinkvln/tools/thinkvln_actor_latency_profile.py`: latency profiling helper.

### `thinkvln/tests/`

Main regression coverage for training scripts, datasets, closed-loop eval, watcher tools, and utility scripts.

Useful tests for orientation:

- `thinkvln/tests/test_two_system_eval.py`
- `thinkvln/tests/test_watcher_sft.py`
- `thinkvln/tests/test_dataset.py`
- `thinkvln/tests/test_streamvln_navigation_model.py`
- `thinkvln/tests/test_close_eval_*.py`

These tests are a good shortcut for learning expected interfaces without reading every implementation file.

### `thinkvln/habitat_extensions/`

Custom Habitat measures and map helpers.

Important files:

- `thinkvln/habitat_extensions/measures.py`
- `thinkvln/habitat_extensions/maps.py`

Read this only when evaluation metrics or simulator-side behavior need to change.

## 4. StreamVLN actor baseline: `streamvln/`

`streamvln/` is an active actor baseline in this repo. It is fine-tuned with subtask-level supervision and also provides prompt/history logic that is still reused by parts of the ThinkVLN stack.

Important files:

- `streamvln/streamvln_train.py`: main baseline training entry.
- `streamvln/streamvln_eval.py`: baseline evaluation.
- `streamvln/streamvln_openloop_eval.py`: open-loop evaluation.
- `streamvln/streamvln_agent.py`: agent wrapper.
- `streamvln/model/stream_video_vln.py`: core model path.
- `streamvln/dataset/streamvln_actor_dataset.py`: actor prompt/data logic for subtask-level finetuning.
- `streamvln/dataset/streamvln_actor_history_layout.py`: history/memory layout logic.

One concrete example: `thinkvln/models/navigation_model.py` imports StreamVLN prompt constants and history helpers. So changes in `streamvln/` can still affect ThinkVLN inference behavior.

## 5. Configs and scripts: how runs are actually controlled

### `config/`

This folder is the best summary of what experiments exist.

High-value configs:

- `config/sft_training.yaml`: main ThinkVLN actor training config.
- `config/watcher_sft_fullv2_streamv2.yaml`: watcher SFT training config.
- `config/two_system_eval.yaml`: actor + watcher closed-loop evaluation config.
- `config/vln_r2r.yaml`: Habitat / VLN environment config.
- `config/streamvln_actor_train*.yaml`: StreamVLN actor experiments.
- `config/ar_training.yaml`, `config/sft_training_fm.yaml`: alternate training tracks.

### `scripts/`

This folder shows the operational workflows that were actually used.

High-value scripts:

- `scripts/train_thinkvln_actor.sh`: actor training launcher.
- `scripts/train_watcher.sh`: watcher training launcher.
- `scripts/train_streamvln_actor.sh`: StreamVLN actor baseline launcher.
- `scripts/eval_thinkvln_actor.sh`: actor eval helper.
- `scripts/run_watcher_rollout_train_full.sh`: watcher rollout generation pipeline.
- `scripts/run_watcher_openai_annotation.sh`: watcher annotation pipeline.
- `scripts/build_streamvln_actor_dataset.py`: StreamVLN actor dataset builder.
- `scripts/generate_summary_full.py`: summary file generation.
- `scripts/summarize_subtask_success.py`: summarize subtask-level evaluation.

If a coworker wants to reproduce a run, `scripts/` plus `config/` is usually faster to inspect than reading package code first.

## 6. Third-party code

Everything under `third_party/` should be treated as external code unless there is a clear local patching reason.

Current third-party folders:

| Folder | What it is for |
| --- | --- |
| `third_party/LLaVA-NeXT/` | External LLaVA-NeXT codebase. |
| `third_party/OmniNav/` | External navigation-related codebase. |
| `third_party/Uni-NaVid/` | External VLN/navigation baseline code. |
| `third_party/habitat-lab/` | Habitat-Lab source snapshot. |
| `third_party/openvla-oft/` | OpenVLA-related external code. |
| `third_party/qwen3_vl/` | Local Qwen3-VL model implementation files. |
| `third_party/streamvln_base/` | External or base StreamVLN reference tree. |
| `third_party/thinkvln_back/` | Old backup/reference code path. |

Rule of thumb:

- modify `thinkvln/` or `streamvln/` first,
- touch `third_party/` only if the change is explicitly meant to patch vendored code.

## 7. Current system design

### Short version

The current architecture is best understood as a two-system navigation loop:

1. an **actor** predicts low-level navigation actions from the current observation plus compressed trajectory context;
2. a **watcher** periodically reviews the rollout span, updates memory, and decides whether the current subtask is complete and what the next subtask should be.

This is more structured than a single end-to-end policy, but lighter-weight than a full hierarchical planner + controller stack.

### Actor path

The actor is the fast loop.

Its main job is:

- read current image/history/instruction,
- condition on the current subtask or plan step,
- predict navigation action,
- optionally predict progress / done signals.

Relevant files:

- `thinkvln/models/thinkvln_actor.py`
- `thinkvln/models/navigation_model.py`
- `thinkvln/engine/sft_trainer.py`
- `thinkvln/dataset/dataset.py`
- `config/sft_training.yaml`

### Watcher path

The watcher is the slower supervisory loop.

Its main job is:

- inspect a rollout chunk,
- compress/update memory,
- judge whether the active subtask is finished,
- emit the next subtask instruction if a handoff is needed.

Relevant files:

- `thinkvln/dataset/watcher_sft_dataset.py`
- `thinkvln/engine/watcher_sft_trainer.py`
- `scripts/run_watcher_rollout_train_full.sh`
- `scripts/run_watcher_openai_annotation.sh`
- `config/watcher_sft_fullv2_streamv2.yaml`

### Where the two are connected

The integration point is:

- `thinkvln/eval/two_system_eval.py`

This file does the following:

- loads actor and watcher config blocks,
- runs the actor for short rollout windows,
- sends rollout frames/actions plus current plan state to the watcher,
- gets back updated memory, `done`, and `next_subtask`,
- continues until success, stop, or step budget termination.

This file is the best executable reference for the current research idea.

### Why this split exists

This split addresses a common long-horizon embodied problem:

- the low-level controller needs to react quickly,
- but the long-horizon state update and subtask handoff should not happen every step,
- and naive end-to-end memory often drifts or becomes too expensive.

The watcher acts as a sparse decision-maker and memory compressor, while the actor handles dense control.

## 8. Key data artifacts

These filenames show up repeatedly across training and evaluation:

- `summary_full.jsonl`: central trajectory summary file used by actor training, eval metadata, and watcher pipelines.
- watcher rollout manifests under `results/.../manifest/`: span-level rollout sample descriptions for watcher training.
- watcher annotation JSONL files under `results/...`: labels and memory updates for watcher supervision.

If someone is confused about “what data drives this run,” they should check the referenced paths inside the YAML config first.

## 9. Practical reading order for a new coworker

Recommended order:

1. `README.md`
2. `config/sft_training.yaml`
3. `config/watcher_sft_fullv2_streamv2.yaml`
4. `config/two_system_eval.yaml`
5. `thinkvln/models/navigation_model.py`
6. `thinkvln/eval/two_system_eval.py`
7. `thinkvln/dataset/dataset.py`
8. `thinkvln/dataset/watcher_sft_dataset.py`
9. `scripts/train_thinkvln_actor.sh`
10. `scripts/run_watcher_rollout_train_full.sh`

That sequence gives a much faster understanding than reading the whole repo in package order.

## 10. External references for the architecture

These are not exact copies of ThinkVLN, but they are useful mental models for understanding why the repo is shaped this way.

### 1. pi0 and pi0.5 / generalist policy

- Physical Intelligence, “π0: Our First Generalist Policy”  
  https://www.physicalintelligence.company/blog/pi0

Why it is relevant:

- shows the end-to-end VLA framing where a large policy maps multimodal context to robot actions;
- useful as a contrast point, because ThinkVLN adds more explicit subtask/memory structure on top of a policy model.

### 2. MEM: Multi-Scale Embodied Memory for VLA

- MEM: Multi-Scale Embodied Memory for Vision Language Action Models  
  https://arxiv.org/abs/2603.03596

Why it is relevant:

- directly addresses long-horizon memory in embodied policies;
- the watcher-side compressed memory in this repo is conceptually similar in spirit, even though the implementation here is task-specific and simpler.


### 3. R2R benchmark

- Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments  
  https://arxiv.org/abs/1711.07280

Why it is relevant:

- this is the core benchmark lineage behind the repo’s navigation framing;
- helpful for coworkers who need the original task definition and evaluation assumptions.


### 4. More
1. StreamVLN: opensourced strong baseline for single VLN model. For the lastest SOTA, reference Omninav.
2. JanusVLN: sliding window memory.
3. AtomicVLA: MoE mechanism for atomic skills.
4. CycleVLA: progress estimation and VLM assistant subtask switching.
5. VLingNav: linguistic memory design for VLN.
6. ECoT: embodied chain of thought for step by step reasoning.
