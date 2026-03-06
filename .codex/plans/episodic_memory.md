# ThinkVLN Actor: Episodic Memory + Sequential Progress/Done (Minimal-Model-Change Plan)

## Summary
Implement StreamVLN-style memory context for ThinkVLN Actor with minimal model changes, keep **4-step action prediction**, switch progress supervision to **current-step scalar**, inject **previous-step progress** into prompts, and add a new **binary done prediction head** (`progress_gt > 0.85`).  
Training will become **episodic sequential** for action data: episodes are sampled randomly, and frames inside each episode are consumed in order (contiguous batches from one episode).

## Public Interfaces / Behavior Changes
1. [thinkvln/models/thinkvln_actor.py](/mnt/swx/ThinkVLN/thinkvln/models/thinkvln_actor.py): `forward(...)` will accept `done_labels` in action mode and return `done_logits`, `done_preds`, `done_loss` in output dict.
2. [thinkvln/dataset/dataset.py](/mnt/swx/ThinkVLN/thinkvln/dataset/dataset.py): action collator output will change from `progress_labels: [B,4]` to `progress_labels: [B]`, and add `done_labels: [B]`.
3. [thinkvln/models/navigation_model.py](/mnt/swx/ThinkVLN/thinkvln/models/navigation_model.py): actor wrapper will maintain episode memory state and previous progress; `predict_action_with_progress` will expose done prediction (either as tuple extension or new companion method while preserving current call sites).
4. [thinkvln/engine/sft_trainer.py](/mnt/swx/ThinkVLN/thinkvln/engine/sft_trainer.py): training args/config wiring will add episodic action training + memory budget controls + done loss weight.
5. [docs/streamvln_memory_thinkvln_actor.md](/mnt/swx/ThinkVLN/docs/streamvln_memory_thinkvln_actor.md): new brief research doc summarizing StreamVLN memory and exact ThinkVLN adaptation.

## Implementation Plan

1. Add the brief research doc first.
File: [docs/streamvln_memory_thinkvln_actor.md](/mnt/swx/ThinkVLN/docs/streamvln_memory_thinkvln_actor.md).  
Content will cover StreamVLN memory mechanics (`<memory>` placeholder, sparse history sampling, history feature injection), and the ThinkVLN adaptation rules: episode anchor frame, subtask anchor frame, sparse uniform history, and image-token budget trimming.

2. Extend dataset utilities for sequential scalar progress/done labels.
File: [thinkvln/tools/dataset_utils.py](/mnt/swx/ThinkVLN/thinkvln/tools/dataset_utils.py).  
Add utilities to compute:
- current-step progress in `[0,1]` from `subtask_sequence` at `frame_idx` (subtask start must be `0.0`);
- previous-step progress (teacher-forced from raw data, reset to `0.0` at subtask boundary);
- done label from raw progress with strict threshold rule `done = (progress > 0.85)`.

3. Implement StreamVLN-style memory frame selection in the action collator.
File: [thinkvln/dataset/dataset.py](/mnt/swx/ThinkVLN/thinkvln/dataset/dataset.py).  
Changes:
- Keep action labels as next 4 actions (existing behavior).
- Replace 4-value progress target with current-step scalar target.
- Add `done_labels` scalar.
- Build action prompt with previous progress text and memory marker:
  - include `<memory>` text only when history exists;
  - include previous progress number in prompt text.
- Memory selection policy per sample:
  - always include episode first frame;
  - always include current subtask first frame;
  - fill remaining memory slots by sparse-uniform sampling over prior frames;
  - always include current frame as observation frame;
  - enforce **image-token budget parameter** by trimming non-anchor sparse history first (anchors kept unless impossible).
- Add collator config params:
  - `memory_num_history_images` (cap on memory images before current frame),
  - `memory_image_token_budget` (hard cap for total visual tokens),
  - `done_threshold` (default `0.85`).

4. Add episodic sequential sampler for action training.
File: [thinkvln/dataset/dataset.py](/mnt/swx/ThinkVLN/thinkvln/dataset/dataset.py).  
Add a batch sampler that:
- groups action samples by `episode_key`,
- shuffles episode order each epoch,
- emits contiguous frame indices in ascending order within each episode,
- yields contiguous multi-step batches from one episode (`batch_size` preserved),
- supports `drop_last`.
CoT path remains unchanged; if episodic mode is on with non-empty CoT data, enforce clear behavior (error or explicit fallback) to avoid silent mixed semantics.

5. Add done head and scalar progress path in actor model with minimal structural change.
File: [thinkvln/models/thinkvln_actor.py](/mnt/swx/ThinkVLN/thinkvln/models/thinkvln_actor.py).  
Changes:
- Keep 4-step action logits unchanged.
- Keep query-token layout mostly unchanged; use one progress-query stream position for current-step scalar progress.
- Add lightweight binary `done` head from the same projected progress feature.
- Compute losses in action mode:
  - `action_loss` (unchanged),
  - `progress_loss` on scalar progress target,
  - `done_loss` with BCE-with-logits on bool label.
- Combine weighted losses via actor config weights.
- Emit output keys for done so trainer/eval can log metrics.

6. Wire trainer and evaluation for episodic + done.
Files:
- [thinkvln/engine/sft_trainer.py](/mnt/swx/ThinkVLN/thinkvln/engine/sft_trainer.py)
- [thinkvln/engine/evaluate.py](/mnt/swx/ThinkVLN/thinkvln/engine/evaluate.py)
- [thinkvln/models/actor_config.py](/mnt/swx/ThinkVLN/thinkvln/models/actor_config.py)

Changes:
- Add config/arg fields for `done_loss_weight`, episodic action mode, memory history cap, image-token budget, done threshold.
- Use episodic batch sampler in train dataloader when action episodic mode is enabled.
- Update logging/eval aggregation to handle scalar progress and add done accuracy metric.
- Include `done_head` in LoRA `modules_to_save` and checkpoint verification expectations.

7. Update actor inference wrapper to be sequential and stateful.
File: [thinkvln/models/navigation_model.py](/mnt/swx/ThinkVLN/thinkvln/models/navigation_model.py).  
Changes:
- Maintain per-episode history buffer for memory images.
- Maintain per-subtask previous progress state (reset to `0.0` when subgoal changes).
- Reuse the same memory selection + token-budget logic as training.
- Prompt includes previous progress number and memory marker.
- Surface done prediction from model (bool from done head sigmoid).
- Keep close-loop compatibility by updating call sites in:
  - [thinkvln/eval/close_eval_runner.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval_runner.py)
  - [thinkvln/eval/close_eval_cli.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval_cli.py)
  so episode state resets are explicit and memory accumulates per episode.

8. Update default config and script wiring.
Files:
- [config/sft_training.yaml](/mnt/swx/ThinkVLN/config/sft_training.yaml)
- [config/eval_config.yaml](/mnt/swx/ThinkVLN/config/eval_config.yaml)
- [scripts/run_close_eval_subtask.sh](/mnt/swx/ThinkVLN/scripts/run_close_eval_subtask.sh) (if needed for new args).  
Add new parameters with conservative defaults:
- `done_threshold: 0.85`,
- `memory_num_history_images: 8`,
- `memory_image_token_budget` (explicit numeric cap),
- episodic action mode enabled.

## Test Cases and Verification Scenarios
1. Dataset utility tests for scalar progress correctness:
- subtask start frame gives `progress=0.0`,
- intra-subtask monotonic progress,
- subtask boundary resets previous progress to `0.0`,
- done label is true iff `progress > 0.85`.

2. Collator memory tests:
- selected memory always includes episode anchor and subtask anchor when available,
- sparse frames are selected from past,
- budget trimming removes non-anchor sparse frames first,
- final tensor outputs include `action_labels [B,4]`, `progress_labels [B]`, `done_labels [B]`.

3. Episodic sampler tests:
- indices within each batch are from one episode and contiguous,
- episode order changes by epoch seed,
- no frame-order violations inside episode.

4. Actor forward tests:
- output shapes: `action_logits [B,4,C]`, `progress_preds [B]`, `done_logits/done_preds [B]`,
- `done_loss` present when `done_labels` provided,
- backward compatibility path (no CoT regression).

5. Close-loop wrapper tests:
- previous progress is injected and updated step-to-step,
- subgoal switch resets previous progress to `0.0`,
- done prediction returned and clipped/typed correctly,
- no crash when memory history is empty.

6. Targeted pytest run:
- `python3 -m pytest thinkvln/tests/test_dataset.py`
- `python3 -m pytest thinkvln/tests/test_actor.py`
- `python3 -m pytest thinkvln/tests/test_close_eval_ladder.py`
- plus one focused smoke on real-data path if available.

## Assumptions and Defaults Locked
1. Keep 4-step action prediction unchanged.
2. Progress supervision is current-step scalar only.
3. Previous progress input during training uses teacher forcing from raw data.
4. Memory is visual and uses `<memory>` marker text; no heavy custom memory-token embedding surgery in the base model.
5. Memory spans the whole episode; subtask switch resets previous-progress state but not episode history.
6. Memory sampling rule is fixed to: episode-first anchor + subtask-first anchor + sparse-uniform past, with explicit image-token budget control.
7. Done is a separate binary head target, not derived from predicted progress at training time.
8. Done ground truth is computed from raw progress using `progress > 0.85`.
