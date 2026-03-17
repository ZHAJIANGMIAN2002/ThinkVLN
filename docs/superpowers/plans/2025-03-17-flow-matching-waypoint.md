# Flow Matching Waypoint Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add flow matching waypoint prediction to ThinkVLN by creating new files (no modification of existing thinkvln_ar_model.py / thinkvln_actor.py), replacing the MLP action head with a flow matching head that conditions on LLM hidden state and generates waypoints from randomly initialized noise.

**Architecture:** A new `ThinkVLNFMActor` model extends ThinkVLNForConditionalGeneration, using the LLM hidden state (from action query tokens) as the condition for conditional flow matching. The flow matching head predicts velocity toward target waypoints; training uses MSE loss on predicted vs. target velocity. Waypoint data is produced by a new trajectory generation path that records agent positions per step and writes `summary_with_waypoints.jsonl`.

**Tech Stack:** PyTorch, Transformers (Qwen3VL), OmniNav FlowmatchingActionHead reference (DiT, ActionEncoder, ActionDecoder), Habitat simulator for position extraction.

---

## File Structure

| Responsibility | Create | Modify |
|----------------|--------|--------|
| Flow matching model | `thinkvln/models/thinkvln_fm_actor.py` | - |
| Flow matching config | `thinkvln/models/fm_actor_config.py` | - |
| OmniNav components (copy/adapt) | `thinkvln/models/fm_components.py` (DiT, ActionEncoder, ActionDecoder) | - |
| Waypoint dataset | `thinkvln/dataset/fm_waypoint_dataset.py` | - |
| Waypoint collator | Extend in `thinkvln/dataset/dataset.py` or new `fm_collator.py` | - |
| Trajectory gen with waypoints | `thinkvln/datagen/generation/trajectory_generation_waypoint.py` | - |
| Summary with waypoints | `scripts/generate_summary_with_waypoints.py` | - |
| SFT trainer for FM | `thinkvln/engine/sft_trainer_fm.py` | - |
| Model __init__ | - | `thinkvln/models/__init__.py` |

---

## Chunk 1: Data Pipeline (Waypoint Extraction & Summary)

### Task 1.1: Trajectory Generation with Position Recording

**Files:**
- Create: `thinkvln/datagen/generation/trajectory_generation_waypoint.py`
- Test: `thinkvln/tests/test_trajectory_generation_waypoint.py`

- [ ] **Step 1: Create trajectory generator that records positions**

Copy `trajectory_generation.py` → `trajectory_generation_waypoint.py`. In the rollout loop, after each `env.step(next_action)`:
- Call `env.sim.get_agent_state().position` and append to a `positions` list.
- Include `positions` in the result dict written to `summary_waypoint.json` (or a separate file).

```python
# In the rollout loop, after observation = env.step(next_action):
positions.append(env.sim.get_agent_state().position.copy())
# In result:
result = {
    ...
    "positions": [[float(p[0]), float(p[2])] for p in positions],  # x,z for 2D
}
```

- [ ] **Step 2: Write to summary_waypoint.json**

Append to `summary_waypoint.json` (or `summary_with_waypoints.json`) instead of/in addition to `summary.json` so downstream scripts can merge waypoints.

- [ ] **Step 3: Add minimal test**

```python
def test_trajectory_generation_waypoint_has_positions():
    # Mock or use small fixture; assert result has "positions" and len(positions) == len(actions)
```

---

### Task 1.2: Summary Merge Script with Waypoints

**Files:**
- Create: `scripts/generate_summary_with_waypoints.py`
- Test: `thinkvln/tests/test_generate_summary_waypoints.py`

- [ ] **Step 1: Implement merge logic**

Read `summary_waypoint.json` (or `summary_with_waypoints.json`), `split_file`, `determination_file`. For each episode:
- Compute per-frame delta waypoints: for frame `t`, `delta_waypoints[t]` = `positions[t+1:t+1+horizon] - positions[t]` (shape `[horizon, 2]` for x,z).
- Pad at trajectory end.
- Add `delta_waypoints` (or `waypoints`) to merged record.
- Write to `summary_full_waypoint.jsonl`.

- [ ] **Step 2: Add CLI**

```bash
python scripts/generate_summary_with_waypoints.py \
  --trajectory_dir ... --split_output_file ... --determination_output_file ... \
  --summary_full_waypoint_file ... --action_horizon 5
```

---

## Chunk 2: Flow Matching Model Components

### Task 2.1: Extract OmniNav Flow Matching Components

**Files:**
- Create: `thinkvln/models/fm_components.py`
- Test: `thinkvln/tests/test_fm_components.py`

- [ ] **Step 1: Copy and adapt OmniNav components**

From `third_party/OmniNav/.../modeling_qwen2_5_vl.py` extract into `fm_components.py`:
- `SinusoidalPositionalEncoding`
- `ActionEncoder` (action_dim, hidden_size)
- `ActionDecoder` (input_dim, hidden_dim, output_dim)
- `DiT` (or minimal DiT block) – ensure it accepts `hidden_states`, `encoder_hidden_states`, `timestep`.

Make them standalone (no Qwen2_5_VL imports). Use `nn.Module` and standard PyTorch.

- [ ] **Step 2: Add unit test**

```python
def test_action_encoder_decoder_shapes():
    enc = ActionEncoder(action_dim=2, hidden_size=64)
    dec = ActionDecoder(64, 64, 2)
    x = torch.randn(2, 5, 2)  # B, T, action_dim
    t = torch.randint(0, 100, (2,))
    out = enc(x, t)
    assert out.shape == (2, 5, 64)
    out2 = dec(out)
    assert out2.shape == (2, 5, 2)
```

---

### Task 2.2: Flow Matching Head and Config

**Files:**
- Create: `thinkvln/models/fm_actor_config.py`
- Create: `thinkvln/models/thinkvln_fm_actor.py`
- Test: `thinkvln/tests/test_thinkvln_fm_actor.py`

- [ ] **Step 1: Define FlowMatchingActorConfig**

```python
@dataclass
class FlowMatchingActorConfig:
    input_embedding_dim: int = 1536   # match hidden_size
    decoder_hidden_size: int = 1024
    action_dim: int = 2               # x,z for VLN
    action_horizon: int = 5
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 100
    num_inference_timesteps: int = 10
    add_pos_embed: bool = True
    max_seq_len: int = 1024
```

- [ ] **Step 2: Implement FlowMatchingActionHead**

Same interface as OmniNav:
- `sample_time(batch_size, device, dtype)` → t in [0,1)
- `get_action(backbone_output)` → denoised waypoints (inference)
- Forward for training: `vl_embeds`, `delta_waypoints` → MSE loss on predicted velocity vs `velocity = delta_waypoints - noise`.

- [ ] **Step 3: Implement ThinkVLNFMActor**

Extend `ThinkVLNForConditionalGeneration`:
- Add `FlowMatchingActionHead` instead of MLP action head.
- Use `action_hidden` (from query tokens, same as ThinkVLNActor) as `vl_embeds` for flow matching.
- Aggregate/pool if needed: e.g. mean over query tokens → `[batch, hidden_size]`.
- Forward: if `waypoint_labels` provided → flow matching loss; else inference → `get_action`.

- [ ] **Step 4: Add test**

```python
def test_thinkvln_fm_actor_forward():
    config = ThinkVLNConfig.from_pretrained("Qwen/Qwen3-VL-2B")
    config.num_query_tokens = 4
    model = ThinkVLNFMActor(config, fm_config=FlowMatchingActorConfig())
    # Dummy forward with waypoint_labels, assert loss is scalar
```

---

## Chunk 3: Dataset and Collator

### Task 3.1: Waypoint Dataset

**Files:**
- Create: `thinkvln/dataset/fm_waypoint_dataset.py`
- Test: `thinkvln/tests/test_fm_waypoint_dataset.py`

- [ ] **Step 1: Implement ThinkVLNFMWaypointDataset**

Load from `summary_full_waypoint.jsonl` (or equivalent). Each sample:
- `episode_key`, `frame_idx`, `instruction`, `plan`, `current_plan_step`, `subtask_sequence`
- `delta_waypoints`: `[action_horizon, action_dim]` for that frame.
- Image path from `image_root` + `episode_key` + `frame_idx`.

- [ ] **Step 2: Add extract_delta_waypoints utility**

```python
def extract_delta_waypoints(
    frame_idx: int,
    positions: List[List[float]],
    horizon: int = 5,
    action_dim: int = 2
) -> np.ndarray:
    """positions: list of [x,z]. Return delta from current to next horizon positions."""
```

- [ ] **Step 3: Test**

```python
def test_fm_waypoint_dataset_load():
    # Create temp jsonl with 1 episode, 3 frames, waypoints
    # Assert len(dataset)==3, sample has "delta_waypoints" shape (horizon, 2)
```

---

### Task 3.2: FM Collator

**Files:**
- Create: `thinkvln/dataset/fm_collator.py` (or extend existing collator)
- Test: `thinkvln/tests/test_fm_collator.py`

- [ ] **Step 1: Implement collator**

Similar to existing action collator but:
- Batch `delta_waypoints` → `[B, horizon, action_dim]` tensor.
- Build prompt + image input for model (reuse existing message format).
- Return `waypoint_labels` for loss computation.

- [ ] **Step 2: Test**

```python
def test_fm_collator_batch():
    samples = [{"delta_waypoints": np.zeros((5,2)), ...}, ...]
    batch = fm_collator(samples)
    assert "waypoint_labels" in batch
    assert batch["waypoint_labels"].shape == (len(samples), 5, 2)
```

---

## Chunk 4: Training

### Task 4.1: SFT Trainer for Flow Matching

**Files:**
- Create: `thinkvln/engine/sft_trainer_fm.py`
- Test: `thinkvln/tests/test_sft_trainer_fm.py` (optional, smoke test)

- [ ] **Step 1: Implement SFT trainer**

Copy `sft_trainer.py` → `sft_trainer_fm.py`:
- Use `ThinkVLNFMWaypointDataset` and FM collator.
- Use `ThinkVLNFMActor` instead of `ThinkVLNActor`.
- In training step: pass `waypoint_labels` to model, use returned flow matching loss.
- No action/progress/done loss; only flow matching MSE loss.

- [ ] **Step 2: Add config**

```yaml
# config/sft_training_fm.yaml
model_name_or_path: "Qwen/Qwen3-VL-2B"
waypoint_data_path: "data/summary_full_waypoint.jsonl"
action_horizon: 5
action_dim: 2
```

- [ ] **Step 3: Smoke test**

```bash
python thinkvln/engine/sft_trainer_fm.py --config config/sft_training_fm.yaml --max_steps 2
```

---

## Chunk 5: Integration and Registration

### Task 5.1: Model Registration

**Files:**
- Modify: `thinkvln/models/__init__.py`

- [ ] **Step 1: Export new classes**

```python
from .thinkvln_fm_actor import ThinkVLNFMActor
from .fm_actor_config import FlowMatchingActorConfig
# Add to __all__
```

---

## Verification Checklist

- [ ] Trajectory generation produces `summary_waypoint.json` with `positions` per episode
- [ ] `generate_summary_with_waypoints.py` produces `summary_full_waypoint.jsonl` with `delta_waypoints`
- [ ] `ThinkVLNFMActor` forward returns loss when `waypoint_labels` provided
- [ ] `ThinkVLNFMActor.get_action()` returns waypoints of shape `[B, horizon, action_dim]`
- [ ] `sft_trainer_fm.py` runs for 2 steps without error
- [ ] No changes to `thinkvln_ar_model.py` or `thinkvln_actor.py`

---

## Reference: OmniNav Flow Matching (Condensed)

- **Condition:** `vl_embeds` = backbone hidden state (e.g. last token or pooled).
- **Target:** `delta_waypoints` shape `[B, action_horizon, action_dim]`.
- **Flow:** `t ~ Beta(alpha, beta)`, `noisy = (1-t)*noise + t*target`, `velocity = target - noise`.
- **Model:** DiT with `encoder_hidden_states=vl_embeds`, predicts velocity.
- **Loss:** `MSE(pred_velocity, velocity)`.

---

## Notes

- **thinkvln_ar_model.py**: This file is a thin loader for AR models; the actual actor with action heads is `thinkvln_actor.py`. The plan creates a parallel `thinkvln_fm_actor.py` without modifying either.
- **Navigation model**: The existing `ThinkVLNActorNavigationModel` handles memory and evaluation. The FM actor can later be wrapped in a similar navigation model for eval; that is out of scope for this plan.
- **Waypoint format**: Using 2D (x,z) from Habitat; OmniNav uses 5-dim (possibly x,y,theta + extras). Adjust `action_dim` if needed.
