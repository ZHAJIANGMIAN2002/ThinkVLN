# navigation_model.py (Brief)

This module defines a unified navigation interface plus concrete wrappers for ThinkVLN and StreamVLN inference.

## Interface contract

- `NavigationModel` is an abstract base class with two required methods:
- `predict_action(observation, instruction, plan=None, prev_subtask=None, **kwargs) -> (action_id, prev_subtask)`
- `eval()` to switch model to evaluation mode.
- Action ids follow evaluator convention:
- `0=stop`, `1=forward`, `2=turn_left`, `3=turn_right`.

## Class graph

```mermaid
classDiagram
    class NavigationModel {
        <<abstract>>
        +predict_action(...)
        +eval()
    }

    class ThinkVLNNavigationModel {
        +model
        +processor
        +device
        +max_new_tokens
        +predict_action(...)
        +eval()
        -_build_user_message(...)
        -_parse_action(...)
        -_extract_prev_subtask(...)
    }

    class StreamVLNNavigationModel {
        +model
        +tokenizer
        +device
        +num_frames
        +num_future_steps
        +num_history
        +env_id
        +predict_action(...)
        +eval()
        -_preprocess_depth_image(...)
        -_get_intrinsic_matrix(...)
        -_preprocess_intrinsic(...)
        -_xyz_yaw_to_tf_matrix(...)
        -_parse_actions(...)
        -_preprocess_qwen(...)
    }

    NavigationModel <|-- ThinkVLNNavigationModel
    NavigationModel <|-- StreamVLNNavigationModel
```

## ThinkVLNNavigationModel

- Input: PIL image observation (typically `RGB + map`) and text instruction.
- Builds a message with `<image>`, instruction, plan, and optional previous subtask.
- Runs inference via `run_batch_inference(...)`.
- Parses model text for `[action]` and maps text to action id.
- Extracts `[cur subtask]` when available and returns it for the next step.
- Safe fallback on errors or parse miss: `stop`.

## StreamVLNNavigationModel

- Input: observation dict with `rgb`, `depth`, `gps`, `compass`, `env`, and camera/depth metadata.
- Maintains streaming state:
- frame/depth/pose/intrinsic history, `time_ids`, queued `action_seq`, `past_key_values`, `output_ids`.
- Preprocesses:
- depth scaling/filtering, RGB/depth resize, intrinsics adjustment, pose transform.
- Builds Qwen-style tokenized prompt (with `<video>`, `<image>`, `<memory>` handling).
- Uses current frame + optional sampled history, then calls `model.generate(...)`.
- Parses generated symbols (`STOP`, `↑`, `←`, `→`) into action ids.
- Resets cache/history periodically (`step_count % num_frames == 0`) via `reset_for_env`.
- On any failure, defaults to `STOP`.

## Predict-action flow graph

```mermaid
flowchart TD
    A[predict_action call] --> B{wrapper type}

    B -->|ThinkVLN| C[build prompt with image + instruction + plan]
    C --> D[run_batch_inference]
    D --> E[parse action text]
    E --> F[extract cur subtask]
    F --> G[return action_id, prev_subtask]

    B -->|StreamVLN| H{action_seq cached?}
    H -->|yes| I[pop next action]
    I --> J[return action_id, None]
    H -->|no| K[preprocess rgb/depth/pose/intrinsics]
    K --> L[update history + build tokenized input]
    L --> M[model.generate]
    M --> N[parse STOP/↑/←/→ into action ids]
    N --> O{periodic reset}
    O -->|if needed| P[reset_for_env + clear cache/history]
    O -->|else| Q[keep state]
    P --> R[pop first action]
    Q --> R
    R --> S[return action_id, None]
```

## Why this module exists

- Keeps evaluator code model-agnostic.
- Concentrates model-specific prompt, preprocessing, and parsing logic in one place.
- Makes it easier to add new models by implementing the same `NavigationModel` API.

## Adding a new model wrapper

1. Subclass `NavigationModel`.
2. Implement `eval()` and `predict_action(...)`.
3. Return `(int_action_id, optional_prev_subtask)` every step.
4. Keep robust fallback behavior (`stop`) for bad outputs or runtime errors.
