"""Dataset utility functions for ThinkVLN."""

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """Load RGB image from path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert('RGB')


def crop_cot_answer(answer: str) -> str:
    """Extract reasoning from [causal observation] onwards."""
    marker = "[causal observation]"
    if marker in answer:
        return answer[answer.index(marker):]
    return answer


# Sentinel value in action_chunk indicating subtask boundary (next subtask).
# Distinct from STOP (0) which is true episode end.
NEXT_SUBTASK_SENTINEL = -1


def extract_action_chunk(
    frame_idx: int,
    actions: List[int],
    subtask_sequence: List[int],
    num_steps: int = 4,
    use_next_token: bool = False,
) -> Tuple[List[int], List[float]]:
    """
    Extract next N actions and progress values with subtask boundary handling.

    Returns (action_chunk, progress_chunk).
    - Episode end: action 0 (STOP).
    - Subtask boundary: NEXT_SUBTASK_SENTINEL (-1) when use_next_token=True,
      else 0 (backward-compatible STOP padding).
    """
    current_subtask = subtask_sequence[frame_idx]
    action_chunk = []
    progress_chunk = []

    subtask_frames = [i for i, s in enumerate(subtask_sequence) if s == current_subtask]
    subtask_start = min(subtask_frames)
    subtask_length = len(subtask_frames)

    hit_boundary = False
    for k in range(1, num_steps + 1):
        next_idx = frame_idx + k

        if hit_boundary:
            # After a subtask boundary or episode end, fill remaining slots
            action_chunk.append(NEXT_SUBTASK_SENTINEL if use_next_token else 0)
            progress_chunk.append(1.0)
        elif next_idx >= len(actions):
            action_chunk.append(0)  # real STOP at episode end
            progress_chunk.append(1.0)
            hit_boundary = True
        elif next_idx >= len(subtask_sequence) or subtask_sequence[next_idx] != current_subtask:
            # First step that crosses into next subtask
            action_chunk.append(NEXT_SUBTASK_SENTINEL if use_next_token else 0)
            progress_chunk.append(1.0)
            hit_boundary = True
        else:
            action_chunk.append(actions[next_idx])
            position = next_idx - subtask_start
            progress = position / (subtask_length - 1) if subtask_length > 1 else 1.0
            progress_chunk.append(progress)

    return action_chunk, progress_chunk


def get_subtask_start_frame(frame_idx: int, subtask_sequence: List[int]) -> int:
    """Return the first frame index of the current subtask segment."""
    if not subtask_sequence:
        return 0
    frame_idx = max(0, min(frame_idx, len(subtask_sequence) - 1))
    current = subtask_sequence[frame_idx]
    start = frame_idx
    while start > 0 and subtask_sequence[start - 1] == current:
        start -= 1
    return start


def get_subtask_end_frame(frame_idx: int, subtask_sequence: List[int]) -> int:
    """Return the last frame index of the current subtask segment."""
    if not subtask_sequence:
        return 0
    frame_idx = max(0, min(frame_idx, len(subtask_sequence) - 1))
    current = subtask_sequence[frame_idx]
    end = frame_idx
    max_idx = len(subtask_sequence) - 1
    while end < max_idx and subtask_sequence[end + 1] == current:
        end += 1
    return end


def compute_current_step_progress(frame_idx: int, subtask_sequence: List[int]) -> float:
    """
    Compute scalar progress at current frame within current subtask.

    Progress is normalized to [0, 1], and is explicitly 0.0 at subtask start.
    """
    if not subtask_sequence:
        return 0.0
    frame_idx = max(0, min(frame_idx, len(subtask_sequence) - 1))
    start = get_subtask_start_frame(frame_idx, subtask_sequence)
    end = get_subtask_end_frame(frame_idx, subtask_sequence)
    if frame_idx == start:
        return 0.0
    denom = max(1, end - start)
    return float((frame_idx - start) / denom)


def compute_previous_step_progress(frame_idx: int, subtask_sequence: List[int]) -> float:
    """
    Compute teacher-forced previous-step progress for current frame.

    Resets to 0.0 at episode start and subtask boundary.
    """
    if frame_idx <= 0 or not subtask_sequence:
        return 0.0
    frame_idx = max(0, min(frame_idx, len(subtask_sequence) - 1))
    prev_idx = frame_idx - 1
    if subtask_sequence[prev_idx] != subtask_sequence[frame_idx]:
        return 0.0
    return compute_current_step_progress(prev_idx, subtask_sequence)


def compute_done_label(progress: float, threshold: float = 0.85) -> float:
    """Binary done label from scalar progress."""
    return 1.0 if float(progress) > float(threshold) else 0.0


def select_memory_frame_indices(
    frame_idx: int,
    subtask_sequence: List[int],
    memory_num_history_images: int = 8,
) -> List[int]:
    """
    Select ordered history frame indices for visual memory.

    Strategy:
    - Current frame is excluded (history-only).
    - History anchors: first frame in episode, first frame in current subtask.
    - Fill remaining history with sparse-uniform samples from prior frames.
    - Enforce history cap; trim non-anchor history first.
    """
    if frame_idx <= 0:
        return []

    current_idx = int(frame_idx)
    history_cap = max(0, int(memory_num_history_images))
    max_history = history_cap
    if max_history == 0:
        return []

    subtask_start = get_subtask_start_frame(current_idx, subtask_sequence)

    anchor_candidates: List[int] = []
    if subtask_start < current_idx:
        anchor_candidates.append(subtask_start)
    if 0 < current_idx and 0 not in anchor_candidates:
        anchor_candidates.append(0)
    # Sort anchors chronologically but preserve uniqueness.
    anchors: List[int] = []
    for idx in sorted(anchor_candidates):
        if idx not in anchors:
            anchors.append(idx)

    if len(anchors) > max_history:
        # Impossible to keep all anchors under budget: keep earliest anchors first.
        anchors = anchors[:max_history]

    remaining = max_history - len(anchors)
    sparse: List[int] = []
    if remaining > 0:
        candidates = [i for i in range(current_idx) if i not in anchors]
        if candidates:
            take = min(remaining, len(candidates))
            if take > 0:
                pos = np.linspace(0, len(candidates) - 1, num=take, dtype=int).tolist()
                sparse = sorted({candidates[p] for p in pos})
                # Fill if de-duplicated by linspace rounding.
                if len(sparse) < take:
                    for idx in candidates:
                        if idx not in sparse:
                            sparse.append(idx)
                        if len(sparse) == take:
                            break
                sparse = sorted(sparse[:take])

    history = sorted(set(anchors + sparse))
    # Hard enforce in rare edge case.
    history = history[:max_history]
    return history


def select_sliding_window_with_anchor(
    frame_idx: int,
    subtask_sequence: List[int],
    num_memory_slots: int,
) -> Tuple[int, List[int]]:
    """
    Select anchor frame (subtask start) + sliding window of history frames.

    Returns (anchor_frame_idx, memory_frame_indices) where memory does NOT
    include the anchor or current frame. Anchor is always the subtask start.
    Window frames are uniformly sampled from (anchor+1, frame_idx).

    This replaces the pre_anchor/post_anchor layout with a simpler design:
    - 1 slot reserved for anchor
    - remaining slots filled with evenly-strided frames within current subtask
    """
    current_idx = int(frame_idx)
    anchor = get_subtask_start_frame(current_idx, subtask_sequence)
    window_slots = max(0, int(num_memory_slots) - 1)  # 1 slot for anchor

    candidates = list(range(anchor + 1, current_idx))
    if not candidates or window_slots <= 0:
        return anchor, []

    take = min(window_slots, len(candidates))
    positions = np.linspace(0, len(candidates) - 1, num=take, dtype=int).tolist()
    window: List[int] = []
    for p in positions:
        idx = int(candidates[int(p)])
        if idx not in window:
            window.append(idx)
    # Fill deduplication gaps
    if len(window) < take:
        for idx in candidates:
            if idx not in window:
                window.append(idx)
            if len(window) == take:
                break
    return anchor, sorted(window[:take])


def parse_frame_key(frame_key: str) -> Tuple[str, int]:
    """
    Parse frame_key to extract episode information.
    
    Args:
        frame_key: "{scene_id}_{episode_id}_{step_id:06d}" e.g., "17DRP5sb8fy_10154_000035"
    
    Returns:
        (episode_key, step_id) e.g., ("17DRP5sb8fy_10154", 35)
        
    Note: The episode_key format in the dataset is "{scene_id}_{episode_id}",
    without the "_r2r_" prefix that may be in the directory structure.
    """
    parts = frame_key.split('_')
    if len(parts) >= 3:
        scene_id = parts[0]
        episode_id = parts[1]
        step_id = int(parts[2])
        # episode_key matches the dataset format
        episode_key = f"{scene_id}_{episode_id}"
        return episode_key, step_id
    raise ValueError(f"Invalid frame_key format: {frame_key}")
