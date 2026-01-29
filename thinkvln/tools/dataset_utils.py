"""Dataset utility functions for ThinkVLN"""

from PIL import Image
from typing import List, Tuple
import os


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


def extract_action_chunk(
    frame_idx: int,
    actions: List[int],
    subtask_sequence: List[int],
    num_steps: int = 4
) -> Tuple[List[int], List[float]]:
    """
    Extract next N actions and progress values with subtask boundary handling.
    
    Returns (action_chunk, progress_chunk) where actions are padded with 0 (stop)
    at subtask boundaries and progress is within-subtask progress [0.0-1.0].
    """
    current_subtask = subtask_sequence[frame_idx]
    action_chunk = []
    progress_chunk = []
    
    # Get current subtask bounds
    subtask_frames = [i for i, s in enumerate(subtask_sequence) if s == current_subtask]
    subtask_start = min(subtask_frames)
    subtask_length = len(subtask_frames)
    
    for k in range(1, num_steps + 1):
        next_idx = frame_idx + k
        
        if next_idx >= len(actions):
            # Beyond trajectory end
            action_chunk.append(0)
            progress_chunk.append(1.0)
        elif next_idx >= len(subtask_sequence) or subtask_sequence[next_idx] != current_subtask:
            # Crossed subtask boundary
            action_chunk.append(0)
            progress_chunk.append(1.0)
        else:
            # Within same subtask
            action_chunk.append(actions[next_idx])
            position = next_idx - subtask_start
            progress = position / (subtask_length - 1) if subtask_length > 1 else 1.0
            progress_chunk.append(progress)
    
    return action_chunk, progress_chunk


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
