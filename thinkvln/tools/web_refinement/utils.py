import os
import json
import cv2
import base64
from typing import List, Dict, Optional
from pathlib import Path


def get_episode_key(scene_id: Optional[str], episode_id: int) -> str:
    """Generate unique episode key from scene_id and episode_id"""
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)


def interpolate_subtask_sequence(
    vlm_annotations: List[Dict],
    num_frames: int,
    num_subtasks: int
) -> List[int]:
    """Interpolate subtask sequence for all frames based on VLM keyframe annotations."""
    result = [-1] * num_frames
    sorted_annotations = sorted(vlm_annotations, key=lambda x: x["frame_index"])
    
    for i, annotation in enumerate(sorted_annotations):
        frame_idx = annotation["frame_index"]
        subtask_idx = annotation["subtask_index"]
        start_frame = frame_idx
        end_frame = sorted_annotations[i + 1]["frame_index"] if i + 1 < len(sorted_annotations) else num_frames
        
        for frame_num in range(start_frame, end_frame):
            if frame_num < num_frames:
                result[frame_num] = subtask_idx
    
    for i in range(num_frames):
        if result[i] == -1:
            for j in range(i - 1, -1, -1):
                if result[j] != -1:
                    result[i] = result[j]
                    break
            if result[i] == -1:
                result[i] = 1
    
    return result


def load_subtask_determination(file_path: str) -> Dict[str, Dict]:
    """Load subtask determination JSONL file into a dictionary keyed by episode_key."""
    episodes = {}
    if not os.path.exists(file_path):
        print(f"Warning: File does not exist: {file_path}")
        return episodes
    
    print(f"Loading from: {file_path}")
    line_count = 0
    error_count = 0
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            try:
                data = json.loads(line)
                episode_key = data.get("episode_key")
                if episode_key:
                    episodes[episode_key] = data
                else:
                    print(f"Warning: Line {line_num} missing episode_key")
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:  # Only print first 5 errors
                    print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(episodes)} episodes from {line_count} lines (errors: {error_count})")
    return episodes


def load_subtask_splits(file_path: str) -> Dict[str, Dict]:
    """Load subtask splits JSONL file into a dictionary keyed by episode_key."""
    splits = {}
    if not os.path.exists(file_path):
        print(f"Warning: File does not exist: {file_path}")
        return splits
    
    print(f"Loading from: {file_path}")
    line_count = 0
    error_count = 0
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            try:
                data = json.loads(line)
                episode_id = data.get("episode_id")
                scene_id = data.get("scene_id")
                episode_key = get_episode_key(scene_id, episode_id)
                splits[episode_key] = data
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:  # Only print first 5 errors
                    print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(splits)} splits from {line_count} lines (errors: {error_count})")
    return splits


def save_subtask_determination(file_path: str, episodes: Dict[str, Dict]):
    """Save subtask determination data back to JSONL file."""
    # Create backup
    if os.path.exists(file_path):
        backup_path = file_path + ".backup"
        import shutil
        shutil.copy2(file_path, backup_path)
    
    # Write all episodes
    with open(file_path, "w") as f:
        for episode_key, data in episodes.items():
            json.dump(data, f)
            f.write("\n")


def extract_frame_from_video(video_path: str, frame_idx: int) -> Optional[bytes]:
    """Extract a single frame from video at the given index."""
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def frame_to_base64(frame_bytes: bytes) -> str:
    """Convert frame bytes to base64 string."""
    return base64.b64encode(frame_bytes).decode("utf-8")


def generate_annotations_from_turning_points(
    turning_points: List[Dict],
    keyframes: List[int],
    num_frames: int,
    num_subtasks: int
) -> List[Dict]:
    """Generate annotations for all keyframes based on turning points.
    
    Args:
        turning_points: List of {frame_index, subtask_index} where subtask_index 
                       is the subtask that STARTS at this frame
        keyframes: List of all keyframe indices
        num_frames: Total number of frames
        num_subtasks: Total number of subtasks
        
    Returns:
        List of annotations for all keyframes with inferred subtask_index
    """
    if not turning_points:
        # No turning points, assign all to subtask 1
        return [{"frame_index": kf, "subtask_index": 1} for kf in keyframes]
    
    # Sort turning points by frame_index
    sorted_turning_points = sorted(turning_points, key=lambda x: x["frame_index"])
    
    # Generate full sequence first
    full_sequence = [-1] * num_frames
    
    # Assign subtasks based on turning points
    for i, tp in enumerate(sorted_turning_points):
        start_frame = tp["frame_index"]
        subtask_idx = tp["subtask_index"]
        
        # Determine end frame (next turning point or end of video)
        if i + 1 < len(sorted_turning_points):
            end_frame = sorted_turning_points[i + 1]["frame_index"]
        else:
            end_frame = num_frames
        
        # Assign this subtask to all frames in range
        for frame_num in range(start_frame, end_frame):
            if frame_num < num_frames:
                full_sequence[frame_num] = subtask_idx
    
    # Fill any remaining gaps (shouldn't happen, but safety check)
    for i in range(num_frames):
        if full_sequence[i] == -1:
            # Find previous non-empty value
            for j in range(i - 1, -1, -1):
                if full_sequence[j] != -1:
                    full_sequence[i] = full_sequence[j]
                    break
            if full_sequence[i] == -1:
                full_sequence[i] = 1
    
    # Generate annotations for all keyframes based on the sequence
    annotations = []
    for kf in keyframes:
        if kf < len(full_sequence):
            annotations.append({
                "frame_index": kf,
                "subtask_index": full_sequence[kf]
            })
    
    return annotations


def update_episode_annotations(
    episode_data: Dict,
    updated_annotations: List[Dict],
    num_frames: int
) -> Dict:
    """Update episode data with new annotations and regenerate subtask_sequence."""
    # Update vlm_annotations
    episode_data["vlm_annotations"] = updated_annotations
    
    # Regenerate subtask_sequence
    num_subtasks = episode_data.get("num_subtasks", 1)
    subtask_sequence = interpolate_subtask_sequence(
        updated_annotations,
        num_frames,
        num_subtasks
    )
    episode_data["subtask_sequence"] = subtask_sequence
    
    # Update subtask_counts
    subtask_counts = {}
    for idx in subtask_sequence:
        subtask_counts[str(idx)] = subtask_counts.get(str(idx), 0) + 1
    episode_data["subtask_counts"] = subtask_counts
    
    return episode_data

