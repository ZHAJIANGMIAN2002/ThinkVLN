#!/usr/bin/env python3
"""
Generate checking videos from subtask_determination.jsonl results.

This script reads the subtask determination results and generates annotated videos
showing the subtask sequence, actions, and keyframes for each episode.
"""

import os
import sys
import json
import cv2
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from subtask_determination import generate_annotated_video
except ImportError:
    # Fallback for when running from different directory
    from streamvln.cot_data.subtask_determination import generate_annotated_video


def get_episode_key(scene_id: Optional[str], episode_id: int) -> str:
    """Generate unique episode key from scene_id and episode_id"""
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)


def load_subtask_splits(subtask_splits_file: str) -> Dict[str, Dict]:
    """Load subtask splits from JSONL file into a dictionary keyed by episode_key"""
    subtask_splits = {}
    print(f"Loading subtask splits from: {subtask_splits_file}")
    
    if not os.path.exists(subtask_splits_file):
        print(f"Error: Subtask splits file not found: {subtask_splits_file}")
        return subtask_splits
    
    with open(subtask_splits_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    episode_id = data["episode_id"]
                    scene_id = data.get("scene_id")
                    episode_key = get_episode_key(scene_id, episode_id)
                    subtask_splits[episode_key] = data
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(subtask_splits)} subtask splits")
    return subtask_splits


def load_episode_from_summary(episode_id: int, scene_id: Optional[str], summary_file: str) -> Optional[Dict]:
    """Load episode data from summary.json file"""
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found: {summary_file}")
        return None
    
    with open(summary_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("id") == episode_id:
                        # If scene_id is provided, also match it
                        if scene_id is None or data.get("scene_id") == scene_id:
                            return data
                except json.JSONDecodeError:
                    continue
    
    return None


def extract_frames_from_video(video_path: str, temp_frame_dir: str = "/tmp/streamvln_frames") -> List[str]:
    """Extract frames from video file and return list of frame paths"""
    frame_paths = []
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return frame_paths
    
    os.makedirs(temp_frame_dir, exist_ok=True)
    
    print(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_frame_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths


def generate_video_for_episode(
    episode_data: Dict,
    subtask_data: Dict,
    trajectory_dir: str,
    output_dir: str,
    temp_frame_dir: str = "/tmp/streamvln_frames"
) -> bool:
    """
    Generate checking video for a single episode.
    
    Args:
        episode_data: Episode data from subtask_determination.jsonl
        subtask_data: Subtask split data containing plan
        trajectory_dir: Directory containing trajectory data
        output_dir: Directory to save output videos
        temp_frame_dir: Temporary directory for frame extraction
        
    Returns:
        True if successful, False otherwise
    """
    episode_key = episode_data.get("episode_key")
    episode_id = episode_data.get("episode_id")
    scene_id = episode_data.get("scene_id")
    
    # Get video path from summary.json
    summary_file = os.path.join(trajectory_dir, "summary.json")
    episode_summary = load_episode_from_summary(episode_id, scene_id, summary_file)
    
    if not episode_summary:
        print(f"Error: Episode {episode_key} not found in summary.json")
        return False
    
    video_rel_path = episode_summary.get("video", "")
    actions = episode_summary.get("actions", [])
    
    if not video_rel_path or not actions:
        print(f"Error: Missing video path or actions for episode {episode_key}")
        return False
    
    video_path = os.path.join(trajectory_dir, video_rel_path, "trajectory.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Get data from episode_data
    subtask_sequence = episode_data.get("subtask_sequence", [])
    keyframes = episode_data.get("keyframes", [])
    plan = subtask_data.get("plan", [])
    
    if not subtask_sequence or not plan:
        print(f"Error: Missing subtask_sequence or plan for episode {episode_key}")
        return False
    
    # Extract frames from video
    frame_paths = extract_frames_from_video(video_path, temp_frame_dir)
    if not frame_paths:
        print(f"Error: Failed to extract frames from video")
        return False
    
    # Ensure actions and subtask_sequence match frame count
    num_frames = len(frame_paths)
    if len(actions) != num_frames:
        print(f"Warning: Actions count ({len(actions)}) != frames count ({num_frames}), truncating/padding")
        if len(actions) > num_frames:
            actions = actions[:num_frames]
        else:
            actions = actions + [0] * (num_frames - len(actions))
    
    if len(subtask_sequence) != num_frames:
        print(f"Warning: Subtask sequence count ({len(subtask_sequence)}) != frames count ({num_frames}), truncating/padding")
        if len(subtask_sequence) > num_frames:
            subtask_sequence = subtask_sequence[:num_frames]
        else:
            # Pad with last subtask
            last_subtask = subtask_sequence[-1] if subtask_sequence else 1
            subtask_sequence = subtask_sequence + [last_subtask] * (num_frames - len(subtask_sequence))
    
    # Generate output video path
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{episode_key}_checking.mp4")
    
    # Generate annotated video
    print(f"Generating checking video for episode {episode_key}...")
    success = generate_annotated_video(
        frame_paths=frame_paths,
        actions=actions,
        subtask_sequence=subtask_sequence,
        subtask_list=plan,
        keyframes=keyframes,
        output_video_path=output_video_path,
        fps=6
    )
    
    if success:
        print(f"✓ Successfully generated checking video: {output_video_path}")
    else:
        print(f"✗ Failed to generate checking video for episode {episode_key}")
    
    return success


def sample_random_episodes(
    subtask_det_file: str,
    num_samples: int = 20,
    display_only: bool = False
) -> List[Dict]:
    """
    Sample random episodes from subtask_det_v2.jsonl and display results.
    
    Args:
        subtask_det_file: Path to subtask_det_v2.jsonl file
        num_samples: Number of random episodes to sample (default: 20)
        display_only: If True, only display info; if False, return episodes for processing
        
    Returns:
        List of sampled episode dictionaries
    """
    print(f"Loading episodes from: {subtask_det_file}")
    if not os.path.exists(subtask_det_file):
        print(f"Error: File not found: {subtask_det_file}")
        return []
    
    # Load all episodes
    all_episodes = []
    with open(subtask_det_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Only include successful episodes
                    if data.get("status") == "success":
                        all_episodes.append(data)
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(all_episodes)} successful episodes from {subtask_det_file}")
    
    if not all_episodes:
        print("No episodes found to sample")
        return []
    
    # Sample random episodes
    num_to_sample = min(num_samples, len(all_episodes))
    sampled_episodes = random.sample(all_episodes, num_to_sample)
    
    print(f"\n{'='*80}")
    print(f"SAMPLED {num_to_sample} RANDOM EPISODES")
    print(f"{'='*80}")
    
    # Display information for each sampled episode
    for idx, episode in enumerate(sampled_episodes, 1):
        episode_key = episode.get("episode_key", "Unknown")
        episode_id = episode.get("episode_id", "N/A")
        scene_id = episode.get("scene_id", "N/A")
        num_frames = episode.get("num_frames", 0)
        num_subtasks = episode.get("num_subtasks", 0)
        keyframes = episode.get("keyframes", [])
        subtask_counts = episode.get("subtask_counts", {})
        
        print(f"\n[{idx}/{num_to_sample}] Episode: {episode_key}")
        print(f"  Episode ID: {episode_id}")
        print(f"  Scene ID: {scene_id}")
        print(f"  Frames: {num_frames}")
        print(f"  Subtasks: {num_subtasks}")
        print(f"  Keyframes: {len(keyframes)} (first 10: {keyframes[:10]})")
        print(f"  Subtask Distribution: {subtask_counts}")
        
        # Show vlm_annotations if available
        vlm_annotations = episode.get("vlm_annotations", [])
        if vlm_annotations:
            print(f"  VLM Annotations: {len(vlm_annotations)} turning points")
            for ann in vlm_annotations[:5]:  # Show first 5
                print(f"    - Frame {ann.get('frame_index')}: Subtask {ann.get('subtask_index')}")
            if len(vlm_annotations) > 5:
                print(f"    ... and {len(vlm_annotations) - 5} more")
    
    print(f"\n{'='*80}")
    print(f"Summary: Sampled {num_to_sample} episodes from {len(all_episodes)} total episodes")
    print(f"{'='*80}\n")
    
    return sampled_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Generate checking videos from subtask_determination.jsonl results"
    )
    parser.add_argument(
        "--subtask_determination_file",
        type=str,
        default="streamvln/cot_data/subtask_determination_1121_final.jsonl",
        help="Path to subtask_determination.jsonl file"
    )
    parser.add_argument(
        "--subtask_splits_file",
        type=str,
        default="streamvln/cot_data/subtask_splits.jsonl",
        help="Path to subtask_splits.jsonl file"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="data/trajectory_data/R2R",
        help="Path to trajectory data directory (contains summary.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/checking_videos",
        help="Directory to save output videos"
    )
    parser.add_argument(
        "--episode_key",
        type=str,
        default=None,
        help="Optional: Process only a specific episode_key"
    )
    parser.add_argument(
        "--temp_frame_dir",
        type=str,
        default="/tmp/streamvln_frames",
        help="Temporary directory for frame extraction"
    )
    parser.add_argument(
        "--sample_random",
        action="store_true",
        help="Sample random episodes from subtask_det_v2.jsonl and generate videos for them"
    )
    parser.add_argument(
        "--subtask_det_v2_file",
        type=str,
        default="streamvln/cot_data/subtask_det_v2.jsonl",
        help="Path to subtask_det_v2.jsonl file (for random sampling)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of random episodes to sample (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Load subtask splits (needed for both modes)
    subtask_splits = load_subtask_splits(args.subtask_splits_file)
    if not subtask_splits:
        print("Error: Failed to load subtask splits")
        return
    
    # Handle random sampling mode
    if args.sample_random:
        sampled_episodes = sample_random_episodes(
            subtask_det_file=args.subtask_det_v2_file,
            num_samples=args.num_samples,
            display_only=False
        )
        if not sampled_episodes:
            print("No episodes to process")
            return
        episodes = sampled_episodes
        print(f"\nGenerating videos for {len(episodes)} sampled episodes...")
    else:
        # Load subtask determination results
        print(f"Loading subtask determination results from: {args.subtask_determination_file}")
        if not os.path.exists(args.subtask_determination_file):
            print(f"Error: Subtask determination file not found: {args.subtask_determination_file}")
            return
        
        episodes = []
        with open(args.subtask_determination_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Only process successful episodes
                        if data.get("status") == "success":
                            episodes.append(data)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(episodes)} successful episodes from subtask_determination.jsonl")
        
        # Filter by episode_key if specified
        if args.episode_key:
            episodes = [e for e in episodes if e.get("episode_key") == args.episode_key]
            print(f"Filtered to {len(episodes)} episode(s) matching episode_key: {args.episode_key}")
    
    if not episodes:
        print("No episodes to process")
        return
    
    # Process each episode
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, episode_data in enumerate(episodes, 1):
        episode_key = episode_data.get("episode_key")
        print(f"\n[{idx}/{len(episodes)}] Processing episode {episode_key}")
        
        # Check if subtask split exists
        if episode_key not in subtask_splits:
            print(f"  ⚠ Skipping: No subtask split found for episode {episode_key}")
            skipped += 1
            continue
        
        subtask_data = subtask_splits[episode_key]
        
        # Generate video
        success = generate_video_for_episode(
            episode_data=episode_data,
            subtask_data=subtask_data,
            trajectory_dir=args.trajectory_dir,
            output_dir=args.output_dir,
            temp_frame_dir=args.temp_frame_dir
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total episodes: {len(episodes)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

