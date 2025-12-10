#!/usr/bin/env python3
"""
Test determine_subtasks_for_video function with real data
"""
import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subtask_determination import determine_subtasks_for_video


def load_test_episodes():
    """Load test episodes from JSONL file"""
    test_file = os.path.join(os.path.dirname(__file__), "test_subtask_split_from_episodes.jsonl")
    episodes = []
    
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return []
    
    with open(test_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return episodes


def load_episode_from_summary(episode_id: int, summary_file: str, scene_id: str = None) -> dict:
    """Load full episode data from summary.json
    
    Args:
        episode_id: The episode ID to search for
        summary_file: Path to summary.json file
        scene_id: Optional scene ID for additional filtering
        
    Returns:
        Episode data dict or None if not found
    """
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found: {summary_file}")
        return None
    
    with open(summary_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if scene_id is not None:
                        if data.get("scene_id") == scene_id:
                            if data.get("id") == episode_id:
                                return data
                    else:
                        if data.get("id") == episode_id:
                            return data
                        else:
                            return data
                except json.JSONDecodeError:
                    continue
    
    return None


def test_episode(episode_id: int, subtask_plan: list, instruction: str, 
                video_dir: str, summary_file: str, output_dir: str, scene_id: str = None):
    """Test determine_subtasks_for_video with a single episode"""
    
    print(f"\n{'='*100}")
    print(f"Testing Episode {episode_id}")
    if scene_id:
        print(f"Scene ID: {scene_id}")
    print(f"{'='*100}")
    
    # Load episode data to get video path
    episode_data = load_episode_from_summary(episode_id, summary_file, scene_id=scene_id)
    if not episode_data:
        print(f"✗ Error: Episode {episode_id} not found in summary.json")
        return None
    
    video_rel_path = episode_data.get("video", "")
    actions = episode_data.get("actions", [])
    if len(actions) == 0:
        print(f"✗ Error: Actions not found for episode {episode_id}")
        return None
    full_video_path = os.path.join(video_dir, video_rel_path, "trajectory.mp4")
    
    if not os.path.exists(full_video_path):
        print(f"✗ Error: Video file not found: {full_video_path}")
        return None
    
    print(f"Video: {full_video_path}")
    print(f"Instruction: {instruction}")
    print(f"Subtasks: {len(subtask_plan)}")
    
    # Call determine_subtasks_for_video
    output_file = os.path.join(output_dir, f"episode_{episode_id}_result.json")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCalling determine_subtasks_for_video...")
    result = determine_subtasks_for_video(
        video_or_frames=full_video_path,
        actions=actions,
        subtask_list=subtask_plan,
        instruction=instruction,
        output_file=output_file,
        use_vlm=True  # Use fallback mode for testing
    )
    
    if result["status"] != "success":
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
        return None
    
    print(f"\n✓ Results:")
    print(f"  Total frames: {result['num_frames']}")
    print(f"  Subtasks: {result['num_subtasks']}")
    print(f"  Subtask sequence: {result['subtask_sequence']}")
    print(f"  Distribution: {result['subtask_counts']}")
    print(f"  Keyframes detected: {len(result['keyframes'])}")
    print(f"  Output saved: {output_file}")
    
    return result


def main():
    print("\n" + "="*100)
    print("TESTING determine_subtasks_for_video WITH REAL DATA")
    print("="*100)
    
    trajectory_dir = "/home/autolab/swx/StreamVLN/data/trajectory_data/R2R"
    summary_file = os.path.join(trajectory_dir, "summary.json")
    output_dir = "/home/autolab/swx/StreamVLN/test/subtask_determination_test_real_data"
    
    # Load test episodes
    test_episodes = load_test_episodes()
    
    if not test_episodes:
        print("Error: Could not load test episodes")
        return
    
    print(f"\nLoaded {len(test_episodes)} test episodes")
    
    results = []
    for test_ep in test_episodes:
        episode_id = test_ep['episode_id']
        subtask_plan = test_ep['plan']
        instruction = test_ep['instruction']
        scene_id = test_ep.get('scene_id')
        
        result = test_episode(
            episode_id=episode_id,
            subtask_plan=subtask_plan,
            instruction=instruction,
            video_dir=trajectory_dir,
            summary_file=summary_file,
            output_dir=output_dir,
            scene_id=scene_id
        )
        
        if result:
            results.append({
                "episode_id": episode_id,
                "status": "success",
                "num_frames": result['num_frames'],
                "num_subtasks": result['num_subtasks'],
                "keyframes": len(result['keyframes'])
            })
        else:
            results.append({
                "episode_id": episode_id,
                "status": "failed"
            })
    
    # Summary
    print(f"\n{'='*100}")
    print("TEST SUMMARY")
    print(f"{'='*100}")
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"Total: {len(results)} | Success: {successful} | Failed: {failed}")
    
    for r in results:
        if r['status'] == 'success':
            print(f"✓ Episode {r['episode_id']}: {r['num_frames']} frames, {r['num_subtasks']} subtasks, {r['keyframes']} keyframes")
        else:
            print(f"✗ Episode {r['episode_id']}: Failed")


if __name__ == "__main__":
    main()
