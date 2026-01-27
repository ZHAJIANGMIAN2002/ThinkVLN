#!/usr/bin/env python3
"""
Test script to generate CoT for a whole episode from a specific video
"""
import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cot_generation import (
    extract_frames_from_video,
    generate_cot_for_step,
    extract_plan_from_cot,
    extract_subtask_index_from_cot,
    frame_to_base64
)

def test_whole_episode():
    video_path = "/home/autolab/swx/StreamVLN/data/trajectory_data/R2R/images/1LXtFkjw3qL_r2r_000085/trajectory.mp4"
    
    # Episode data from summary.json
    episode_data = {
        "id": 85,
        "video": "images/1LXtFkjw3qL_r2r_000085",
        "instructions": ["Go straight to the hallway and then turn left.  Go past the bed.  Veer to the right and go through the white door.  Stop when you're in the doorway. "],
        "actions": [-1, 3, 3, 3, 3, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "trajectory_id": 67,
        "scene_id": "1LXtFkjw3qL"
    }
    
    instruction = episode_data["instructions"][0]
    actions = episode_data["actions"]
    
    print(f"Episode ID: {episode_data['id']}")
    print(f"Scene ID: {episode_data['scene_id']}")
    print(f"Trajectory ID: {episode_data['trajectory_id']}")
    print(f"Instruction: {instruction}")
    print(f"Total actions: {len(actions)}")
    print(f"Video path: {video_path}")
    print()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        return
    
    # Extract frames to a temporary directory
    output_dir = "/tmp/test_cot_episode_frames"
    print(f"Extracting frames from video...")
    frame_paths = extract_frames_from_video(video_path, output_dir)
    print(f"Extracted {len(frame_paths)} frames")
    print()
    
    if len(frame_paths) == 0:
        print("ERROR: No frames extracted from video")
        return
    
    if len(frame_paths) != len(actions):
        print(f"WARNING: Frame count ({len(frame_paths)}) != Action count ({len(actions)})")
        print(f"Will process up to step {min(len(frame_paths), len(actions)) - 1}")
        print()
    
    # Generate CoT for all steps sequentially
    cot_dataset = []
    plan = ""  # Will be extracted from step 0
    max_steps = min(len(frame_paths), len(actions))
    previous_subtask_index = -1
    previous_action = None
    output_file = "/tmp/test_cot_episode_result.json"
    
    print(f"Generating CoT for {max_steps} steps...")
    print(f"Results will be saved incrementally to: {output_file}")
    print("="*80)
    
    for step_id in range(10):
        frame_path = frame_paths[step_id]
        action = actions[step_id]
        
        print(f"\n[{step_id+1}/{max_steps}] Processing Step {step_id}")
        print(f"  Frame: {os.path.basename(frame_path)}")
        print(f"  Action: {action}")
        if step_id > 0:
            print(f"  Previous Subtask Index: {previous_subtask_index if previous_subtask_index >= 0 else 'N/A'}")
            action_names = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
            prev_action_name = action_names.get(previous_action, "unknown") if previous_action is not None else "N/A"
            print(f"  Previous Action: {prev_action_name}")
        
        if not os.path.exists(frame_path):
            print(f"  ERROR: Frame not found, skipping step {step_id}")
            continue
        
        print(f"  Generating CoT...")
        cot_result = generate_cot_for_step(
            instruction=instruction,
            cur_frame_path=frame_path,
            history_cot_results=cot_dataset,
            plan=plan,
            action=action,
            step_id=step_id,
            max_steps=max_steps,
            previous_subtask_index=previous_subtask_index,
            previous_action=previous_action
        )
        
        if cot_result:
            cot_dataset.append(cot_result)
            
            # Extract plan from step 0 for use in later steps
            if step_id == 0:
                plan = extract_plan_from_cot(cot_result["cot"])
                if plan:
                    print(f"  ✓ Plan extracted: {plan[:100]}...")
                else:
                    print(f"  ⚠ Warning: Could not extract plan from step 0")
            
            # Extract subtask index and action for next step (update after each step)
            previous_subtask_index = extract_subtask_index_from_cot(cot_result["cot"])
            previous_action = action
            
            if previous_subtask_index >= 0:
                print(f"  ✓ Subtask index extracted: {previous_subtask_index}")
            
            print(f"  ✓ CoT generated successfully")
            
            # Save results after each step
            episode_result = {
                "episode_id": episode_data["id"],
                "trajectory_id": episode_data["trajectory_id"],
                "scene_id": episode_data["scene_id"],
                "instruction": instruction,
                "plan": plan,
                "num_steps": len(cot_dataset),
                "cot_steps": cot_dataset
            }
            
            with open(output_file, "w") as f:
                json.dump(episode_result, f, indent=2)
            
            print(f"  💾 Results saved (step {step_id+1}/{max_steps})")
        else:
            print(f"  ✗ ERROR: Failed to generate CoT for step {step_id}")
            print(f"  Stopping episode processing")
            break
    
    print("\n" + "="*80)
    print(f"COMPLETED: Generated CoT for {len(cot_dataset)}/{max_steps} steps")
    print("="*80)
    print(f"\nFinal results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Episode ID: {episode_data['id']}")
    print(f"Steps processed: {len(cot_dataset)}/{max_steps}")
    print(f"Plan: {plan[:200] if plan else 'Not extracted'}")
    print("\nFirst step CoT preview:")
    if cot_dataset:
        first_cot = cot_dataset[0]["cot"]
        print(first_cot[:500] + "..." if len(first_cot) > 500 else first_cot)
    
    if len(cot_dataset) > 1:
        print(f"\nLast step CoT preview:")
        last_cot = cot_dataset[-1]["cot"]
        print(last_cot[:500] + "..." if len(last_cot) > 500 else last_cot)

if __name__ == "__main__":
    test_whole_episode()

