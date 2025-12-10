#!/usr/bin/env python3
"""
Test script to generate subtask split for a specific instruction
"""
import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subtask_split_deploy import generate_plan_from_instruction, extract_plan_from_response, process_episodes_for_subtask_split

def test_single_instruction():
    """Test generating plan for a single instruction"""
    
    # Sample instruction from the dataset
    instruction = "Go straight to the hallway and then turn left. Go past the bed. Veer to the right and go through the white door. Stop when you're in the doorway."
    
    print("="*80)
    print("TEST: Subtask Split Generation")
    print("="*80)
    print(f"\nInstruction: {instruction}\n")
    
    print("Generating plan from instruction...")
    print("-"*80)
    
    result = generate_plan_from_instruction(instruction)
    
    if result:
        plan_list = result["plan"]
        
        print(f"\n✓ Plan generated successfully!")
        print(f"\nNumber of subtasks: {len(plan_list)}\n")
        
        print("Subtasks:")
        for i, subtask in enumerate(plan_list, 1):
            print(f"  {i}. {subtask}")
        
        print("\n" + "="*80)
        print("Raw API Response:")
        print("="*80)
        print(result["raw_response"][:800] + "..." if len(result["raw_response"]) > 800 else result["raw_response"])
        
        # Save to file
        output_file = "/tmp/test_subtask_split_result.json"
        output_data = {
            "instruction": instruction,
            "num_subtasks": len(plan_list),
            "plan": plan_list,
            "raw_response": result["raw_response"]
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print("\n" + "="*80)
        print(f"Results saved to: {output_file}")
        print("="*80)
    else:
        print("✗ ERROR: Failed to generate plan")
        return

def test_multiple_instructions():
    """Test generating plans for multiple sample instructions"""
    
    sample_instructions = [
        "Go straight to the hallway and then turn left. Go past the bed.",
        "Walk to the doorway next to the bedroom. Walk straight to the shower door.",
        "Exit the bedroom, enter the bathroom, wait at the toilet.",
        "Turn right walk out doorway. Turn right walk towards bathroom. Turn right walk past bathroom.",
        "Go between the two tables and through the doorway and go to your left to the sink."
    ]
    
    print("\n" + "="*80)
    print("TEST: Multiple Instructions Subtask Split")
    print("="*80)
    
    results = []
    
    for idx, instruction in enumerate(sample_instructions, 1):
        print(f"\n[{idx}/{len(sample_instructions)}] Processing instruction:")
        print(f"  {instruction}")
        
        result = generate_plan_from_instruction(instruction)
        
        if result:
            plan_list = result["plan"]
            print(f"  ✓ Generated {len(plan_list)} subtasks:")
            for i, subtask in enumerate(plan_list, 1):
                print(f"    {i}. {subtask}")
            
            results.append({
                "instruction": instruction,
                "num_subtasks": len(plan_list),
                "plan": plan_list
            })
        else:
            print(f"  ✗ Failed to generate plan")
    
    # Save all results
    output_file = "/tmp/test_subtask_split_multiple_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print(f"Successfully processed: {len(results)}/{len(sample_instructions)}")
    print("="*80)

def test_with_episode_data():
    """Test generating plans for episodes from summary.json"""
    
    summary_file = "/home/autolab/swx/StreamVLN/data/trajectory_data/R2R/summary.json"
    
    if not os.path.exists(summary_file):
        print(f"ERROR: Summary file not found: {summary_file}")
        return
    
    print("\n" + "="*80)
    print("TEST: Generate Plans for Episodes from summary.json")
    print("="*80)
    
    # Load first 3 episodes
    episodes = []
    with open(summary_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only first 3 episodes
                break
            if line.strip():
                episodes.append(json.loads(line))
    
    print(f"\nLoaded {len(episodes)} episodes from summary.json")
    
    results = []
    output_file = "/home/autolab/swx/StreamVLN/streamvln/cot_data/tests/test_subtask_split_from_episodes.jsonl"
    
    # Clear output file
    with open(output_file, "w") as f:
        pass
    
    for idx, episode_data in enumerate(episodes, 1):
        episode_id = episode_data["id"]
        instruction = episode_data["instructions"][0] if isinstance(episode_data["instructions"], list) else episode_data["instructions"]
        scene_id = episode_data.get("scene_id")
        
        print(f"\n[{idx}/{len(episodes)}] Episode {episode_id} (Scene: {scene_id})")
        print(f"  Instruction: {instruction[:80]}...")
        
        result = generate_plan_from_instruction(instruction)
        
        if result:
            plan_list = result["plan"]
            print(f"  ✓ Generated {len(plan_list)} subtasks")
            
            # Save to JSONL
            record = {
                "episode_id": episode_id,
                "scene_id": scene_id,
                "trajectory_id": episode_data.get("trajectory_id"),
                "instruction": instruction,
                "num_subtasks": len(plan_list),
                "plan": plan_list
            }
            
            with open(output_file, "a") as f:
                json.dump(record, f)
                f.write("\n")
            
            results.append(record)
            
            print(f"  Plan preview:")
            for i, subtask in enumerate(plan_list[:3], 1):
                print(f"    {i}. {subtask}")
            if len(plan_list) > 3:
                print(f"    ... and {len(plan_list) - 3} more")
        else:
            print(f"  ✗ Failed to generate plan")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print(f"Successfully processed: {len(results)}/{len(episodes)} episodes")
    print("="*80)

def test_process_episodes_for_subtask_split():
    """Test the full process_episodes_for_subtask_split function"""
    
    trajectory_dir = "/home/autolab/swx/StreamVLN/data/trajectory_data/R2R"
    output_file = "/tmp/test_process_episodes_subtask_split.jsonl"
    
    # Clean up output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("\n" + "="*80)
    print("TEST: process_episodes_for_subtask_split")
    print("="*80)
    print(f"\nTrajectory directory: {trajectory_dir}")
    print(f"Output file: {output_file}")
    print(f"Max workers: 4 (for testing)")
    print("-"*80)
    
    # Call the function with limited workers for testing
    process_episodes_for_subtask_split(
        trajectory_dir=trajectory_dir,
        output_file=output_file,
        max_workers=4
    )
    
    # Verify output file was created
    if os.path.exists(output_file):
        print("\n" + "="*80)
        print("VERIFICATION: Checking output file")
        print("="*80)
        
        # Count processed episodes
        processed_count = 0
        sample_records = []
        
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_count += 1
                        if len(sample_records) < 3:
                            sample_records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n✓ Output file created successfully")
        print(f"  Total records: {processed_count}")
        
        if sample_records:
            print(f"\n  Sample records (first {len(sample_records)}):")
            for i, record in enumerate(sample_records, 1):
                episode_id = record.get("episode_id")
                scene_id = record.get("scene_id")
                num_subtasks = record.get("num_subtasks", 0)
                plan = record.get("plan", [])
                
                print(f"\n  [{i}] Episode {episode_id} (Scene: {scene_id})")
                print(f"      Subtasks: {num_subtasks}")
                if plan:
                    print(f"      Plan preview:")
                    for j, subtask in enumerate(plan[:2], 1):
                        print(f"        {j}. {subtask[:60]}...")
                    if len(plan) > 2:
                        print(f"        ... and {len(plan) - 2} more")
        
        # Verify record structure
        if sample_records:
            required_fields = ["episode_id", "scene_id", "instruction", "plan", "num_subtasks"]
            first_record = sample_records[0]
            missing_fields = [field for field in required_fields if field not in first_record]
            
            if missing_fields:
                print(f"\n  ⚠ Warning: Missing fields in records: {missing_fields}")
            else:
                print(f"\n  ✓ All required fields present in records")
        
        print("\n" + "="*80)
        print(f"Test completed successfully!")
        print(f"Output file: {output_file}")
        print("="*80)
    else:
        print("\n✗ ERROR: Output file was not created")
        return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test subtask split generation")
    parser.add_argument("--test", type=str, default="process", 
                       choices=["single", "multiple", "episodes", "process"],
                       help="Type of test to run")
    args = parser.parse_args()
    
    if args.test == "single":
        test_single_instruction()
    elif args.test == "multiple":
        test_multiple_instructions()
    elif args.test == "episodes":
        test_with_episode_data()
    elif args.test == "process":
        test_process_episodes_for_subtask_split()
