import os
import json
import base64
import argparse
from openai import OpenAI


TASK_DECOMPOSE_PROMPT = """
1. Role and Objective

You are an expert Visual-Language Navigation (VLN) AI assistant. Your task is to parse user's navigation instruction and create a high-level plan by decomposing the task into numbered sub-tasks.

2. Output Format

Generate a structured response with the following format:

[START_REASONING]
TASK: [Briefly paraphrase the navigation instruction]
(Example: "Go past the painting and enter the dining room, then stop.")

PLAN: [Break down the instruction into a sequence of numbered, high-level sub-tasks. Each subtask should be a clear action step.]
(Example:
1. Walk straight down the hallway.
2. Turn left and enter the room on the left.
3. Stop near the sink.
)
[END_REASONING]

3. Guidelines

- Each subtask should be simple and actionable (e.g., "Walk forward", "Turn left", "Stop at X").
- Subtasks should be ordered sequentially reflecting the navigation flow.
- Use clear, concise language without complex descriptions.
- The plan will be used by downstream agents to guide step-by-step navigation.
- The num of subtasks must no more than 5. This rule should be strictly followed.
"""


# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "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"),
)

def extract_plan_from_response(response_text: str) -> list:
    """Extract plan list from API response
    
    Args:
        response_text: The full response from the API
        
    Returns:
        List of plan items (e.g., ["Walk straight down the hallway", "Turn left", ...])
    """
    try:
        # Find PLAN section
        if "PLAN:" in response_text:
            plan_start = response_text.find("PLAN:")
            plan_end = response_text.find("[END_REASONING]", plan_start)
            if plan_end == -1:
                plan_end = len(response_text)
            
            plan_section = response_text[plan_start:plan_end]
            
            # Extract numbered items (e.g., "1. Walk straight", "2. Turn left")
            import re
            pattern = r"^\s*(\d+)\.\s+(.+?)(?=(?:\n\s*\d+\.)|$)"
            matches = re.findall(pattern, plan_section, re.MULTILINE | re.DOTALL)
            
            if matches:
                # Return list of plan items (clean up whitespace)
                plan_list = [item[1].strip() for item in matches]
                return plan_list
    except Exception as e:
        print(f"Warning: Could not extract plan from response: {e}")
    
    return []

def generate_plan_from_instruction(instruction: str) -> dict:
    """Generate plan/subtask decomposition from instruction using API
    
    Args:
        instruction: The navigation instruction
        
    Returns:
        Dictionary with keys:
            - instruction: original instruction
            - plan: list of subtask strings
            - raw_response: full API response
    """
    try:
        prompt = f"""{TASK_DECOMPOSE_PROMPT}

User Instruction: {instruction}
"""
        
        # Call API
        response = client.chat.completions.create(
            # model="qwen/qwen3-vl-32b-instruct",
            model='qwen/qwen2.5-vl-32b-instruct:free"',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        response_text = response.choices[0].message.content
        plan_list = extract_plan_from_response(response_text)
        
        return {
            "instruction": instruction,
            "plan": plan_list,
            "raw_response": response_text
        }
    
    except Exception as e:
        print(f"Error generating plan for instruction '{instruction}': {e}")
        import traceback
        traceback.print_exc()
        return None

def process_episodes_for_subtask_split(trajectory_dir: str, output_file: str):
    """Process all episodes and generate subtask splits
    
    Args:
        trajectory_dir: Path to trajectory data directory (contains summary.json)
        output_file: Output file to save all subtask splits (JSONL format)
    """
    # Load annotations
    annotation_file = os.path.join(trajectory_dir, "summary.json")
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return
    
    annotations = []
    with open(annotation_file, "r") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    
    print(f"Loaded {len(annotations)} episodes")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load already processed episode IDs from output file if it exists
    processed_episode_ids = set()
    if os.path.exists(output_file):
        print(f"\nLoading previously processed episodes from {output_file}")
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_episode_ids.add(record["episode_id"])
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_episode_ids)} previously processed episodes")
    
    # Process each episode
    new_episodes_count = 0
    skipped_episodes_count = 0
    
    for idx, episode_data in enumerate(annotations):
        episode_id = episode_data["id"]
        
        # Skip if already processed
        if episode_id in processed_episode_ids:
            skipped_episodes_count += 1
            print(f"\n[{idx+1}/{len(annotations)}] Skipping episode {episode_id} (already processed)")
            continue
        
        instruction = episode_data["instructions"][0] if isinstance(episode_data["instructions"], list) else episode_data["instructions"]
        
        print(f"\n[{idx+1}/{len(annotations)}] Processing episode {episode_id}")
        print(f"Instruction: {instruction}")
        
        # Generate plan for this instruction
        result = generate_plan_from_instruction(instruction)
        
        if result:
            # Create output record
            output_record = {
                "episode_id": episode_id,
                "scene_id": episode_data.get("scene_id"),
                "trajectory_id": episode_data.get("trajectory_id"),
                "instruction": instruction,
                "plan": result["plan"],
                "num_subtasks": len(result["plan"])
            }
            
            # Append to output file (one record per line)
            with open(output_file, "a") as f:
                json.dump(output_record, f)
                f.write("\n")
            
            new_episodes_count += 1
            print(f"Generated {len(result['plan'])} subtasks:")
            for i, subtask in enumerate(result['plan'], 1):
                print(f"  {i}. {subtask}")
        else:
            print(f"Warning: Failed to generate plan for episode {episode_id}")
    
    print(f"\n{'='*80}")
    print(f"Completed!")
    print(f"  Previously processed: {len(processed_episode_ids)} episodes")
    print(f"  Newly processed: {new_episodes_count} episodes")
    print(f"  Skipped: {skipped_episodes_count} episodes")
    print(f"  Total in file: {len(processed_episode_ids) + new_episodes_count} episodes")
    print(f"Subtask splits saved to: {output_file}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R", 
                       help="Path to trajectory data directory")
    parser.add_argument("--output_file", type=str, default="data/subtask_splits/R2R/subtask_splits.jsonl",
                       help="Output file for subtask splits (JSONL format)")
    args = parser.parse_args()
    
    process_episodes_for_subtask_split(args.trajectory_dir, args.output_file)

if __name__ == "__main__":
    main()
