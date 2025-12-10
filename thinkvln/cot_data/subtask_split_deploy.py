import os
import json
import base64
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


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


# # Initialize OpenAI client
# client = OpenAI(
#     base_url="http://localhost:11451/v1",
#     api_key="EMPTY"
# )

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
            # model="openai/gpt-oss-120b",
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

def get_episode_key(scene_id, episode_id):
    """Generate unique episode key from scene_id and episode_id
    
    Args:
        scene_id: Scene ID (can be None)
        episode_id: Episode ID
        
    Returns:
        Unique string key combining scene_id and episode_id
    """
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)

def process_single_episode(episode_data, episode_idx, total_episodes, processed_episode_keys, output_file, write_lock):
    """Process a single episode and write result to file
    
    Args:
        episode_data: Episode data dictionary
        episode_idx: Index of episode in the list
        total_episodes: Total number of episodes
        processed_episode_keys: Set of already processed episode keys (scene_id_episode_id)
        output_file: Output file path
        write_lock: Lock for thread-safe file writing
        
    Returns:
        Tuple (episode_key, success, skipped)
    """
    episode_id = episode_data["id"]
    scene_id = episode_data.get("scene_id")
    episode_key = get_episode_key(scene_id, episode_id)
    
    # Skip if already processed
    if episode_key in processed_episode_keys:
        print(f"\n[{episode_idx+1}/{total_episodes}] Skipping episode {episode_key} (already processed)")
        return (episode_key, False, True)
    
    instruction = episode_data["instructions"][0] if isinstance(episode_data["instructions"], list) else episode_data["instructions"]
    
    print(f"\n[{episode_idx+1}/{total_episodes}] Processing episode {episode_key}")
    print(f"Instruction: {instruction}")
    
    # Generate plan for this instruction
    result = generate_plan_from_instruction(instruction)
    
    if result:
        # Create output record
        output_record = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "trajectory_id": episode_data.get("trajectory_id"),
            "instruction": instruction,
            "plan": result["plan"],
            "num_subtasks": len(result["plan"])
        }
        
        # Append to output file (one record per line) - thread-safe
        with write_lock:
            with open(output_file, "a") as f:
                json.dump(output_record, f)
                f.write("\n")
                f.flush()
        
        print(f"Generated {len(result['plan'])} subtasks:")
        for i, subtask in enumerate(result['plan'], 1):
            print(f"  {i}. {subtask}")
        return (episode_key, True, False)
    else:
        print(f"Warning: Failed to generate plan for episode {episode_key}")
        return (episode_key, False, False)

def process_episodes_for_subtask_split(trajectory_dir: str, output_file: str, max_workers: int = 8):
    """Process all episodes and generate subtask splits
    
    Args:
        trajectory_dir: Path to trajectory data directory (contains summary.json)
        output_file: Output file to save all subtask splits (JSONL format)
        max_workers: Maximum number of concurrent workers
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
    
    # Load already processed episode keys from output file if it exists
    processed_episode_keys = set()
    if os.path.exists(output_file):
        print(f"\nLoading previously processed episodes from {output_file}")
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        scene_id = record.get("scene_id")
                        episode_id = record.get("episode_id")
                        if episode_id is not None:
                            episode_key = get_episode_key(scene_id, episode_id)
                            processed_episode_keys.add(episode_key)
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_episode_keys)} previously processed episodes")
    
    # Filter out already processed episodes
    tasks = []
    for idx, episode_data in enumerate(annotations):
        episode_id = episode_data["id"]
        scene_id = episode_data.get("scene_id")
        episode_key = get_episode_key(scene_id, episode_id)
        if episode_key not in processed_episode_keys:
            tasks.append((episode_data, idx, len(annotations)))
    
    print(f"Processing {len(tasks)} episodes with {max_workers} workers")
    
    # Process episodes concurrently
    write_lock = Lock()
    new_episodes_count = 0
    skipped_episodes_count = len(annotations) - len(tasks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_single_episode,
                episode_data,
                idx,
                total,
                processed_episode_keys,
                output_file,
                write_lock
            )
            for episode_data, idx, total in tasks
        ]
        
        for fut in as_completed(futures):
            try:
                episode_key, success, skipped = fut.result()
                if success:
                    new_episodes_count += 1
                elif skipped:
                    skipped_episodes_count += 1
            except Exception as e:
                print(f"Error processing episode: {e}")
    
    print(f"\n{'='*80}")
    print(f"Completed!")
    print(f"  Previously processed: {len(processed_episode_keys)} episodes")
    print(f"  Newly processed: {new_episodes_count} episodes")
    print(f"  Skipped: {skipped_episodes_count} episodes")
    print(f"  Total in file: {len(processed_episode_keys) + new_episodes_count} episodes")
    print(f"Subtask splits saved to: {output_file}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R", 
                       help="Path to trajectory data directory")
    parser.add_argument("--output_file", type=str, default="data/subtask_splits/R2R/subtask_splits.jsonl",
                       help="Output file for subtask splits (JSONL format)")
    parser.add_argument("--max_workers", type=int, default=8,
                       help="Maximum number of concurrent workers")
    args = parser.parse_args()
    
    process_episodes_for_subtask_split(args.trajectory_dir, args.output_file, args.max_workers)

if __name__ == "__main__":
    main()
