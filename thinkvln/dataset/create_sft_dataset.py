#!/usr/bin/env python3
"""
Create SFT training dataset in mllm_demo.json format from COT data and extracted frames.
Combines instruction, plan, image, and answer into LLaMA-Factory compatible format.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import argparse


# Two versions of prompts - simple placeholders
PROMPT_WITH_TABLES = """### Role & Objective
You are a Visual-Language Navigation (VLN) agent. Your task is to generate a coherent reasoning chain for navigation.

### Input Definition
**You will receive:**
- **instruction**: Natural language navigation instruction
- **plan**: Step-by-step decomposition of the instruction
- **image**: A composite image showing:
  - **LEFT SIDE**: RGB first-person view from the robot's camera
  - **RIGHT SIDE**: Top-down occupancy map with robot position (blue arrow), goal (red square), paths, and obstacles

### Structured Knowledge Base
**Table 1: Navigation Meta-Actions (Transition Logic)**
- **Turn**: Rotating until orientation aligns with new path
- **Region Transition**: Moving to cross threshold on the map
- **Visual Approach**: Moving to object until it's immediate (high pixel ratio)
- **General Cruise**: Moving down hall until reaching structural event

**Table 2: Critical Observation Components**
- **Target Objects**: Type and relative pose - if target seen, approach it
- **Structural Features**: Doorways and intersections - if doorway ahead, prepare for transition
- **Path Constraints**: Obstacles and walls - if path blocked, deviation needed

**Table 3: Action Alignment Rules**
- **Preparation**: Turn action may need forward first to reach geometric center
- **Obstacle**: Forward action may need turn first to avoid obstacle
- **Counter-Flow**: Right turn may need left turn first to widen turning radius

### Output Format
[localization]
Describe where you are structurally and visually based on the RGB and map.

[subtask determination]
 [prev subtask] <Index> (<Description>)
 [transition check] Compare previous vs current subtask. Explain trigger events or why not triggered.
 [cur subtask] <Index> (<Description>)

[causal observation]
Identify critical objects, structural features, or path constraints affecting the action.

[reason]
Explain why the next action makes sense given the current situation.

[action]
Choose one of: forward, turn_left, turn_right, stop
"""

PROMPT_SIMPLE = """### Role & Objective
You are a Visual-Language Navigation (VLN) agent. Generate a reasoning chain for the next navigation action.

### Input Definition
**You will receive:**
- **instruction**: Natural language navigation instruction
- **plan**: Step-by-step decomposition of the instruction
- **image**: A composite image with RGB view (left) and top-down map (right)

### Your Task
Analyze the current situation and determine the next action considering:
- Your current location and orientation
- Obstacles or structural features ahead
- Current navigation subtask
- Whether subtask has changed

### Output Format
[localization]
Describe your current position and what you see (RGB view and map).

[subtask determination]
 [prev subtask] Previous subtask index and description
 [transition check] Has the subtask changed? If yes, why? If no, why not?
 [cur subtask] Current subtask index and description

[causal observation]
Identify critical objects, structural features, or path constraints affecting the next action.

[reason]
Explain why the next action makes sense given the current situation.

[action]
Choose one of: forward, turn_left, turn_right, stop
"""


def load_cot_dataset(cot_file: str) -> List[Dict]:
    """Load COT answer dataset from JSONL file."""
    data = []
    with open(cot_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_frame_mapping(mapping_file: str) -> Dict[str, str]:
    """Load frame mapping from extracted frames."""
    mapping = {}
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            frames = json.load(f)
            for frame_info in frames:
                frame_key = frame_info.get('frame_key')
                image_path = frame_info.get('image_path')
                if frame_key and image_path:
                    mapping[frame_key] = image_path
    except Exception as e:
        print(f"Warning: Could not load frame mapping: {e}")
    return mapping


def action_id_to_name(action_id: int) -> str:
    """Convert action ID to action name.
    
    Args:
        action_id: 1=forward, 2=turn_left, 3=turn_right, 4=stop
    
    Returns:
        Action name string
    """
    action_map = {
        1: "forward",
        2: "turn_left",
        3: "turn_right",
        4: "stop"
    }
    return action_map.get(action_id, "unknown")


def extract_prev_subtask_from_answer(answer: str) -> Optional[str]:
    """Extract [prev subtask] information from answer text.
    
    Args:
        answer: The answer text containing reasoning chain
        
    Returns:
        Previous subtask string (e.g., "4 (Enter the bathroom.)") or None
    """
    if not answer:
        return None
    
    # Look for [prev subtask] or [prev subtask] pattern
    pattern = r'\[prev subtask\]\s*(\d+)\s*\(([^)]+)\)'
    match = re.search(pattern, answer, re.IGNORECASE)
    
    if match:
        subtask_index = match.group(1)
        subtask_desc = match.group(2).strip()
        return f"{subtask_index} ({subtask_desc})"
    
    return None


def build_user_message(instruction: str, plan: str, prev_subtask: Optional[str] = None, prompt_version: str = "simple") -> str:
    """Build the user message combining instruction, plan, and prompt.
    
    Args:
        instruction: Navigation instruction text
        plan: Step-by-step plan
        prev_subtask: Previous subtask information (e.g., "4 (Enter the bathroom.)")
        prompt_version: "simple", "with_tables", or "clean" (no prompt)
    
    Returns:
        Formatted user message string
    """
    
    # Clean version: no prompt, just instruction and plan
    if prompt_version == "clean":
        prev_subtask_section = ""
        if prev_subtask:
            prev_subtask_section = f"\n**Previous Subtask**: {prev_subtask}"
        
        user_message = f"""<image>
**Instruction**: {instruction}

**Plan**: {plan}{prev_subtask_section}"""
        return user_message
    
    # Select prompt version
    if prompt_version == "with_tables":
        prompt = PROMPT_WITH_TABLES
    else:
        prompt = PROMPT_SIMPLE
    
    # Format plan as numbered list if it's a string
    if isinstance(plan, str):
        plan_text = plan
    else:
        plan_text = str(plan)
    
    # Build the user message with instruction, plan, and prev_subtask
    prev_subtask_section = ""
    if prev_subtask:
        prev_subtask_section = f"\n**Previous Subtask**: {prev_subtask}"
    
    user_message = f"""<image>{prompt}

### Current Task Instance
**Instruction**: {instruction}

**Plan**: {plan_text}{prev_subtask_section}

Analyze the image and generate your reasoning chain following the output format."""
    
    return user_message


def create_sft_dataset(
    cot_file: str,
    frame_mapping_file: str,
    output_file: str,
    prompt_version: str = "simple"
) -> Dict:
    """
    Create SFT dataset in mllm_demo.json format.
    
    Args:
        cot_file: Path to COT answer JSONL file
        frame_mapping_file: Path to frame mapping JSON file
        output_file: Path to output mllm_demo.json file
        prompt_version: "simple" or "with_tables"
    
    Returns:
        Statistics dictionary
    """
    
    print(f"Loading COT dataset from {cot_file}...")
    cot_data = load_cot_dataset(cot_file)
    print(f"Loaded {len(cot_data)} COT records")
    
    print(f"Loading frame mapping from {frame_mapping_file}...")
    frame_mapping = load_frame_mapping(frame_mapping_file)
    print(f"Loaded {len(frame_mapping)} frame mappings")
    
    # Build dataset
    dataset = []
    stats = {
        'total': len(cot_data),
        'success': 0,
        'missing_image': 0,
        'missing_answer': 0,
        'errors': 0
    }
    
    for idx, record in enumerate(cot_data, 1):
        try:
            frame_key = record.get('frame_key')
            instruction = record.get('instruction', '')
            plan = record.get('plan', '')
            answer = record.get('answer')
            action_id = record.get('ground_truth_action')
            
            # Check if answer exists
            if not answer:
                stats['missing_answer'] += 1
                print(f"[{idx}/{len(cot_data)}] Skipped: no answer for frame {frame_key}")
                continue
            
            # Check if frame mapping exists
            if frame_key not in frame_mapping:
                stats['missing_image'] += 1
                print(f"[{idx}/{len(cot_data)}] Skipped: no image mapping for frame {frame_key}")
                continue
            
            image_path = frame_mapping[frame_key]
            
            # Check if image file exists
            if not os.path.exists(image_path):
                stats['missing_image'] += 1
                print(f"[{idx}/{len(cot_data)}] Skipped: image file not found at {image_path}")
                continue
            
            # Extract previous subtask from answer
            prev_subtask = extract_prev_subtask_from_answer(answer)
            
            # Build user message with prev_subtask information
            user_message = build_user_message(instruction, plan, prev_subtask, prompt_version)
            
            # Create dataset entry
            entry = {
                "messages": [
                    {
                        "content": user_message,
                        "role": "user"
                    },
                    {
                        "content": answer,
                        "role": "assistant"
                    }
                ],
                "images": [image_path]
            }
            
            dataset.append(entry)
            stats['success'] += 1
            
            if idx % 10 == 0:
                print(f"[{idx}/{len(cot_data)}] Progress: {stats['success']} processed")
        
        except Exception as e:
            stats['errors'] += 1
            print(f"[{idx}/{len(cot_data)}] Error: {e}")
            continue
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Dataset Creation Summary:")
    print(f"  Total records: {stats['total']}")
    print(f"  Successfully processed: {stats['success']}")
    print(f"  Missing images: {stats['missing_image']}")
    print(f"  Missing answers: {stats['missing_answer']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Output file: {output_file}")
    print(f"="*60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Create SFT dataset in mllm_demo.json format'
    )
    parser.add_argument(
        '--cot-file',
        type=str,
        default='/mnt/swx/ThinkVLN/data/cot_dataset/cot_dataset_answer.jsonl',
        help='Path to COT answer JSONL file'
    )
    parser.add_argument(
        '--frame-mapping-file',
        type=str,
        default='/mnt/swx/dataset/sft-dataset/frame_mapping.json',
        help='Path to frame mapping JSON file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/mnt/swx/ThinkVLN/data/cot_dataset/sft_dataset_v2.json',
        help='Output file path for SFT dataset'
    )
    parser.add_argument(
        '--prompt-version',
        type=str,
        choices=['simple', 'with_tables', 'clean'],
        default='simple',
        help='Prompt version to use: simple (default), with_tables, or clean (no prompt, just instruction and plan)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.cot_file):
        print(f"Error: COT file not found: {args.cot_file}")
        return
    
    if not os.path.exists(args.frame_mapping_file):
        print(f"Error: Frame mapping file not found: {args.frame_mapping_file}")
        print(f"Please run extract_frames_for_sft.py first to generate frame mappings")
        return
    
    # Create dataset
    create_sft_dataset(
        args.cot_file,
        args.frame_mapping_file,
        args.output_file,
        args.prompt_version
    )


if __name__ == '__main__':
    main()

