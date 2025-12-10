import os
import json
import cv2
import base64
import argparse
import re
import shutil
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Initialize OpenAI client
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ.get("OPENROUTER_API_KEY", "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"),
# )
client = OpenAI(
    base_url="http://localhost:11451/v1",
    api_key="EMPTY"
)

# ==============================================================================
# 1. STATIC SYSTEM PROMPT (NO VARIABLES)
# ==============================================================================
ACTION_JUSTIFICATION_PROMPT = r"""### Role & Objective
You are an expert Visual-Language Navigation (VLN) agent acting as a **Defense Attorney** for the Ground Truth.
You will be provided with the **Ground Truth (GT) Subtask** and **GT Action** for the current step in the User Message.
Your SOLE objective is to **construct a coherent causal chain** that proves the GT Action is the *only* logical choice based on visual evidence, even if it seems counter-intuitive.

### 0. The "No Critique" Rule
* **NEVER** suggest the GT Action is wrong, inconsistent, or a data error.
* **NEVER** say "Although the plan says X, the robot does Y."
* **INSTEAD**, invent a valid micro-reason (see Table 3).

### 1. Perception & Realism Constraints
* **Visual Source (RGB)**: Use this for object identification and depth perception.
* **Geometric Source (Top-Down Map)**: Use this for orientation and structural boundaries.
* **STRICT REALISM RULE**: **IGNORE** the Green Line (GT Path) and Red Square (Goal) on the map. Reason only using your current position (Blue Arrow) and the visible environment.

### 2. Structured Knowledge Base
**Table 1: Navigation Meta-Actions (Transition Logic)**
| Meta-Action | Current Behavior | **Next Task Trigger** (When to Switch) |
| :--- | :--- | :--- |
| **Turn** | Rotating. | **Switch when**: Orientation aligns with new path (90°/180° done). |
| **Region Transition** | Moving to door. | **Switch when**: Physically **crossing the threshold** on the map. |
| **Visual Approach** | Moving to object. | **Switch when**: Object is **immediate** (high pixel ratio) or robot stops. |
| **General Cruise** | Moving down hall. | **Switch when**: Reaching a structural event (intersection/end of hall). |

**Table 2: Critical Observation Components**
| Category | Attributes to Record | Decision Logic |
| :--- | :--- | :--- |
| **Target Objects** | Type, Relative Pose. | If target seen -> Approach. |
| **Structural Features** | Doorways, Intersections. | If doorway ahead -> Prepare for Region Transition. |
| **Path Constraints** | Obstacles, Wall boundaries. | If path blocked -> Deviation needed. |

**Table 3: Action Alignment Rules (Micro-Movement Justification)**
| Context | Discrepancy | **Reasoning Strategy (The "Why")** |
| :--- | :--- | :--- |
| **Preparation** | Task="Turn", Action="Forward" | "I must move forward to reach the geometric center of the intersection before turning." |
| **Obstacle** | Task="Forward", Action="Turn" | "I am micro-adjusting my trajectory to avoid an obstacle/wall." |
| **Counter-Flow** | Task="Right", Action="Left" | "I am steering left to widen my turning radius or avoid clipping the corner on the right." |

### 3. Output Format
Generate your response strictly following this structure. Use natural sentences.

[localization]
(Describe where you are structurally and visually. E.g., "I am in the middle of a long corridor...")

[subtask determination]
 [prev subtask] <Index> (<Description>)
 [transition check] (Use **Table 1**. Compare Previous vs. GT Subtask.
  - If Index Changed: Describe the **Trigger Event** (e.g., "Crossed threshold").
  - If Index Same: Explain why Trigger is NOT met (e.g., "Still approaching pivot point").)
 [cur subtask] <GT Index> (<GT Description>)

[causal observation]
(Use **Table 2**. Highlight the specific **Critical Object** or **Structural Feature** that dictates the current action.)

[reason]
(Synthesize logic. **CRITICAL**: Use **Table 3** if GT Action seems to contradict the plan. E.g., "Although the plan is to veer right, I must momentarily turn left to clear the doorframe.")

[action]
<GT Action Name>
"""

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def extract_subtask_name_from_plan(plan: str, subtask_index: int) -> str:
    if not plan or subtask_index < 1:
        return ""
    lines = plan.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith(f"{subtask_index}. "):
            return line[len(f"{subtask_index}. "):].strip()
    return ""

def frame_to_base64(frame_path: str) -> str:
    try:
        with open(frame_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error converting frame {frame_path} to base64: {e}")
        return ""

def extract_frames_from_video(video_path: str, episode_id: int) -> list:
    tmp_frame_dir = os.path.join("/tmp", f"cot_frames_{episode_id}")
    os.makedirs(tmp_frame_dir, exist_ok=True)
    
    # Clear existing
    for filename in os.listdir(tmp_frame_dir):
        file_path = os.path.join(tmp_frame_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(tmp_frame_dir, f"frame_{frame_idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1
    cap.release()
    return frame_paths

def extract_subtask_index_from_cot(cot_text: str) -> int:
    """Extract subtask index from CoT text using the prompt's output format."""
    try:
        pattern = r"\[cur subtask\]\s*(\d+)"
        match = re.search(pattern, cot_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Warning: Error extracting subtask index: {e}")
    return -1

# ==============================================================================
# 3. GENERATION LOGIC WITH RETRY
# ==============================================================================

def generate_cot_for_step(instruction: str, cur_frame_path: str, 
                          plan: str, action: int, step_id: int, max_steps: int, 
                          previous_subtask_index: int = -1, previous_action: int = None, 
                          ground_truth_subtask: int = None,
                          max_retries: int = 3) -> dict:
    """Generate CoT with retry mechanism."""
    try:
        if step_id == 0:
            return {
                "step_id": step_id,
                "action": action,
                "ground_truth_action": action,
                "ground_truth_subtask": ground_truth_subtask,
                "cot": ""
            }
            
        # 1. Prepare Data
        cur_image_base64 = frame_to_base64(cur_frame_path)
        if not cur_image_base64:
            return None
        
        # Labels
        action_names = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
        gt_action_name = action_names.get(action, "unknown")
        previous_action_name = action_names.get(previous_action, "unknown") if previous_action is not None else "N/A"
        
        prev_subtask_name = extract_subtask_name_from_plan(plan, previous_subtask_index) if previous_subtask_index >= 1 else "Start"
        target_subtask_name = extract_subtask_name_from_plan(plan, ground_truth_subtask) if ground_truth_subtask >= 1 else "Unknown"
        
        # 2. Construct Dynamic User Message
        full_user_content = f"""
**CURRENT EPISODE CONTEXT**
* **Instruction**: "{instruction}"
* **High-Level Plan**:
{plan}

**CURRENT STATE (Step {step_id}/{max_steps})**
* **History Action**: {previous_action_name}
* **Previous Subtask**: {previous_subtask_index} ({prev_subtask_name})

**GROUND TRUTH TARGETS (Please Justify These)**
* **Target GT Subtask**: {ground_truth_subtask} ({target_subtask_name})
* **Target GT Action**: {gt_action_name}
"""
        
        # 3. Construct Payload
        image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_image_base64}"}}]
        
        # 4. API Call with Retry
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    # model="qwen/qwen3-vl-30b-a3b-instruct", 
                    model='Qwen/Qwen3-VL-235B-A22B-Thinking',
                    messages=[
                        {"role": "system", "content": ACTION_JUSTIFICATION_PROMPT},
                        {"role": "user", "content": [{"type": "text", "text": full_user_content}] + image_contents}
                    ],
                    temperature=0.7
                )
                
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    cot_text = response.choices[0].message.content
                    if cot_text:
                        return {
                            "step_id": step_id,
                            "action": action,
                            "ground_truth_action": action,
                            "ground_truth_subtask": ground_truth_subtask,
                            "cot": cot_text
                        }
                
                if attempt < max_retries:
                    print(f"  Retry {attempt}/{max_retries}: Empty response, retrying...")
                    continue
            
            except Exception as e:
                print(f"  Retry {attempt}/{max_retries}: {str(e)}")
                if attempt < max_retries:
                    continue
        
        return None
    
    except Exception as e:
        print(f"Error generating CoT for step {step_id}: {e}")
        return None

# ==============================================================================
# 4. PROCESS SINGLE FRAME (THREAD-SAFE)
# ==============================================================================

def process_single_frame(episode_key: str, step_id: int, episode_id: int, 
                         scene_id: str, instruction: str, plan: str,
                         frame_path: str, action: int, gt_subtask: int,
                         previous_subtask_index: int, previous_action: int,
                         max_steps: int, aggregate_file: str, write_lock: Lock,
                         processed_frame_keys: set) -> tuple:
    """Process a single frame and append to output file (thread-safe).
    
    Returns:
        (frame_key, status, result_dict or error)
    """
    frame_key = f"{episode_key}_{step_id:06d}"
    
    # Check if already processed (thread-safe)
    with write_lock:
        if frame_key in processed_frame_keys:
            return (frame_key, "skipped", "already processed")
    
    try:
        # Generate CoT
        cot_result = generate_cot_for_step(
            instruction, frame_path, plan, action, step_id, max_steps,
            previous_subtask_index=previous_subtask_index,
            previous_action=previous_action,
            ground_truth_subtask=gt_subtask,
            max_retries=3
        )
        
        if not cot_result:
            return (frame_key, "failed", "CoT generation failed after retries")
        
        # Create frame record
        frame_record = {
            "frame_key": frame_key,
            "episode_key": episode_key,
            "episode_id": episode_id,
            "scene_id": scene_id,
            "instruction": instruction,
            "plan": plan,
            "step_id": step_id,
            **cot_result
        }
        
        # Append to JSONL file (thread-safe)
        with write_lock:
            with open(aggregate_file, "a") as f:
                json.dump(frame_record, f)
                f.write("\n")
                f.flush()
            processed_frame_keys.add(frame_key)
        
        return (frame_key, "success", frame_record)
    
    except Exception as e:
        return (frame_key, "failed", str(e))

# ==============================================================================
# 5. MAIN PROCESS LOOP WITH CONCURRENCY
# ==============================================================================

def process_episode_concurrent(episode_data: dict, video_dir: str, aggregate_file: str,
                              write_lock: Lock, processed_frame_keys: set, max_workers: int = 4) -> int:
    """Process episode with concurrent frame processing. Returns count of processed frames.
    
    Frames are independent - state (previous_subtask_index, previous_action) is retrieved
    directly from keyframe indices and subtask_sequence array, not from saved records."""
    episode_id = episode_data.get("episode_id") or episode_data.get("id")
    # Note: 'instructions' field has been removed; use 'instruction' only
    instruction = episode_data.get("instruction", "")
    if not instruction:
        print(f"  Warning: No instruction found for episode {episode_id}")
        return 0
    actions = episode_data["actions"]
    
    # Format Plan
    plan = episode_data.get("plan", [])
    if isinstance(plan, list):
        plan = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)]) if plan else ""
    
    # Keyframes & Sequence
    keyframes = episode_data.get("keyframes", [])
    if not keyframes: 
        keyframes = list(range(len(actions)))
    
    subtask_sequence = episode_data.get("subtask_sequence", [])
    if not subtask_sequence: 
        subtask_sequence = [1] * len(actions)
    
    # Pad Sequence
    if len(subtask_sequence) < len(actions):
        subtask_sequence.extend([subtask_sequence[-1]] * (len(actions) - len(subtask_sequence)))
    
    # Extract Frames
    video_path = os.path.join(video_dir, "trajectory.mp4")
    if not os.path.exists(video_path):
        print(f"  Video missing: {video_path}")
        return 0
        
    frame_paths = extract_frames_from_video(video_path, episode_id)
    if len(frame_paths) != len(actions):
        print(f"  Mismatch: {len(frame_paths)} frames vs {len(actions)} actions")
        return 0
    
    scene_id = episode_data.get("scene_id")
    episode_key = episode_data.get("episode_key") or f"{scene_id}_{episode_id}"
    
    # Get keyframe indices
    keyframe_indices = sorted(set(keyframes))
    print(f"  Processing {len(keyframe_indices)} keyframes for Ep {episode_id} (workers={max_workers})")
    
    processed_count = 0
    max_steps = len(frame_paths)
    
    # Process frames concurrently (frames are independent)
    # Get previous state from keyframe indices (direct lookup from subtask_sequence, no saved records needed)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, step_id in enumerate(keyframe_indices):
            if step_id >= len(frame_paths):
                continue
            
            frame_key = f"{episode_key}_{step_id:06d}"
            
            # Skip if already processed
            if frame_key in processed_frame_keys:
                print(f"    [{idx+1}/{len(keyframe_indices)}] Frame {frame_key}: skipped")
                continue
            
            # Get previous state directly from subtask_sequence (no saved records needed)
            previous_subtask_index = -1
            previous_action = None
            
            # Find the previous keyframe index
            current_idx = keyframe_indices.index(step_id)
            if current_idx > 0:
                prev_keyframe_step_id = keyframe_indices[current_idx - 1]
                # Get previous subtask and action from the ground truth sequence array
                previous_subtask_index = subtask_sequence[prev_keyframe_step_id]
                previous_action = actions[prev_keyframe_step_id]
            
            frame_path = frame_paths[step_id]
            action = actions[step_id]
            gt_subtask = subtask_sequence[step_id]
            
            future = executor.submit(
                process_single_frame,
                episode_key, step_id, episode_id, scene_id, instruction, plan,
                frame_path, action, gt_subtask, previous_subtask_index, previous_action,
                max_steps, aggregate_file, write_lock, processed_frame_keys
            )
            futures[future] = (idx + 1, frame_key, len(keyframe_indices), previous_subtask_index)
        
        # Process completed futures
        for future in as_completed(futures):
            idx, frame_key, total, prev_subtask = futures[future]
            try:
                frame_key_result, status, result = future.result()
                
                if status == "success":
                    processed_count += 1
                    print(f"    [{idx}/{total}] Frame {frame_key}: ✓ success (prev_subtask={prev_subtask})")
                elif status == "skipped":
                    print(f"    [{idx}/{total}] Frame {frame_key}: - skipped")
                else:
                    print(f"    [{idx}/{total}] Frame {frame_key}: ✗ failed - {result}")
            except Exception as e:
                print(f"    [{idx}/{total}] Frame {frame_key}: ✗ error - {str(e)}")
    
    # Cleanup
    tmp_dir = os.path.join("/tmp", f"cot_frames_{episode_id}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
     
    return processed_count

# ==============================================================================
# 6. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deploy version: CoT generation with concurrency and resume")
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R",
                       help="Path to trajectory data directory")
    parser.add_argument("--annotation_file", type=str, default="data/trajectory_data/R2R/summary_full.jsonl",
                       help="Path to annotation file")
    parser.add_argument("--output_file", type=str, default="data/cot_dataset/R2R/cot_dataset.jsonl",
                       help="Output JSONL file (one line per frame)")
    parser.add_argument("--max_episodes", type=int, default=None,
                       help="Maximum number of episodes to process")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of concurrent workers for frame processing (default: 4)")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    if not os.path.exists(args.annotation_file):
        print(f"Annotation file missing: {args.annotation_file}")
        return
    
    # Load annotations
    annotations = []
    with open(args.annotation_file, "r") as f:
        for line in f:
            if line.strip(): 
                annotations.append(json.loads(line))
    
    print(f"Loaded {len(annotations)} episodes")
    print(f"Using {args.max_workers} concurrent workers")
    
    # Load processed frame keys for resuming
    processed_frame_keys = set()
    if os.path.exists(args.output_file):
        print(f"Loading processed frames from {args.output_file}...")
        with open(args.output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        frame_key = data.get("frame_key")
                        if frame_key:
                            processed_frame_keys.add(frame_key)
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_frame_keys)} already processed frames")
    
    # Process episodes
    write_lock = Lock()
    total_frames = 0
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, ep_data in enumerate(annotations):
        if args.max_episodes and idx >= args.max_episodes: 
            break
        
        episode_id = ep_data.get('episode_id')
        print(f"\n[{idx+1}/{len(annotations)}] Processing Episode {episode_id}")
        
        try:
            frame_count = process_episode_concurrent(
                ep_data, 
                os.path.join(args.trajectory_dir, ep_data["video"]), 
                args.output_file,
                write_lock,
                processed_frame_keys,
                max_workers=args.max_workers
            )
            
            total_frames += frame_count
            if frame_count > 0:
                successful += 1
            elif frame_count == 0:
                skipped += 1
            
            print(f"  → Processed {frame_count} frames")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total episodes: {len(annotations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total frames processed: {total_frames}")
    print(f"Concurrent workers: {args.max_workers}")
    print(f"Output: {args.output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

