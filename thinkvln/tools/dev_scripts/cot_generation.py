import os
import json
import cv2
import base64
import argparse
import re
import shutil
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
)

# ==============================================================================
# 1. STATIC SYSTEM PROMPT (NO VARIABLES)
# ==============================================================================
ACTION_JUSTIFICATION_PROMPT = r"""### Role & Objective
You are an expert Visual-Language Navigation (VLN) agent acting as a **Defense Attorney** for the Ground Truth.
You will be provided with the **Ground Truth (GT) Subtask** and **GT Action** for the current step in the User Message.
Your SOLE objective is to **construct a coherent causal chain** that proves the GT Action is the *only* logical choice based on visual evidence, even if it seems counter-intuitive.
**Constraint:** Your reasoning must be extremely **CONCISE** (High Information Density). 
**Rule:** Limit each section to **1-2 sentences**. No filler words. As short as possible.

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

def get_stepid_from_framepath(frame_path: str) -> int:
    try:
        return int(frame_path.split("/")[-1].split(".")[0].split("_")[-1])
    except:
        return 0

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
        # Format: [cur subtask] 1 (Description)
        pattern = r"\[cur subtask\]\s*(\d+)"
        match = re.search(pattern, cot_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Warning: Error extracting subtask index: {e}")
    return -1

# ==============================================================================
# 3. GENERATION LOGIC
# ==============================================================================

def generate_cot_for_step(instruction: str, cur_frame_path: str, 
                          plan: str, action: int, step_id: int, max_steps: int, 
                          previous_subtask_index: int = -1, previous_action: int = None, 
                          ground_truth_subtask: int = None) -> dict:
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
        
        # Labels
        action_names = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
        gt_action_name = action_names.get(action, "unknown")
        previous_action_name = action_names.get(previous_action, "unknown") if previous_action is not None else "N/A"
        
        prev_subtask_name = extract_subtask_name_from_plan(plan, previous_subtask_index) if previous_subtask_index >= 1 else "Start"
        target_subtask_name = extract_subtask_name_from_plan(plan, ground_truth_subtask) if ground_truth_subtask >= 1 else "Unknown"
        
        # 2. Construct Dynamic User Message (NO HISTORY)
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
        
        # 3. Construct Payload (ONLY CURRENT IMAGE, NO HISTORY)
        image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_image_base64}"}}]
        
        # 4. API Call
        response = client.chat.completions.create(
            model="qwen/qwen3-vl-30b-a3b-instruct", 
            messages=[
                {"role": "system", "content": ACTION_JUSTIFICATION_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": full_user_content}] + image_contents}
            ],
            temperature=0.7
        )
        
        cot_text = response.choices[0].message.content
        
        return {
            "step_id": step_id,
            "action": action,
            "ground_truth_action": action,
            "ground_truth_subtask": ground_truth_subtask,
            "cot": cot_text
        }
    
    except Exception as e:
        print(f"Error generating CoT for step {step_id}: {e}")
        return None

# ==============================================================================
# 4. MAIN PROCESS LOOP
# ==============================================================================

def process_episode(episode_data: dict, video_dir: str, aggregate_file: str) -> int:
    """Process episode and append frames to aggregate JSONL file. Returns count of processed frames."""
    episode_id = episode_data.get("episode_id") or episode_data.get("id")
    instruction = episode_data.get("instruction") or (episode_data["instructions"][0] if isinstance(episode_data.get("instructions"), list) else episode_data.get("instructions", ""))
    actions = episode_data["actions"]
    
    # Format Plan
    plan = episode_data.get("plan", [])
    if isinstance(plan, list):
        plan = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)]) if plan else ""
    
    # Keyframes & Sequence
    keyframes = episode_data.get("keyframes", [])
    if not keyframes: keyframes = list(range(len(actions)))
    
    subtask_sequence = episode_data.get("subtask_sequence", [])
    if not subtask_sequence: subtask_sequence = [1] * len(actions)
    
    # Pad Sequence
    if len(subtask_sequence) < len(actions):
        subtask_sequence.extend([subtask_sequence[-1]] * (len(actions) - len(subtask_sequence)))
    
    # Extract Frames
    video_path = os.path.join(video_dir, "trajectory.mp4")
    if not os.path.exists(video_path):
        print(f"Video missing: {video_path}")
        return 0
        
    frame_paths = extract_frames_from_video(video_path, episode_id)
    if len(frame_paths) != len(actions):
        print(f"Mismatch: {len(frame_paths)} frames vs {len(actions)} actions")
        return 0
    
    scene_id = episode_data.get("scene_id")
    episode_key = episode_data.get("episode_key") or f"{scene_id}_{episode_id}"
    
    # Loop
    keyframe_indices = set(keyframes)
    print(f"Processing {len(keyframe_indices)} keyframes for Ep {episode_id}")
    
    processed_count = 0
    max_steps = len(frame_paths)
    previous_subtask_index = -1
    previous_action = None
    
    for step_id in sorted(keyframe_indices):
        if step_id >= len(frame_paths): continue
        
        frame_path = frame_paths[step_id]
        action = actions[step_id]
        gt_subtask = subtask_sequence[step_id]
        
        print(f"Step {step_id}: GT Subtask {gt_subtask} | Prev {previous_subtask_index}")
        
        cot_result = generate_cot_for_step(
            instruction, frame_path, plan, action, step_id, max_steps,
            previous_subtask_index=previous_subtask_index,
            previous_action=previous_action,
            ground_truth_subtask=gt_subtask
        )
        
        if cot_result:
            # Create frame_key combining episode_key and step_id
            frame_key = f"{episode_key}_{step_id:06d}"
            
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
            
            # Append to JSONL file (one line per frame)
            with open(aggregate_file, "a") as f:
                json.dump(frame_record, f)
                f.write("\n")
            
            processed_count += 1
            
            # Update State
            extracted_idx = extract_subtask_index_from_cot(cot_result["cot"])
            previous_subtask_index = extracted_idx if extracted_idx != -1 else gt_subtask
            previous_action = action
        else:
            break
         
    # Cleanup
    tmp_dir = os.path.join("/tmp", f"cot_frames_{episode_id}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
     
    return processed_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R")
    parser.add_argument("--output_dir", type=str, default="data/cot_dataset/R2R")
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
     
    os.makedirs(args.output_dir, exist_ok=True)
    annotation_file = os.path.join(args.trajectory_dir, "summary_full.jsonl")
    
    if not os.path.exists(annotation_file):
        print("Annotation file missing.")
        return
        
    annotations = []
    with open(annotation_file, "r") as f:
        for line in f:
            if line.strip(): annotations.append(json.loads(line))
            
    total_frames = 0
    aggregate_file = os.path.join(args.output_dir, "cot_dataset.jsonl")
    
    for idx, ep_data in enumerate(annotations):
        if args.max_episodes and idx >= args.max_episodes: break
        
        episode_id = ep_data.get('episode_id')
        print(f"\n[{idx+1}/{len(annotations)}] Processing Episode {episode_id}")
        
        frame_count = process_episode(
            ep_data, 
            os.path.join(args.trajectory_dir, ep_data["video"]), 
            aggregate_file
        )
        
        total_frames += frame_count
        print(f"  → Processed {frame_count} frames")

    print(f"\n✓ Complete. Total frames: {total_frames}")
    print(f"Output: {aggregate_file}")

if __name__ == "__main__":
    main()
