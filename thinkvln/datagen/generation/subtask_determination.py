import os
import json
import re
import base64
import cv2
import numpy as np
import argparse
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ.get("OPENROUTER_API_KEY", "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"),
# )

# # # Initialize OpenAI client
client = OpenAI(
    base_url="http://localhost:11451/v1",
    api_key="EMPTY"
)

def identify_action_change_keyframes(actions: List[int], window_size: int = 3) -> List[int]:
    """Identify keyframes where actions change using sliding window to filter noise."""
    if len(actions) < 2:
        return [0] if len(actions) > 0 else []

    actions = actions[1:]
    keyframes = [0]
    
    for i in range(1, len(actions)):
        if i == len(actions) - 1:
            keyframes.append(i)
            continue
        is_change = False
        start_idx = max(0, i - window_size)
        end_idx = i
        prev_actions = actions[start_idx:end_idx]
        prev_mode = max(set(prev_actions), key=prev_actions.count) if prev_actions else actions[i-1]
        curr_action = actions[i]
        
        if curr_action != prev_mode:
            if i + window_size <= len(actions):
                future_actions = actions[i:min(i + window_size, len(actions))]
                future_mode = max(set(future_actions), key=future_actions.count) if future_actions else curr_action
                if curr_action == future_mode or (i + 1 < len(actions) and actions[i+1] == curr_action):
                    is_change = True
            else:
                is_change = True
        
        if is_change and i not in keyframes:
            keyframes.append(i)

    
    
    min_spacing = 5
    result_keyframes = [keyframes[0]]
    i = 1
    while i < len(keyframes):
        last_kf = result_keyframes[-1]
        kf = keyframes[i]
        if kf - last_kf >= min_spacing:
            next_kf = last_kf + min_spacing
            while next_kf < kf:
                result_keyframes.append(next_kf)
                next_kf += min_spacing
            result_keyframes.append(kf)
        else:
            result_keyframes.append(kf)
        i += 1

    if len(actions) - 1 not in result_keyframes:
        result_keyframes.append(len(actions) - 1)

    # Final: ensure no gaps larger than min_spacing remain
    i = 0
    while i < len(result_keyframes) - 1:
        if result_keyframes[i + 1] - result_keyframes[i] > min_spacing:
            # Insert keyframe to fill gap
            result_keyframes.insert(i + 1, result_keyframes[i] + min_spacing)
        i += 1

    return sorted(set(result_keyframes))


def downsample_frame(frame_path: str, target_size: tuple = (768, 240)) -> str:
    """Downsample an image frame to reduce resolution for VLM processing."""
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            return ""
        downsampled = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', downsampled, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.standard_b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"Error downsampling frame {frame_path}: {e}")
        return ""


def preprocess_first_subtask(plan: List[str]) -> List[str]:
    """
    Preprocess the first subtask by adding "Turn around and" prefix.
    
    Args:
        plan: List of subtask descriptions
        
    Returns:
        Modified plan with first subtask prefixed with "Turn around and "
    """
    if not plan or len(plan) == 0:
        return plan
    
    # Create a copy to avoid modifying the original
    modified_plan = plan.copy()
    
    # Add "Turn around and " prefix to the first subtask
    first_subtask = modified_plan[0].strip()
    modified_plan[0] = "Turn around and " + first_subtask
    
    return modified_plan


def build_transition_point_prompt(
    instruction: str,
    plan: List[str],
    num_frames: int,
    num_keyframes: int
) -> str:
    """
    Build the prompt for VLM to identify turning points (transition points between subtasks).
    
    Args:
        instruction: The original navigation instruction
        plan: List of subtask descriptions
        num_frames: Total number of frames in the trajectory
        num_keyframes: Number of keyframes being shown
        
    Returns:
        Complete prompt string
    """
    # Preprocess first subtask
    processed_plan = preprocess_first_subtask(plan)
    
    # Format plan for prompt
    plan_str = "\n".join([f"{i+1}. {subtask}" for i, subtask in enumerate(processed_plan)])
    num_subtasks = len(plan)
    
    prompt = f"""You are an expert AI assistant specializing in Vision-and-Language Navigation (VLN). Your task is to analyze a sequence of keyframes from a navigation trajectory and identify turning points where subtasks transition.

**CONTEXT:**
- **Overall Goal (Instruction):** "{instruction}"
- **High-Level Plan (Subtasks):**
{plan_str}
- **Trajectory Information:** The full trajectory consists of {num_frames} frames. You are being shown {num_keyframes} carefully selected keyframes that represent potential turning points or changes in action.
- **Important Note:** Subtask 1 implicitly starts at frame 0. You do not need to mark this.

**INPUT DEFINITION:**

For each keyframe, you will be provided with a **Single Composite Image** and metadata. You must parse the image as follows:

1. **The Image Layout:**
   - **LEFT SIDE (RGB View):** This is the first-person view from the robot's camera. Use this to identify objects (beds, doors, counters) and current visibility.
   - **RIGHT SIDE (Top-Down Map):** This is the 2D occupancy map.
     - **Blue Arrow:** The robot's current position and orientation.
     - **Green Line:** The Ground Truth path (where the robot *should* go).
     - **Blue Line:** The History path (where the robot *has* been).
     - **Red Square:** The Goal location.
     - **White/Gray:** Obstacles and passable space.

2. **Ground Truth Action:** The action taken at this frame (e.g., `turn_right`).
3. **Frame ID:** The temporal index.
4. **Detailed analysis of the Top-Down Occupancy Map**: A 2D top-down view map that provides spatial context. This map includes:
   - **Current Position and Orientation**: Shown as a blue arrow indicating where the robot currently is and which direction it is facing
   - **Passable Areas**: Shown in gray, representing areas where the robot can navigate
   - **Obstacles**: Shown in white, representing walls, furniture, or other impassable objects
   - **Starting Point**: Shown as a blue box, marking where the navigation task began
   - **History Path**: Shown as a blue line, representing the path the robot has taken from the start up to the current frame
   - **Ground Truth Trajectory**: Shown as a green line, representing the complete planned path from start to goal
   - **Goal**: Shown as a red square, marking the final destination of the navigation task

**HOW TO USE THE MAP (CRITICAL):**
The Map is your best tool for detecting "Turning Points" related to geometry.
- **For Turn Actions:** Look at the **Blue Arrow** on the map. The subtask "Turn Right" is only complete when the Blue Arrow has finished rotating and is pointing down the new path (the Green Line).
- **For Room Exits:** Look at the **Blue Arrow** relative to the doorway constrictions on the map. The subtask is complete when the arrow has passed through the narrow doorway.


Use the RGB image to understand visual context and object recognition, the action to understand movement patterns, the frame ID to track temporal progression, and the top-down map to understand spatial relationships, geometric transitions, and overall navigation context.

**DEFINITION OF TURNING POINT (CRITICAL):**
A **turning point** is the frame where the **previous subtask is FULLY COMPLETED**. 

**RULES FOR SUBTASK COMPLETION:**
1. **Turn Instructions:** If the subtask is "Turn Right", the subtask is NOT complete when the turning starts. It is only complete when the robot **STOPS turning** and resumes `forward` movement or `stop`.
   - *Incorrect:* Frame 20 (Action changes from Forward -> Turn_Right).
   - *Correct:* Frame 25 (Action changes from Turn_Right -> Forward).

2. **Walk Instructions:** If the subtask is "Walk to the door", the subtask is complete when the robot is **close enough to touch it**, not just when it sees it.

**YOUR TASK:**
Identify exactly **{num_subtasks - 1} turning points** (where subtask transitions occur). Since subtask 1 starts at frame 0, you need to identify:
- The frame where subtask 2 begins
- The frame where subtask 3 begins
- ... and so on until subtask {num_subtasks}

**META-ACTION CLASSIFICATION GUIDANCE:**

To help you identify turning points, here are 6 meta-action categories that describe common types of subtask transitions. You may optionally classify each turning point according to these categories, but classification is not required (some transitions may be outliers):

1. **Turn (转向)**
   - Typical Instructions: "Turn left", "Turn right", "Turn around"
   - Start Condition: Action changes to Left/Right. Map: Trajectory tangent angle changes significantly.
   - End Condition: Action returns to Forward or Stop. Map: Angle change stabilizes, and robot begins displacement.
   - Core Dependencies: Action + Map (Trajectory)

2. **Region Transition (区域转移)**
   - Typical Instructions: "Exit the room", "Go through doorway", "Enter the hall"
   - Start Condition: Map: Robot is in open area, moving toward a narrow area (Doorway/Choke point). Action: Forward.
   - End Condition: Map: Robot trajectory passes through a narrow area (bottleneck) on the map and enters a new open area or long corridor.
   - Core Dependencies: Map (Geometry Analysis)

3. **Visual Approach (走向某个物体)**
   - Typical Instructions: "Go to the bed", "Walk to the sink"
   - Start Condition: RGB: Target object appears in view. Action: Forward/Adjust.
   - End Condition: Action: Stop. RGB: Target object occupies large portion of view (Depth is small).
   - Core Dependencies: RGB (Object Detection) + Action

4. **Visual Passing (走过某个物体)**
   - Typical Instructions: "Walk past the sofa", "Pass the table"
   - Start Condition: RGB: Target object appears in view. Action: Forward.
   - End Condition: RGB: Target object moves from center to edge of view, then disappears. Action: Forward continues after object disappears.
   - Core Dependencies: RGB (Tracking) + Action

5. **General Cruise (一般行进)**
   - Typical Instructions: "Walk straight", "Go down the hall"
   - Start Condition: Action: Forward. Map: Robot moves straight in long corridor or open area.
   - End Condition: Event: Triggers start condition of any other Meta-Action (e.g., detects intersection requiring turn, or detects object).
   - Core Dependencies: Action + Map

6. **Stop (终止)**
   - Typical Instructions: "Stop at", "Wait there"
   - Start Condition: RGB: End point target object appears in view. Action: Forward/Adjust.
   - End Condition: Action: Stop or N/A (task ends).
   - Core Dependencies: RGB, Action

**STATE MACHINE LOGIC EXAMPLES:**

Example 1: "Exit the room" (Geometric Transition)
- State: IDLE -> EXITING
  - Start: Robot is Moving Forward. Algorithm analyzes local map geometry in real-time, detects obstacles converging ahead (doorway/choke point).
  - Process: Robot coordinates continue forward movement.
  - End (Success): Algorithm detects that obstacles around robot no longer converge (area becomes open), or robot enters a long corridor-like area. Trajectory shows it has crossed the narrow point.

Example 2: "Turn left" (Action-based)
- State: CRUISING -> TURNING
  - Start: Very strict condition. When Current Action == Turn Left, state machine immediately switches.
  - End (Success): When Current Action returns to Forward, and robot's orientation compared to before Turn has changed approximately 90 degrees (tolerance +/- 30 degrees).

Example 3: "Walk past the sofa" (RGB Semantic)
- State: CRUISING -> PASSING
  - Start: RGB detects "Sofa".
  - Process: Robot maintains Forward. Sofa's Bounding Box in RGB moves from center to one side (e.g., moves to right).
  - End (Success): Sofa completely disappears from RGB view (out of view), but Action remains Forward. This means robot has left the object behind.

Example 4: "Go to the sink" (RGB + Action Termination)
- State: CRUISING -> APPROACHING
  - Start: RGB detects "Sink".
  - Process: Robot moves toward Sink, Sink becomes larger in RGB.
  - End (Success): Action becomes Stop, and Sink's area in RGB exceeds threshold (very close).

**OUTPUT FORMAT:**
Your response must be formatted using special tags to separate the reasoning and answer sections:

**Format:**
```
<reason>
[Your detailed Chain-of-Thought analysis here explaining your reasoning process. For each turning point, you should:
- Analyze what visual cues or action patterns indicate the completion of the previous subtask
- Explain how the keyframes relate to the subtask descriptions in the plan
- Provide reasoning for determining when one subtask ends and the next begins
- Optionally classify each transition according to the meta-action categories above (if applicable)
- Explain how actions (forward, turn_left, turn_right, stop) relate to the subtask progression
- Include examples of your reasoning process for different meta-action types]
</reason>
<answer>
[JSON array with your final turning point annotations]
</answer>
```

**Requirements:**
1. The `<reason>` section must contain your detailed analysis and reasoning for each turning point.
2. The `<answer>` section must contain a valid JSON array starting with `[` and ending with `]`.
3. Each element in the JSON array must be a JSON object with two keys:
   - `"turning_point_to_subtask"` (integer): The subtask number that starts at this frame (2-based, since subtask 1 starts at 0)
   - `"frame_index"` (integer): The frame index where this transition occurs
4. You must identify exactly {num_subtasks - 1} turning points (one for each transition from subtask 1->2, 2->3, ..., {num_subtasks - 1}->{num_subtasks}).
5. The `turning_point_to_subtask` values must be strictly increasing: 2, 3, 4, ..., {num_subtasks}.
6. The `frame_index` values must be strictly increasing and within the valid range [0, {num_frames - 1}].
7. **CRITICAL: Your answer must strictly follow your reasoning process.** The frame indices you provide in the `<answer>` section must be consistent with the analysis you present in the `<reason>` section. Do not provide frame indices that contradict your own reasoning.
8. Control your reasoning process to be as concise as possible. You should only point out the critical reason that you choose the answer.

**IMPORTANT NOTE ON INCOMPLETE TASKS AND KEYFRAME CONSTRAINTS:**

Sometimes the robot may not actually complete all subtasks. The navigation task may be judged as successful even if the last one or two subtasks have not been fully completed. In such cases, the final turning point(s) may not appear in the trajectory.

**KEYFRAME RANGE CONSTRAINT:**

**CRITICAL:** The frame indices you specify for each subtask turning point **must be one of the keyframe indices provided to you**. You are only shown specific keyframes (not all frames), so you can only identify turning points at those keyframe positions that you actually see.

- If a subtask transition clearly occurs at a keyframe you see, use that keyframe's frame index.
- If a subtask transition appears to occur between keyframes, choose the keyframe that is closest to where the transition likely happened.
- **If a subtask did not actually occur in the trajectory** (i.e., the robot did not complete that subtask), you must mark it as the **last keyframe index** you were shown in the keyframes list (not the last frame of the entire trajectory). This ensures consistency with the keyframes you actually observed and prevents you from specifying frame indices that were not shown to you.

**SPECIAL CASE - INCOMPLETE TASKS:**

**If you cannot identify a clear turning point for the last subtask(s)**, you should mark the missing turning point(s) as the **last keyframe index** you were shown (which should be the highest frame index among all keyframes provided to you). For example:
- If you can identify turning points for subtasks 2, 3, but not for subtask 4, then mark subtask 4's turning point as the last keyframe index you see.
- If you can only identify turning points for subtasks 2, 3, 4, but the plan has 5 subtasks, then mark subtask 5's turning point as the last keyframe index you see.

This ensures that all required {num_subtasks - 1} turning points are provided, even when the task was completed early, while maintaining consistency with the keyframes you actually observed.

**EXAMPLE OUTPUT FORMAT:**
```
<reason>
Looking at the keyframes, I can see that frame 0 shows the starting position. The agent needs to walk forward to the doorway adjacent to the bedroom (subtask 1). 

At frame 9, I observe a turning action (turn_left), which suggests the agent is positioning itself to face the shower door direction. This indicates the completion of subtask 1 and the start of subtask 2. This transition can be classified as "Turn (转向)" - the action changed to turn_left, and the trajectory angle changed significantly.

Frame 15 still shows movement toward the shower, so it's still subtask 2. By frame 32, the agent appears to be approaching the shower door, indicating the start of subtask 3. This transition can be classified as "Visual Approach (视觉趋近)" - the target object (shower door) appears in view and the agent is moving toward it.

Finally, frame 41 shows the agent stopping at the shower door, completing subtask 3 and starting subtask 4. This transition can be classified as "Stop (终止)" - the action becomes stop and the target is very close in view.
</reason>
<answer>
[
  {{"turning_point_to_subtask": 2, "frame_index": 9}},
  {{"turning_point_to_subtask": 3, "frame_index": 32}},
  {{"turning_point_to_subtask": 4, "frame_index": 41}}
]
</answer>
```

Keyframes:"""
    
    return prompt


def get_vlm_keyframe_annotations(
    instruction: str,
    plan: List[str],
    frame_paths: List[str],
    keyframe_indices: List[int],
    num_subtasks: int,
    actions: List[int] = None,
    episode_key: str = None
) -> Dict[str, any]:
    """
    Use VLM to identify turning points (transition points between subtasks).
    
    Args:
        instruction: The original navigation instruction
        plan: List of subtask descriptions
        frame_paths: List of all frame paths
        keyframe_indices: List of candidate keyframe indices
        num_subtasks: Number of subtasks
        actions: List of ground truth actions for each frame (optional)
        episode_key: Episode identifier for logging (optional)
        
    Returns:
        Dictionary with raw VLM response (parsing will be done later)
    """
    try:
        # Action name mapping
        action_names = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
        
        # Prepare downsampled keyframes for VLM
        keyframe_images = []
        for idx in keyframe_indices:
            if idx < len(frame_paths):
                b64_frame = downsample_frame(frame_paths[idx])
                if b64_frame:
                    # Get action for this frame
                    action_idx = actions[idx] if actions and idx < len(actions) else -1
                    action_name = action_names.get(action_idx, "unknown")
                    keyframe_images.append({
                        "frame_index": idx,
                        "base64": b64_frame,
                        "action": action_name,
                        "action_idx": action_idx
                    })
        
        # Build prompt using the new function
        prompt_header = build_transition_point_prompt(
            instruction=instruction,
            plan=plan,
            num_frames=len(frame_paths),
            num_keyframes=len(keyframe_images)
        )
        
        # Build content with frame_index labels paired with each image
        content_items = [
            {
                "type": "text",
                "text": prompt_header
            }
        ]
        
        # Add each image with its frame_index label
        for kf_info in keyframe_images:
            # Add text label before each image (include action if available)
            action_text = f", Action: {kf_info['action']}" if kf_info.get('action') != "unknown" else ""
            content_items.append({
                "type": "text",
                "text": f"\nFrame_id {kf_info['frame_index']}{action_text}:"
            })
            # Add the image
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{kf_info['base64']}"
                }
            })
        
        # Check if content_items is not empty
        if not content_items or len(content_items) == 0:
            print(f"Error: Empty content provided to VLM API")
            return {"status": "error", "error": "Empty content provided to VLM API", "raw_response": ""}
        
        # Retry logic: if content is not empty but response is empty, retry up to 3 times
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Call VLM API
                response = client.chat.completions.create(
                    model='Qwen/Qwen3-VL-235B-A22B-Thinking',
                    # model='qwen/qwen2.5-vl-32b-instruct',
                    messages=[
                        {
                            "role": "user",
                            "content": content_items
                        }
                    ]
                )
                
                # Check if response is valid
                if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                    print(f"Attempt {attempt}/{max_retries}: Empty response from VLM API")
                    if attempt < max_retries:
                        continue
                    return {"status": "error", "error": "Empty response from VLM API after retries", "raw_response": ""}
                
                # Check if message content exists
                if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                    print(f"Attempt {attempt}/{max_retries}: Response missing message content")
                    if attempt < max_retries:
                        continue
                    return {"status": "error", "error": "Response missing message content after retries", "raw_response": ""}
                
                response_text = response.choices[0].message.content
                
                # Check if content is None or empty
                if response_text is None or not response_text.strip():
                    episode_prefix = f"[{episode_key}] " if episode_key else ""
                    print(f"{episode_prefix}Attempt {attempt}/{max_retries}: VLM API returned null or empty content")
                    if attempt < max_retries:
                        print(f"  Retrying...")
                        continue
                    return {"status": "error", "error": "VLM API returned null or empty content after retries", "raw_response": ""}
                
                # Success: got valid response
                episode_prefix = f"[{episode_key}] " if episode_key else ""
                print(f"{episode_prefix}✓ Successfully received response from VLM API (attempt {attempt}/{max_retries})")
                return {
                    "status": "success",
                    "raw_response": response_text
                }
                
            except Exception as e:
                episode_prefix = f"[{episode_key}] " if episode_key else ""
                print(f"{episode_prefix}Attempt {attempt}/{max_retries}: Exception during API call: {e}")
                if attempt < max_retries:
                    print(f"  Retrying...")
                    continue
                return {"status": "error", "error": f"Exception after retries: {str(e)}", "raw_response": ""}
        
        # Should not reach here, but just in case
        return {"status": "error", "error": "Failed to get response after all retries", "raw_response": ""}
    
    except Exception as e:
        print(f"Error getting VLM keyframe annotations: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def parse_transition_points(raw_response: str) -> List[Dict]:
    """
    Parse turning points from raw VLM response.
    Handles cases where VLM doesn't follow format strictly.
    
    Args:
        raw_response: Raw response text from VLM
        
    Returns:
        List of transition point dictionaries with keys:
        - "turning_point_to_subtask": int (subtask number that starts at this frame)
        - "frame_index": int (frame index where transition occurs)
    """
    if not raw_response:
        return []
    
    transition_points = []
    
    try:
        # Strategy 1: Try to extract JSON from <answer> tags
        # Find <answer> section
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            
            # Try to find JSON array in the answer text
            # Look for array pattern
            json_match = re.search(r'\[.*?\]', answer_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Try to parse JSON
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                tp_subtask = item.get("turning_point_to_subtask") or item.get("subtask")
                                frame_idx = item.get("frame_index") or item.get("frame_idx") or item.get("frame")
                                
                                if tp_subtask is not None and frame_idx is not None:
                                    transition_points.append({
                                        "turning_point_to_subtask": int(tp_subtask),
                                        "frame_index": int(frame_idx)
                                    })
                except json.JSONDecodeError:
                    pass
        
        # Strategy 2: If no <answer> tags, try to find JSON array anywhere in response
        if not transition_points:
            json_match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                tp_subtask = item.get("turning_point_to_subtask") or item.get("subtask")
                                frame_idx = item.get("frame_index") or item.get("frame_idx") or item.get("frame")
                                
                                if tp_subtask is not None and frame_idx is not None:
                                    transition_points.append({
                                        "turning_point_to_subtask": int(tp_subtask),
                                        "frame_index": int(frame_idx)
                                    })
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Try to extract from markdown code blocks
        if not transition_points:
            code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_response, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                tp_subtask = item.get("turning_point_to_subtask") or item.get("subtask")
                                frame_idx = item.get("frame_index") or item.get("frame_idx") or item.get("frame")
                                
                                if tp_subtask is not None and frame_idx is not None:
                                    transition_points.append({
                                        "turning_point_to_subtask": int(tp_subtask),
                                        "frame_index": int(frame_idx)
                                    })
                except json.JSONDecodeError:
                    pass
        
    except Exception as e:
        print(f"Warning: Error parsing transition points: {e}")
        import traceback
        traceback.print_exc()
    
    # Sort by turning_point_to_subtask to ensure correct order
    transition_points.sort(key=lambda x: x["turning_point_to_subtask"])
    
    return transition_points


def interpolate_subtask_sequence(
    transition_points: List[Dict],
    num_frames: int,
    num_subtasks: int
) -> List[int]:
    """
    Interpolate subtask sequence for all frames based on transition points.
    
    Args:
        transition_points: List of transition point dictionaries with:
            - "turning_point_to_subtask": int (subtask number that starts at this frame)
            - "frame_index": int (frame index where transition occurs)
        num_frames: Total number of frames in the video
        num_subtasks: Total number of subtasks
        
    Returns:
        List where index is frame number and value is subtask index (1-indexed)
    """
    # Initialize result array (all frames start with subtask 1)
    result = [1] * num_frames
    
    if not transition_points:
        # No transition points found, all frames are subtask 1
        return result
    
    # Sort transition points by frame index
    sorted_transitions = sorted(transition_points, key=lambda x: x["frame_index"])
    
    # Build subtask sequence based on transition points
    # Frames 0 to first_turning_point-1 = subtask 1
    # first_turning_point to second_turning_point-1 = subtask 2
    # etc.
    
    for i, transition in enumerate(sorted_transitions):
        frame_idx = transition["frame_index"]
        subtask_idx = transition["turning_point_to_subtask"]
        
        # Validate subtask index
        if subtask_idx < 2 or subtask_idx > num_subtasks:
            print(f"Warning: Invalid subtask index {subtask_idx} at frame {frame_idx}, skipping")
            continue
        
        # Validate frame index
        if frame_idx < 0 or frame_idx >= num_frames:
            print(f"Warning: Invalid frame index {frame_idx}, skipping")
            continue
        
        # Determine the range for this subtask
        start_frame = frame_idx
        
        # Find the end frame (where next subtask starts)
        if i + 1 < len(sorted_transitions):
            end_frame = sorted_transitions[i + 1]["frame_index"]
        else:
            end_frame = num_frames
        
        # Fill frames in this range with current subtask
        for frame_num in range(start_frame, end_frame):
            if frame_num < num_frames:
                result[frame_num] = subtask_idx
    
    return result


def get_episode_key(scene_id, episode_id):
    """Generate unique episode key from scene_id and episode_id"""
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)


def process_single_episode(episode_data, episode_idx, total_episodes, subtask_splits, trajectory_dir, output_file, write_lock, processed_episode_keys):
    """Process a single episode and determine subtasks
    
    Returns:
        Tuple (episode_key, status, result_dict or error)
    """
    episode_id = episode_data["id"]
    scene_id = episode_data.get("scene_id")
    episode_key = get_episode_key(scene_id, episode_id)
    
    # Check if episode_key already exists (avoid duplicates) - thread-safe
    with write_lock:
        if episode_key in processed_episode_keys:
            return (episode_key, "skipped", "already processed")
    
    if episode_key not in subtask_splits:
        return (episode_key, "skipped", None)
    
    subtask_data = subtask_splits[episode_key]
    instruction = subtask_data.get("instruction")
    plan = subtask_data.get("plan", [])
    video_rel_path = episode_data.get("video", "")
    actions = episode_data.get("actions", [])
    
    if not video_rel_path or not actions:
        return (episode_key, "skipped", None)
    
    video_path = os.path.join(trajectory_dir, video_rel_path, "trajectory.mp4")
    if not os.path.exists(video_path):
        return (episode_key, "failed", f"Video not found: {video_path}")
    
    print(f"[{episode_idx+1}/{total_episodes}] Processing {episode_key}")
    
    try:
        # Extract frames
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        temp_frame_dir = "/tmp/streamvln_frames_deploy"
        os.makedirs(temp_frame_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_frame_dir, f"{episode_key}_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1
        cap.release()
        
        if not frame_paths:
            return (episode_key, "failed", "No frames extracted")
        
        # Identify keyframes
        keyframes = identify_action_change_keyframes(actions, window_size=3)
        
        # Get VLM annotations
        num_subtasks = len(plan)
        vlm_result = get_vlm_keyframe_annotations(
            instruction=instruction,
            plan=plan,
            frame_paths=frame_paths,
            keyframe_indices=keyframes,
            num_subtasks=num_subtasks,
            actions=actions,
            episode_key=episode_key
        )
        
        # Parse transition points from raw response
        transition_points = []
        subtask_sequence = [1] * len(frame_paths)
        parsed_answer = None
        raw_response = ""
        
        # Determine success/failure based on raw_response
        if vlm_result.get("status") == "success":
            raw_response = vlm_result.get("raw_response", "")
            
            # Check if raw_response is not empty
            if raw_response and raw_response.strip():
                # Parse turning points
                transition_points = parse_transition_points(raw_response)
                
                if transition_points:
                    # Generate subtask sequence from transition points
                    subtask_sequence = interpolate_subtask_sequence(
                        transition_points=transition_points,
                        num_frames=len(frame_paths),
                        num_subtasks=num_subtasks
                    )
                    
                    # Extract the answer section for saving
                    answer_match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL | re.IGNORECASE)
                    if answer_match:
                        parsed_answer = answer_match.group(1).strip()
                    else:
                        # If no <answer> tags, try to find JSON in markdown code blocks
                        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_response, re.DOTALL)
                        if code_block_match:
                            parsed_answer = code_block_match.group(1).strip()
                        else:
                            # If still not found, reconstruct JSON from parsed transition_points
                            if transition_points:
                                parsed_answer = json.dumps(transition_points, indent=2)
        
        # Determine final status based on raw_response
        is_success = (raw_response and raw_response.strip()) if raw_response else False
        
        # Calculate statistics
        subtask_counts = {}
        for idx in subtask_sequence:
            subtask_counts[idx] = subtask_counts.get(idx, 0) + 1
        
        # Prepare result
        result_record = {
            "episode_key": episode_key,
            "episode_id": episode_id,
            "scene_id": scene_id,
            "status": "success" if is_success else "failed",
            "num_frames": len(frame_paths),
            "num_subtasks": num_subtasks,
            "keyframes": keyframes,
            "subtask_sequence": subtask_sequence,
            "subtask_counts": subtask_counts,
            "vlm_raw_response": raw_response if raw_response else None,
            "transition_points": transition_points if transition_points else None,
            "parsed_answer": parsed_answer if parsed_answer else None
        }
        
        if not is_success:
            result_record["error"] = vlm_result.get("error", "VLM API returned empty response after retries")
        
        # Write result to file - thread-safe
        with write_lock:
            with open(output_file, "a") as f:
                json.dump(result_record, f)
                f.write("\n")
                f.flush()
            # Mark as processed (thread-safe)
            processed_episode_keys.add(episode_key)
        
        print(f"  ✓ Success [{episode_key}]: {len(frame_paths)} frames, {num_subtasks} subtasks")
        return (episode_key, "success" if is_success else "failed", result_record)
    
    except Exception as e:
        print(f"  ✗ Error [{episode_key}]: {str(e)}")
        return (episode_key, "failed", str(e))


def determine_subtasks_for_all_episodes(
    trajectory_dir: str,
    subtask_splits_file: str,
    output_file: str,
    max_workers: int = 4
):
    """Process all episodes to determine subtasks for each frame
    
    Args:
        trajectory_dir: Path to trajectory data directory (contains summary.json)
        subtask_splits_file: Path to subtask_splits.jsonl file
        output_file: Output JSONL file to save results
        max_workers: Maximum number of concurrent workers
    """
    # Load subtask splits
    subtask_splits = {}
    print(f"Loading subtask splits from: {subtask_splits_file}")
    if not os.path.exists(subtask_splits_file):
        print(f"Error: Subtask splits file not found: {subtask_splits_file}")
        return
    
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
    
    # Load episode data
    summary_file = os.path.join(trajectory_dir, "summary.json")
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found: {summary_file}")
        return
    
    episodes = []
    print(f"Loading episodes from: {summary_file}")
    with open(summary_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    episodes.append(data)
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing episode_keys to avoid duplicates
    processed_episode_keys = set()
    if os.path.exists(output_file):
        print(f"Loading existing episode_keys from {output_file}...")
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        episode_key = data.get("episode_key")
                        if episode_key:
                            processed_episode_keys.add(episode_key)
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_episode_keys)} already processed episodes")
    
    # Process episodes concurrently
    print(f"Processing episodes with {max_workers} workers...")
    write_lock = Lock()
    successful = 0
    failed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                process_single_episode,
                episode_data,
                idx,
                len(episodes),
                subtask_splits,
                trajectory_dir,
                output_file,
                write_lock,
                processed_episode_keys
            )
            for idx, episode_data in enumerate(episodes)
        ]
        
        for fut in as_completed(futures):
            try:
                episode_key, status, result = fut.result()
                if status == "success":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing episode: {e}")
                failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total episodes: {len(episodes)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Determine subtasks for all episodes (batch processing)")
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R",
                       help="Path to trajectory data directory (contains summary.json)")
    parser.add_argument("--subtask_splits_file", type=str, default="streamvln/cot_data/subtask_splits.jsonl",
                       help="Path to subtask_splits.jsonl file")
    parser.add_argument("--output_file", type=str, default="data/subtask_determination_results/subtask_determination.jsonl",
                       help="Output JSONL file to save all results (one line per episode)")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    determine_subtasks_for_all_episodes(
        trajectory_dir=args.trajectory_dir,
        subtask_splits_file=args.subtask_splits_file,
        output_file=args.output_file,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
