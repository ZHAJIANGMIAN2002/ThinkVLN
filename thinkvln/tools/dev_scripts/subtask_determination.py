import os
import json
import re
import base64
import cv2
import numpy as np
import argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from openai import OpenAI
from habitat.utils.visualizations.utils import images_to_video

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
)


def identify_action_change_keyframes(actions: List[int], window_size: int = 3) -> List[int]:
    """
    Identify keyframes where actions change using sliding window to filter noise.
    
    Args:
        actions: List of ground truth actions for each frame
        window_size: Size of sliding window to detect significant action changes
        
    Returns:
        List of frame indices that are candidate keyframes
    """
    if len(actions) < 2:
        return [0] if len(actions) > 0 else []

    actions = actions[1:]
    keyframes = [0]  # Always include first frame
    
    # Find frames where action changes
    for i in range(1, len(actions)):
        # Always include the last frame
        if i == len(actions) - 1:
            keyframes.append(i)
            continue
        
        is_change = False
        
        # Get the "dominant" action in the window before this frame
        start_idx = max(0, i - window_size)
        end_idx = i
        prev_actions = actions[start_idx:end_idx]
        prev_mode = max(set(prev_actions), key=prev_actions.count) if prev_actions else actions[i-1]
        
        # Get the action at current frame
        curr_action = actions[i]
        
        # Check if this represents a real change
        if curr_action != prev_mode:
            # Verify it's not just a brief flicker - check if current action persists
            if i + window_size <= len(actions):
                future_actions = actions[i:min(i + window_size, len(actions))]
                future_mode = max(set(future_actions), key=future_actions.count) if future_actions else curr_action
                if curr_action == future_mode or (i + 1 < len(actions) and actions[i+1] == curr_action):
                    is_change = True
            else:
                is_change = True
        
        if is_change and i not in keyframes:
            keyframes.append(i)
    
    # Ensure minimum spacing between keyframes (at least every 5 steps)
    min_spacing = 5
    result_keyframes = [keyframes[0]]
    
    # Enforce minimum spacing by inserting intermediate keyframes
    i = 1
    while i < len(keyframes):
        last_kf = result_keyframes[-1]
        kf = keyframes[i]
        if kf - last_kf >= min_spacing:
            # Insert additional keyframes every min_spacing between last_kf and kf
            next_kf = last_kf + min_spacing
            while next_kf < kf:
                result_keyframes.append(next_kf)
                next_kf += min_spacing
            result_keyframes.append(kf)
        else:
            result_keyframes.append(kf)
        i += 1

    # Always include last frame
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


def downsample_frame(frame_path: str, target_size: Tuple[int, int] = None) -> str:
    """
    Downsample an image frame to reduce resolution for VLM processing.
    
    Args:
        frame_path: Path to the frame image
        target_size: Target resolution (width, height) - if None, will downsample by half
        
    Returns:
        Base64 encoded string of downsampled frame
    """
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            return ""
        
        # Get original image resolution
        original_height, original_width = frame.shape[:2]
        
        # Calculate target size as half of each dimension
        calculated_target_size = (original_width // 2, original_height // 2)
        
        # Use calculated target size (half resolution)
        downsampled = cv2.resize(frame, calculated_target_size, interpolation=cv2.INTER_AREA)
        
        # Verify downsample is correct (each side downgraded by half)
        downsampled_height, downsampled_width = downsampled.shape[:2]
        expected_width = original_width // 2
        expected_height = original_height // 2
        if downsampled_width != expected_width or downsampled_height != expected_height:
            print(f"Warning: Downsample verification failed for {frame_path}")
            print(f"  Original: {original_width}x{original_height}, Expected: {expected_width}x{expected_height}, Got: {downsampled_width}x{downsampled_height}")
        _, buffer = cv2.imencode('.jpg', downsampled, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        return base64.standard_b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"Error downsampling frame {frame_path}: {e}")
        return ""


def frame_to_base64(frame_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        with open(frame_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error converting frame {frame_path} to base64: {e}")
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
    actions: List[int] = None
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
                    # model="qwen/qwen3-vl-32b-instruct",
                    model='qwen/qwen3-vl-30b-a3b-instruct',
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
                    print(f"Attempt {attempt}/{max_retries}: VLM API returned null or empty content")
                    if attempt < max_retries:
                        print(f"  Retrying...")
                        continue
                    return {"status": "error", "error": "VLM API returned null or empty content after retries", "raw_response": ""}
                
                # Success: got valid response
                print(f"✓ Successfully received response from VLM API (attempt {attempt}/{max_retries})")
                return {
                    "status": "success",
                    "raw_response": response_text
                }
                
            except Exception as e:
                print(f"Attempt {attempt}/{max_retries}: Exception during API call: {e}")
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


def generate_annotated_video(
    frame_paths: List[str],
    actions: List[int],
    subtask_sequence: List[int],
    subtask_list: List[str],
    keyframes: List[int],
    output_video_path: str,
    fps: int = 6
) -> bool:
    """
    Generate a video with annotations for subtasks, actions, and keyframes.
    Uses Habitat's images_to_video function for video generation.
    
    Args:
        frame_paths: List of frame image paths
        actions: List of ground truth actions for each frame
        subtask_sequence: List of subtask indices for each frame
        subtask_list: List of subtask descriptions
        keyframes: List of keyframe indices
        output_video_path: Path to save the annotated video
        fps: Frames per second for the output video
        
    Returns:
        True if successful, False otherwise
    """
    try:
        action_names = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
        keyframe_set = set(keyframes)
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            print(f"Error: Could not read first frame: {frame_paths[0]}")
            return False
        
        height, width = first_frame.shape[:2]
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_video_path) if os.path.dirname(output_video_path) else "."
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating annotated video: {output_video_path}")
        
        # Build annotated frames as numpy arrays
        annotated_frames = []
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame: {frame_path}")
                continue
            
            # Ensure frame is in correct format (BGR for OpenCV)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Get frame information
            frame_subtask = subtask_sequence[frame_idx] if frame_idx < len(subtask_sequence) else 1
            subtask_text = subtask_list[frame_subtask - 1] if frame_subtask <= len(subtask_list) else "Unknown"
            
            frame_action = actions[frame_idx] if frame_idx < len(actions) else -1
            action_text = action_names.get(frame_action, "unknown")
            
            is_keyframe = frame_idx in keyframe_set
            keyframe_text = "[KEYFRAME]" if is_keyframe else ""
            
            # Add text annotations to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            color_white = (255, 255, 255)
            color_red = (0, 0, 255)
            color_green = (0, 255, 0)
            
            # Background for text (dark overlay)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            # Frame info
            cv2.putText(frame, f"Frame: {frame_idx}/{len(frame_paths)-1}", 
                       (10, 25), font, font_scale, color_white, font_thickness)
            
            # Subtask info
            cv2.putText(frame, f"Subtask {frame_subtask}: {subtask_text}", 
                       (10, 50), font, font_scale, color_green, font_thickness)
            
            # Action info
            cv2.putText(frame, f"Action: {action_text}", 
                       (10, 75), font, font_scale, color_white, font_thickness)
            
            # Keyframe info
            if is_keyframe:
                cv2.putText(frame, keyframe_text, 
                           (width - 200, 25), font, font_scale, color_red, font_thickness)
            
            # Convert BGR to RGB for images_to_video (Habitat expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frames.append(frame_rgb)
        
        # Use Habitat's images_to_video function
        video_name = os.path.basename(output_video_path).replace('.mp4', '')
        images_to_video(
            annotated_frames,
            output_dir,
            video_name,
            fps=fps,
            quality=9
        )
        
        actual_video_path = os.path.join(output_dir, video_name + '.mp4')
        print(f"✓ Annotated video saved: {actual_video_path}")
        return True
        
    except Exception as e:
        print(f"Error generating annotated video: {e}")
        import traceback
        traceback.print_exc()
        return False


def determine_subtasks_for_video(
    video_or_frames: str,
    actions: List[int],
    subtask_list: List[str],
    instruction: str,
    output_file: Optional[str] = None,
    use_vlm: bool = True,
    generate_video: bool = False
) -> Dict[str, any]:
    """
    Main function to determine subtask for each frame in a video.
    
    Args:
        video_or_frames: Path to video file or directory containing frame images
        subtask_list: List of subtask descriptions (usually from subtask_split)
        instruction: The original navigation instruction
        output_file: Optional path to save results as JSON
        use_vlm: Whether to use VLM for keyframe annotation (default: True)
        generate_video: Whether to generate annotated video output (default: True)
        
    Returns:
        Dictionary with results including:
            - num_frames: Total number of frames
            - num_subtasks: Total number of subtasks
            - subtask_sequence: List of subtask indices (1-indexed) for each frame
            - keyframes: List of identified keyframe indices
            - vlm_annotations: VLM keyframe annotations (if use_vlm=True)
            - annotated_video: Path to annotated video (if generate_video=True)
    """
    
    # Step 1: Load frames
    frame_paths = []
    if os.path.isfile(video_or_frames):
        # Extract frames from video
        print(f"Extracting frames from video: {video_or_frames}")
        cap = cv2.VideoCapture(video_or_frames)
        frame_idx = 0
        temp_frame_dir = "/tmp/streamvln_frames"
        os.makedirs(temp_frame_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_frame_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1
        cap.release()
    else:
        # Load frames from directory
        print(f"Loading frames from directory: {video_or_frames}")
        frame_dir = Path(video_or_frames)
        frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        frame_paths = [str(p) for p in frame_paths]
    
    if not frame_paths:
        print("Error: No frames found!")
        return {"status": "error", "error": "No frames found"}
    
    print(f"Loaded {len(frame_paths)} frames")
    

    # Step 2: Identify candidate keyframes
    print("Step 1: Identifying candidate keyframes based on action changes...")
    keyframes = identify_action_change_keyframes(actions, window_size=3)
    print(f"Identified {len(keyframes)} candidate keyframes at indices: {keyframes[:10]}..." if len(keyframes) > 10 else f"Identified {len(keyframes)} candidate keyframes at indices: {keyframes}")
    
    # Step 3: Get VLM annotations if enabled
    num_subtasks = len(subtask_list)
    vlm_annotations = []
    
    if use_vlm:
        print(f"Step 2: Getting VLM annotations for {len(keyframes)} keyframes...")
        vlm_result = get_vlm_keyframe_annotations(
            instruction=instruction,
            plan=subtask_list,
            frame_paths=frame_paths,
            keyframe_indices=keyframes,
            num_subtasks=num_subtasks,
            actions=actions
        )
        
        if vlm_result["status"] == "success":
            raw_response = vlm_result.get("raw_response", "")
            print(f"Received VLM raw response (length: {len(raw_response)} characters)")
            print(f"Raw response preview: {raw_response[:500]}...")
        else:
            print(f"Warning: VLM annotation failed with status: {vlm_result['status']}")
            raw_response = vlm_result.get('raw_response', '')
            print(f"Raw response: {raw_response[:500]}")
    else:
        raw_response = ""
    
    # Step 4: Parse turning points from raw response and generate subtask sequence
    transition_points = []
    subtask_sequence = [1] * len(frame_paths)
    parsed_answer = None
    
    if use_vlm and raw_response:
        print(f"Step 3: Parsing turning points from VLM response...")
        transition_points = parse_transition_points(raw_response)
        
        if transition_points:
            print(f"  ✓ Parsed {len(transition_points)} turning points:")
            for tp in transition_points:
                print(f"    - Subtask {tp['turning_point_to_subtask']} starts at frame {tp['frame_index']}")
            
            # Generate subtask sequence from transition points
            subtask_sequence = interpolate_subtask_sequence(
                transition_points=transition_points,
                num_frames=len(frame_paths),
                num_subtasks=num_subtasks
            )
            
            # Extract the answer section for saving
            # Try to find <answer></answer> tags first
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
        else:
            print(f"  ⚠ Warning: Could not parse turning points from response, using default (all frames = subtask 1)")
    else:
        print(f"Step 3: No VLM response to parse, using default (all frames = subtask 1)")
    
    # Calculate subtask counts
    subtask_counts = {}
    for subtask_idx in subtask_sequence:
        subtask_counts[subtask_idx] = subtask_counts.get(subtask_idx, 0) + 1
    
    # Prepare result
    result = {
        "status": "success",
        "num_frames": len(frame_paths),
        "num_subtasks": num_subtasks,
        "subtask_sequence": subtask_sequence,
        "keyframes": keyframes,
        "vlm_raw_response": raw_response if use_vlm else None,
        "transition_points": transition_points if transition_points else None,
        "parsed_answer": parsed_answer if parsed_answer else None,
        "subtask_counts": subtask_counts
    }
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w") as f:
            # Convert subtask_sequence to JSON-serializable format
            result_copy = result.copy()
            result_copy["subtask_sequence"] = subtask_sequence
            json.dump(result_copy, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    # Step 5: Generate annotated video (if requested)
    if generate_video:
        print(f"Step 4: Generating annotated video...")
        if output_file:
            # Generate video output path from JSON output path
            video_output_path = output_file.replace('.json', '_annotated.mp4')
        else:
            video_output_path = "/tmp/subtask_annotated.mp4"
        
        success = generate_annotated_video(
            frame_paths=frame_paths,
            actions=actions,
            subtask_sequence=subtask_sequence,
            subtask_list=subtask_list,
            keyframes=keyframes,
            output_video_path=video_output_path,
            fps=6
        )
        
        if success:
            result["annotated_video"] = video_output_path
    else:
        print(f"Step 4: Skipping video generation (generate_video=False)")
    
    return result


def get_episode_key(scene_id, episode_id):
    """Generate unique episode key from scene_id and episode_id"""
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)


def process_all_episodes_subtask_determination(
    trajectory_dir: str,
    subtask_splits_file: str,
    output_dir: str,
    use_vlm: bool = True,
    generate_video: bool = False
):
    """Process all episodes to determine subtasks for each frame
    
    Args:
        trajectory_dir: Path to trajectory data directory (contains summary.json)
        subtask_splits_file: Path to subtask_splits.jsonl file
        output_dir: Directory to save results
        use_vlm: Whether to use VLM for keyframe annotation
        generate_video: Whether to generate annotated videos
    """
    # Load subtask splits into a dictionary keyed by (scene_id, episode_id)
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
    
    # Load episode data from summary.json
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
    os.makedirs(output_dir, exist_ok=True)
    
    # Open output file for appending (each episode will be written immediately)
    output_file = os.path.join(output_dir, "subtask_determination_results.jsonl")
    print(f"Results will be appended to: {output_file}")
    
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
    
    # Process each episode
    successful = 0
    failed = 0
    skipped = 0
    
    # Open file in append mode - each episode will be written immediately
    with open(output_file, "a") as f:
        for idx, episode_data in enumerate(episodes, 1):
            episode_id = episode_data["id"]
            scene_id = episode_data.get("scene_id")
            episode_key = get_episode_key(scene_id, episode_id)
            
            # Check if episode_key already exists (avoid duplicates)
            if episode_key in processed_episode_keys:
                print(f"\n[{idx}/{len(episodes)}] Skipping episode {episode_key} (already processed)")
                skipped += 1
                continue
            
            # Check if subtask split exists for this episode
            if episode_key not in subtask_splits:
                print(f"\n[{idx}/{len(episodes)}] Skipping episode {episode_key} (no subtask split)")
                skipped += 1
                continue
            
            subtask_data = subtask_splits[episode_key]
            instruction = subtask_data.get("instruction")
            plan = subtask_data.get("plan", [])
            
            # Get video path
            video_rel_path = episode_data.get("video", "")
            actions = episode_data.get("actions", [])
            
            if not video_rel_path or not actions:
                print(f"\n[{idx}/{len(episodes)}] Skipping episode {episode_key} (missing video or actions)")
                skipped += 1
                continue
            
            video_path = os.path.join(trajectory_dir, video_rel_path, "trajectory.mp4")
            if not os.path.exists(video_path):
                print(f"\n[{idx}/{len(episodes)}] Skipping episode {episode_key} (video not found: {video_path})")
                skipped += 1
                continue
            
            print(f"\n[{idx}/{len(episodes)}] Processing episode {episode_key}")
            print(f"  Video: {video_path}")
            print(f"  Subtasks: {len(plan)}")
            print(f"  Instruction: {instruction[:80]}...")
            
            # Don't create individual output files in batch mode - will append to single file
            # Determine subtasks for this episode
            result = determine_subtasks_for_video(
                video_or_frames=video_path,
                actions=actions,
                subtask_list=plan,
                instruction=instruction,
                output_file=None,  # Don't write individual files in batch mode
                use_vlm=use_vlm,
                generate_video=generate_video
            )
            
            # Determine success/failure based on raw_response (not status)
            # If raw_response is None or empty, it's a failure
            raw_response = result.get("vlm_raw_response") if use_vlm else None
            is_success = False
            
            if use_vlm:
                # Check if raw_response exists and is not empty
                if raw_response and raw_response.strip():
                    is_success = True
                else:
                    is_success = False
            else:
                # If VLM is disabled, use status
                is_success = (result.get("status") == "success")
            
            if is_success:
                successful += 1
                print(f"  ✓ Successfully processed")
                # Build result with episode metadata at the beginning (matching deploy version format)
                result_with_metadata = {
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "status": "success",
                    **{k: v for k, v in result.items() if k != "status"}  # Add all other fields except status
                }
                
                # Immediately append to file
                f.write(json.dumps(result_with_metadata) + "\n")
                f.flush()  # Ensure data is written to disk immediately
                print(f"  ✓ Result appended to {output_file}")
                # Mark as processed
                processed_episode_keys.add(episode_key)
            else:
                failed += 1
                error_msg = result.get("error", "Unknown error")
                if use_vlm and (not raw_response or not raw_response.strip()):
                    error_msg = "VLM API returned empty response after retries"
                print(f"  ✗ Failed: {error_msg}")
                error_result = {
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "status": "failed",
                    "error": error_msg
                }
                # Immediately append error result to file
                f.write(json.dumps(error_result) + "\n")
                f.flush()  # Ensure data is written to disk immediately
                # Mark as processed (even if failed, to avoid retrying)
                processed_episode_keys.add(episode_key)
    
    print(f"\n✓ All results saved to: {output_file}")
    
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
    parser = argparse.ArgumentParser(description="Determine subtasks for each frame in a video")
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "batch"],
                       help="Mode: 'single' for a single episode, 'batch' for all episodes")
    
    # Single episode arguments
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file or frames directory (for single mode)")
    parser.add_argument("--subtask_file", type=str, default=None,
                       help="Path to subtask split file (for single mode)")
    parser.add_argument("--episode_id", type=int, default=None,
                       help="Episode ID to process (for single mode)")
    parser.add_argument("--summary_file", type=str, default=None,
                       help="Path to summary.json file to get actions (for single mode)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save results (optional, for single mode)")
    
    # Batch mode arguments
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R",
                       help="Path to trajectory data directory (for batch mode)")
    parser.add_argument("--subtask_splits_file", type=str, default="streamvln/cot_data/subtask_splits.jsonl",
                       help="Path to subtask_splits.jsonl file (for batch mode)")
    parser.add_argument("--output_dir", type=str, default="data/subtask_determination_results",
                       help="Output directory for results (for batch mode)")
    
    # Common arguments
    parser.add_argument("--no_vlm", action="store_true",
                       help="Disable VLM annotation (use fallback mode)")
    parser.add_argument("--generate_video", action="store_true",
                       help="Generate annotated video output")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Single episode mode (original functionality)
        if not args.video or not args.subtask_file or args.episode_id is None:
            print("Error: Single mode requires --video, --subtask_file, and --episode_id")
            return
        
        # Load subtask information from file
        episode_id = args.episode_id
        subtask_data = None
        instruction = None
        
        with open(args.subtask_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data["episode_id"] == episode_id:
                        subtask_data = data
                        instruction = data["instruction"]
                        break
        
        if not subtask_data:
            print(f"Error: Episode {episode_id} not found in {args.subtask_file}")
            return
        
        # Load actions from summary.json if provided
        actions = []
        if args.summary_file:
            print(f"Loading actions from summary file: {args.summary_file}")
            with open(args.summary_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("id") == episode_id:
                                actions = data.get("actions", [])
                                print(f"Found {len(actions)} actions for episode {episode_id}")
                                break
                        except json.JSONDecodeError:
                            continue
            if not actions:
                print(f"Warning: No actions found for episode {episode_id} in summary file")
        else:
            print("Warning: No summary_file provided, actions will be empty")
        
        # Determine subtasks
        result = determine_subtasks_for_video(
            video_or_frames=args.video,
            actions=actions,
            subtask_list=subtask_data["plan"],
            instruction=instruction,
            output_file=args.output_file,
            use_vlm=not args.no_vlm,
            generate_video=args.generate_video
        )
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(json.dumps({k: v for k, v in result.items() if k != "subtask_sequence"}, indent=2))
        print(f"\nSubtask sequence (first 50 frames): {result['subtask_sequence'][:50]}")
    
    elif args.mode == "batch":
        # Batch mode: process all episodes
        process_all_episodes_subtask_determination(
            trajectory_dir=args.trajectory_dir,
            subtask_splits_file=args.subtask_splits_file,
            output_dir=args.output_dir,
            use_vlm=not args.no_vlm,
            generate_video=args.generate_video
        )


if __name__ == "__main__":
    main()
