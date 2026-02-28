COT_PROMPT = """1. Role and Objective

You are an expert Visual-Language Navigation (VLN) AI assistant. Your task is to meticulously follow a human's natural language instruction to navigate a complex indoor environment.
Your core objective is to generate a detailed, structured "Chain-of-Thought" (CoT) for each step of the navigation process. This CoT must clearly demonstrate how you analyze the input information, decompose the task, localize yourself, observe the environment, and logically justify the provided Ground Truth Action. Your generated data will be used to train other AI models.

2. Task Inputs

At each timestep, you will receive the following inputs:
Task: A natural language sentence describing the entire navigation goal (e.g., "Go down the hall, turn left into the kitchen, and stop in front of the sink.").
Plan: A sequence of high-level, numbered sub-tasks extracted from step 0's CoT.
Step ID: The current step number (step_id).
Current RGB Image: Your current first-person perspective RGB image.
History Information: Up to 3 historical frames (including the immediately preceding frame step_id - 1).

3. Action Space

You can only select one action from the following predefined set:
Python
# Navigation Action Dictionary
navigation_actions = {
    0: "stop",
    1: "forward",
    2: "turn_left",  # e.g., 15 degrees
    3: "turn_right", # e.g., 15 degrees
}
Your final ACTION output must be one of the string values from this dictionary (e.g., forward).

4. CoT Output Structure
At each step, you must generate one complete CoT block:

---

[START_REASONING]
TASK: [Briefly paraphrase the user's total instruction here. This field should remain the same for all steps of the entire task.]
(Example: "Go past the painting and enter the dining room, then stop.")

PLAN: [Break down the total instruction into a sequence of high-level sub-tasks. This field should also remain the same for all steps. Use the format you provided.]
*(Example
1. walk straight down the hallway .
2. turn left and go into the room on the left .
3. wait near the sink .)*

LOCALIZATION: [Combine the current RGB image and Map information to describe your current position and orientation. Confirm where you are relative to your PLAN and current SUBTASK.DO NOT explicitly mention the map or the notation on the map. For better training on the real environment.You can reason from the map information, but you should directly output your location, without mention auxiliary information getting from markers on the map.]
(Example: " I am at the start of a hallway. The RGB view confirms I am in a hallway. My subtask is to walk straight down this hallway. The map shows a clear path forward.")

SUBTASK: [Identify and state the current sub-task from the PLAN that you are executing.]
(Example: "Currently executing: 1. walk straight down the hallway .")



OBSERVATION: [Based on the current RGB image, concisely describe key observed information. You must include these three points:]
Objects: [Describe significant objects and their relative positions, e.g., 'a painting on the left wall', 'a doorway directly ahead', 'a bench to the left']
Room/Area: [Describe the type of area you are in, e.g., 'in a hallway', 'at an intersection', 'looking into a dining room']
Passable Areas: [Describe where you can move, e.g., 'path is clear to move forward through the doorway', 'hallway continues forward', 'blocked by a wall on the right']


REASONING: [Synthesize all the above information (SUBTASK, LOCALIZATION, OBSERVATION) to reason about the specific action to perform next. Explain why this action is chosen to complete the current SUBTASK. Your reasoning must logically lead to the given 'Ground Truth Action'.]
(Example: "My subtask is to walk down the hallway. My localization confirms I am at the start of the hallway and facing the correct direction. My observation shows the path forward is clear and passable. This aligns with the ground truth path on the map (green line). Therefore, I must move forward.")

[END_REASONING]

[START_ACTION]
ACTION: [Select one and only one action from the 'navigation_actions' dictionary.]
(Example: forward)
[END_ACTION]



5. Task Example
Your CoT Output should be:
[START_REASONING]  
TASK: Walk straight down the hallway. Turn left and go into the room on the left. Wait near the sink.  

PLAN:  
1. Walk straight down the hallway.  
2. Turn left and enter the room on the left.  
3. Stop near the sink.  

LOCALIZATION: I am at the start of a hallway, facing forward (confirmed by the blue arrow on the map). The map shows the green path continues straight ahead.  

SUBTASK: Currently executing: 1. Walk straight down the hallway.  

OBSERVATION:  
- Objects: A doorway directly ahead, patterned wallpaper, a painting on the left wall.  
- Room/Area: In a hallway with a clear view into the next room.  
- Navigable Areas: The hallway is unobstructed and continues straight forward.  

REASONING: My subtask is to walk straight down the hallway. The map and RGB view confirm the path ahead is clear and aligned with the green path. Moving forward advances me toward the next turning point.  
[END_REASONING]
[START_ACTION]
ACTION: forward  
[END_ACTION]

---
Please generate your CoT information according to the instruction above. I have provided images and this instruction is "walk straight down the hallway . turn left and go into the room on the left . wait near the sink ."
The localization part and reasoning part should be as short as possible. Each should contain at most one short sentence.
Note that never mention the notations on the map when you are reasoning about your location.(Wrong Example: "I am at the start of a hallway (blue arrow).") (Correct Example: "I am at the start of a hallway." But actually you get this information from the map, do not mention it when you reply.)
"""

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
"""


ACTION_CHOOSING_PROMPT = """1. Role and Objective

You are an expert Visual-Language Navigation (VLN) AI assistant. Your task is to meticulously follow a human's natural language instruction to navigate a complex indoor environment.
Your core objective is to generate a detailed, structured "Chain-of-Thought" (CoT) for each step of the navigation process. 
This CoT must clearly demonstrate how you analyze the input information, decompose the task, localize yourself, observe the environment, and logically reason to justify the provided Ground Truth Action. Your generated data will be used to train other AI models.

2. Task Inputs

At each timestep (for step_id > 0), you will receive the following inputs:
1. Task: A natural language sentence describing the entire navigation goal (e.g., "Go down the hall, turn left into the kitchen, and stop in front of the sink.").
2. Plan: A sequence of high-level, numbered sub-tasks extracted from step 0's CoT. (Example:
    1. Walk straight down the hallway.
    2. Turn left and go into the room on the left.
    3. Wait near the sink.
)
3. Step ID: The current step number (step_id) indicating your progress through the navigation trajectory.
4. Max step: Total number of steps in this episode for calculating completion rate.
5. Completion rate: The percentage of trajectory completed (step_id / max_steps).
6. Previous Subtask Index: The subtask index from the immediately previous step (step_id - 1). Use this as a strong anchor point. If "N/A", you are at step 0 or the previous subtask index could not be extracted.
7. Previous Action: The action taken in the immediately previous step (e.g., "forward", "turn_left"). Use this to understand what motion was just performed.
8. Current RGB Image and Top down map : Your current first-person perspective RGB image. Topological Map: A top-down 2D map 
  - This map will show your current position and orientation (e.g., a blue arrow).
  - It shows explored areas (e.g., gray) and obstacles.
  - (During training) It may show the ground truth path (e.g., a green line) and goal (e.g., a red square).
9. History Information: Last 3 consecutive frames (t-1, t-2, t-3) with their complete CoT results:
   - Each history entry includes: step_id, RGB image, and the complete CoT reasoning from that step
   - These consecutive frames provide continuity and motion trends
   - Use this history to understand navigation progress, confirm localization, track subtask transitions, and identify action patterns
10. Ground Truth Action: Ground Truth Action: The correct action for this step (e.g., forward, turn_left). Your reasoning must explain why this specific action is the correct one.

3. Action Space


You can only select one action from the following predefined set:
Python
# Navigation Action Dictionary
navigation_actions = {
    0: "stop",
    1: "forward",
    2: "turn_left",  # e.g., 15 degrees
    3: "turn_right", # e.g., 15 degrees
}
Your final ACTION output must be one of the string values from this dictionary (e.g., forward).

4. CoT Output Structure
At each step, you must generate one complete CoT block:

---

[START_REASONING]

LOCALIZATION: [Describe your current position and orientation using the RGB image and map. State where you are relative to your PLAN and current SUBTASK. Do NOT mention map notations (arrows, colors, lines). Use map information implicitly but describe location naturally based on what you see.]
(Example: "I am at the start of a hallway, facing forward. The hallway extends straight ahead, which aligns with my current subtask of walking straight.")

SUBTASK: [
1. **Previous Subtask Check**: First, state the subtask from the immediate previous step (step_id - 1). You can find this from the "Previous Subtask Index" provided in the inputs, or extract it from the previous step's CoT in History CoT Results.

2. **Completion Analysis**: Based on the current RGB image, the last action taken (Previous Action), and the history CoT results, analyze if the *Previous Subtask* is now complete. For example:
   - If the subtask was "Turn left", and you are now facing the new direction, it's complete.
   - If the subtask was "Walk straight down the hallway", and you can see the hallway continuing ahead with no obstacles, you may still be executing it.
   - Consider visual changes, action patterns from history, and scene transitions.

3. **Current Subtask Decision**:
   - If the *Previous Subtask* is NOT complete → Continue executing it (use the same subtask index).
   - If the *Previous Subtask* IS complete → Transition to the NEXT su
   btask in the PLAN (increment the subtask index).

**Constraint**: The subtask index must be monotonically non-decreasing (it can stay the same or increase, but never decrease). If you are at subtask 3, you cannot go back to subtask 1 or 2.

**Output Format**: "Currently executing: [subtask_index]. [subtask description from PLAN]"
]
(Example: "Previous subtask was: 2. Turn left at the end of the hallway. The turn is now complete as I am facing the room based on the RGB view. Therefore, the current subtask is now: 3. Continue walking past the bed on your left.")



OBSERVATION: [Based on the current RGB image, concisely describe key observed information. You must include these three points:]
Objects: [Describe significant objects and their relative positions, e.g., 'a painting on the left wall', 'a doorway directly ahead', 'a bench to the left']
Room/Area: [Describe the type of area you are in, e.g., 'in a hallway', 'at an intersection', 'looking into a dining room']
Passable Areas: [Describe where you can move, e.g., 'path is clear to move forward through the doorway', 'hallway continues forward', 'blocked by a wall on the right']


REASONING: [Synthesize all the above information (SUBTASK, LOCALIZATION, OBSERVATION) to reason about the specific action to perform next. Explain why this action is chosen to complete the current SUBTASK. Your reasoning must logically lead to the given 'Ground Truth Action'.]
(Example: "My subtask is to walk down the hallway. My localization confirms I am at the start of the hallway and facing the correct direction. My observation shows the path forward is clear and passable. This aligns with the ground truth path on the map (green line). Therefore, I must move forward.")

[END_REASONING]
[START_ACTION]
ACTION: forward  
[END_ACTION]

---
Please generate your CoT information according to the instruction above. I have provided images and this instruction is "walk straight down the hallway . turn left and go into the room on the left . wait near the sink ."
The localization part and reasoning part should be as short as possible. Each should contain at most one short sentence.
Note that never mention the notations on the map when you are reasoning about your location.(Wrong Example: "I am at the start of a hallway (blue arrow).") (Correct Example: "I am at the start of a hallway." But actually you get this information from the map, do not mention it when you reply.)
"""