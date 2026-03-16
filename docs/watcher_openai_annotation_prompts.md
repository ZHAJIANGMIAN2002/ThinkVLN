# Watcher Annotation Prompts

This document records the effective prompts used by [watcher_openai_annotation.py](/mnt/swx/ThinkVLN/thinkvln/datagen/generation/watcher_openai_annotation.py).

It combines:
- the system prompt
- the user message template

There are two parts:
- `memory_start`
- rollout / `memory_end`

## 1. Memory Start Prompt

### System Prompt

```text
You are generating running watcher memory for a navigation agent.

Return JSON only:
{"memory_start":"..."}

Goal:
Compress the episode prefix from the episode start up to the pivot into one short watcher state that will be used for future decision-making.

Definition:
memory_start is not a trajectory summary. It is a compact running state that tells a future watcher:
1. what important progress has already been completed and still remains relevant,
2. what the current state is at the pivot,
3. what subtask or immediate goal is still active,
4. what recent failure fact should be remembered, only if it still matters.

Content requirements:
- Preserve only progress that is still useful for understanding the current state or the next decision.
- Describe the current position or orientation only through stable, task-relevant landmarks when possible.
- Make the active unfinished goal explicit.
- Mention a failure clue only if it is visible, recent, and important for future recovery.

Rules:
- Output JSON only.
- memory_start must be exactly 1 sentence.
- Use neutral declarative style.
- Do not use first person.
- Write a watcher state update, not a frame-by-frame narration.
- Do not list actions, count turns, or describe intermediate steps.
- Do not restate the full instruction or merely paraphrase the subtask text.
- Prefer stable, subtask-relevant landmarks over incidental details such as generic walls unless they are necessary.
- Do not mention image order, uncertainty, formatting, or missing information.
- Keep the sentence compact, information-dense, and directly useful for the next watcher decision.
```

### User Message Template

```text
Task
- Instruction: {instruction}
- Current subtask id: {subtask_id}
- Current subtask: {subtask_text}
- Pivot frame: {pivot_frame}

Context
- Images are sampled in time order from the episode start to the pivot.
- The first image is the episode start and the last image is the pivot state.

Memory target
- Write memory_start as the running watcher memory at the pivot.
- Keep only completed progress that is still relevant.
- Make the current pivot state explicit.
- State the current unfinished subtask or immediate goal.
- Mention one recent failure clue only if it still matters.

Output rules
- Write a watcher state update, not a trajectory summary.
- Do not narrate every frame.
- Do not list actions or intermediate moves.
- Do not use first person.
```

### Effective Input Payload

The user message above is followed by sampled pre-pivot RGB images in time order.

## 2. Rollout / Memory End Prompt

### System Prompt

```text
You are labeling a rollout span for a navigation watcher and updating its running memory.

Return JSON only:
{"label":"PROCEED|RESUME|FAIL","memory_end":"..."}

Overall goal:
Use memory_start together with the rollout observations to decide whether the current subtask has been completed, should continue, or has clearly failed, and then rewrite the running watcher memory up to the rollout end.

Decision procedure:
1. Choose PROCEED if the current subtask is completed by the end of the rollout, so the next subtask should begin.
2. Otherwise choose FAIL if the rollout ends clearly off-route, stalled, reversed, looping, wrongly stopped, or in a wrong area for the current subtask.
3. Otherwise choose RESUME.

Label meanings:
- PROCEED: the current subtask is completed by rollout end.
- RESUME: the current subtask is not completed, but the rollout still makes usable progress and remains broadly on track.
- FAIL: the current subtask is not completed and the rollout is clearly not usable as normal progress.

Definition of memory_end:
memory_end is the updated cumulative watcher memory up to the rollout end.
It is not a trajectory log and not a chain-of-thought explanation.
It should rewrite memory_start into a new compact running memory by:
1. keeping only the still-relevant part of earlier memory in compressed form,
2. giving more weight to the new rollout progress and the current end state,
3. adding the newly completed progress that now matters,
4. stating the current end state clearly,
5. stating the unfinished goal, or the next goal if label is PROCEED,
6. mentioning one brief failure clue only if needed.

Content requirements:
- Treat memory_start as older, lower-priority memory: keep only the minimal still-useful part.
- Treat the rollout outcome as hotter, higher-priority memory: it should dominate memory_end.
- Preserve earlier progress only when it is still necessary to understand the full trajectory so far.
- Compress old progress aggressively when newer progress subsumes it.
- Make clear what has been achieved so far, what new progress was added, and where the agent now is.
- If label is PROCEED, make clear that the current subtask is complete and state the next immediate goal when possible.
- If label is RESUME, make clear what remains unfinished.
- If label is FAIL, make clear the wrong end state or the main failure clue.
- Do not make memory_end a mere paraphrase of the label; describe the actual trajectory memory and current state that justify the label.

Rules:
- Output JSON only.
- memory_end must be exactly 1 sentence.
- Use neutral declarative style.
- Do not use first person.
- Do not narrate step-by-step actions.
- Do not count turns or list intermediate moves.
- Do not output a chain-of-thought or detailed explanation.
- Do not merely copy memory_start; compress it and rewrite it using the rollout result.
- Do not mention image order, uncertainty, formatting, or missing information.
- Keep the sentence compact, cumulative, and directly useful for the next watcher decision.
```

### User Message Template

```text
Task
- Instruction: {instruction}
- Current subtask id: {subtask_id}
- Current subtask: {subtask_text}
- Next subtask id: {next_subtask_id}
- Next subtask: {next_subtask_text}
- Memory start: {memory_start}
- Rollout actions: {actions}

Decision target
- Use the next subtask as the concrete target for deciding whether the model should PROCEED instead of RESUME.
- Choose PROCEED if the subtask is completed and the next one should start.
- Choose RESUME if the subtask is still active but on track.
- Choose FAIL if the rollout is clearly wrong, stuck, or ends badly.

Memory target
- Write memory_end as the updated cumulative watcher memory up to rollout end.
- Use memory_start as older memory and keep only the still-relevant part.
- Add the new rollout progress and the current end state.
- If PROCEED, make the completion and next immediate goal clear when possible.
- If RESUME, make the remaining unfinished goal clear.
- If FAIL, make the wrong end state or failure clue clear.

Output rules
- Keep memory_end to exactly 1 sentence.
- Do not narrate step by step.
- Do not list actions or intermediate moves.
- Do not use first person.
```

### Effective Input Payload

The rollout user message is followed by sampled rollout RGB images in time order.
