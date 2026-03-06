# StreamVLN Memory to ThinkVLN Actor (Brief)

## StreamVLN memory mechanics

StreamVLN uses a memory placeholder token (`<memory>`) in text prompt plus sampled historical visual observations:

1. Prompt contains `<memory>` only when historical context is available.
2. History frames are sampled sparsely from earlier trajectory steps.
3. Current frame is always kept.
4. The multimodal model injects history image features at `<memory>` placeholder positions.
5. Fast context (current chunk) and slow context (memory) are balanced by limiting history count.

Core effect: model sees compressed long-horizon context while keeping runtime bounded.

## ThinkVLN adaptation (actor)

For ThinkVLN Actor we keep model changes minimal and implement memory at collator/inference-wrapper level:

1. Use visual-memory prompt marker (`<memory>`) in action prompt when history exists.
2. Build memory image set with anchors:
   - episode first frame
   - current subtask first frame
3. Fill remaining memory slots by sparse-uniform sampling from past frames.
4. Always include current frame.
5. Enforce image-token budget:
   - configurable history cap
   - configurable total image-token budget
   - trim sparse non-anchor history first.

## Sequential supervision changes

1. Keep 4-step action supervision unchanged.
2. Progress supervision becomes scalar current-step progress.
3. Input prompt includes teacher-forced previous-step progress during training.
4. Add binary done target: `done = (current_progress > 0.85)`.
5. Keep episode memory accumulation across subtasks, but reset previous-progress state at subtask boundary.

