from dataclasses import dataclass
from typing import List

import numpy as np

from thinkvln.tools.dataset_utils import get_subtask_start_frame


@dataclass(frozen=True)
class AnchorMemoryLayout:
    anchor_frame_idx: int
    post_anchor_frame_indices: List[int]
    pre_anchor_frame_indices: List[int]
    memory_frame_indices: List[int]


def _sample_evenly(candidates: List[int], count: int) -> List[int]:
    if count <= 0 or not candidates:
        return []
    take = min(int(count), len(candidates))
    if take <= 0:
        return []
    positions = np.linspace(0, len(candidates) - 1, num=take, dtype=int).tolist()
    selected: List[int] = []
    for pos in positions:
        idx = int(candidates[pos])
        if idx not in selected:
            selected.append(idx)
    if len(selected) < take:
        for idx in candidates:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) == take:
                break
    return sorted(selected[:take])


def select_anchor_memory_layout(
    frame_idx: int,
    subtask_sequence: List[int],
    memory_post_anchor_count: int,
    memory_pre_anchor_count: int,
) -> AnchorMemoryLayout:
    current_idx = max(0, int(frame_idx))
    anchor_frame_idx = int(get_subtask_start_frame(current_idx, subtask_sequence))
    pre_candidates = list(range(max(0, anchor_frame_idx)))
    post_candidates = list(range(anchor_frame_idx + 1, current_idx))

    post_anchor = _sample_evenly(post_candidates, int(memory_post_anchor_count))
    pre_anchor = _sample_evenly(pre_candidates, int(memory_pre_anchor_count))

    target_memory = max(0, int(memory_post_anchor_count) + int(memory_pre_anchor_count))
    selected = set(post_anchor + pre_anchor)
    if len(selected) < target_memory:
        for candidate_pool in (post_candidates, pre_candidates):
            for idx in candidate_pool:
                idx = int(idx)
                if idx == anchor_frame_idx or idx in selected:
                    continue
                selected.add(idx)
                if len(selected) == target_memory:
                    break
            if len(selected) == target_memory:
                break

    return AnchorMemoryLayout(
        anchor_frame_idx=anchor_frame_idx,
        post_anchor_frame_indices=post_anchor,
        pre_anchor_frame_indices=pre_anchor,
        memory_frame_indices=sorted(selected),
    )
