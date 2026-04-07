from streamvln.dataset.streamvln_actor_history_layout import select_anchor_memory_layout


def test_select_anchor_memory_layout_prioritizes_subtask_start_and_keeps_anchor_out_of_memory():
    subtask_sequence = [1, 1, 1, 2, 2, 2, 2, 2, 3]

    layout = select_anchor_memory_layout(
        frame_idx=7,
        subtask_sequence=subtask_sequence,
        memory_post_anchor_count=3,
        memory_pre_anchor_count=2,
    )

    assert layout.anchor_frame_idx == 3
    assert layout.anchor_frame_idx not in layout.memory_frame_indices
    assert layout.post_anchor_frame_indices == [4, 5, 6]
    assert layout.pre_anchor_frame_indices == [0, 2]
    assert layout.memory_frame_indices == [0, 2, 4, 5, 6]


def test_select_anchor_memory_layout_handles_first_subtask_frame_without_post_history():
    subtask_sequence = [1, 1, 1, 2, 2, 2]

    layout = select_anchor_memory_layout(
        frame_idx=3,
        subtask_sequence=subtask_sequence,
        memory_post_anchor_count=4,
        memory_pre_anchor_count=2,
    )

    assert layout.anchor_frame_idx == 3
    assert layout.post_anchor_frame_indices == []
    assert layout.pre_anchor_frame_indices == [0, 2]
    assert layout.memory_frame_indices == [0, 1, 2]
