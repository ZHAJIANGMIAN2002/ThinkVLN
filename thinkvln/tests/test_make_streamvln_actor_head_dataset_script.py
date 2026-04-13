import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "make_streamvln_actor_head_dataset.py"
    spec = importlib.util.spec_from_file_location("make_streamvln_actor_head_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_head_sequences_keeps_full_first_n_sequences():
    module = _load_module()
    rows = [
        {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 0},
        {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 1},
        {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 2},
        {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 3},
        {"episode_key": "ep2", "subtask_position": "1/1", "frame_idx": 0},
    ]

    kept = module.collect_head_sequences(rows, max_sequences=2)

    assert [(row["episode_key"], row["subtask_position"], row["frame_idx"]) for row in kept] == [
        ("ep1", "1/2", 0),
        ("ep1", "1/2", 1),
        ("ep1", "2/2", 2),
        ("ep1", "2/2", 3),
    ]


def test_drop_fields_removes_unused_debug_keys():
    module = _load_module()
    row = {
        "episode_key": "ep1",
        "subtask_position": "1/2",
        "frame_idx": 0,
        "done_label": 1.0,
        "previous_progress": 0.5,
        "watcher_hint": "hint",
        "progress_label": 0.75,
    }

    cleaned = module.drop_fields([row], ["done_label", "previous_progress", "watcher_hint"])

    assert cleaned == [
        {
            "episode_key": "ep1",
            "subtask_position": "1/2",
            "frame_idx": 0,
            "progress_label": 0.75,
        }
    ]
