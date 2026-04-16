import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "summarize_subtask_success.py"
    spec = importlib.util.spec_from_file_location("summarize_subtask_success", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class SubtaskSuccessSummaryScriptTest(unittest.TestCase):
    def test_collect_stats_aggregates_success_rates(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_jsonl(
                root / "subtask_closed_loop_rank0.jsonl",
                [
                    {"subtask_idx": 1, "success": True, "level": 1, "progress_mae": 0.10, "progress_smooth_mae": 0.05},
                    {"subtask_idx": 1, "success": False, "level": 1, "progress_mae": 0.40, "progress_smooth_mae": 0.20},
                    {"subtask_idx": 2, "success": True, "level": 2, "progress_mae": 0.20, "progress_smooth_mae": 0.10},
                ],
            )
            _write_jsonl(
                root / "subtask_closed_loop_rank1.jsonl",
                [
                    {"subtask_idx": 2, "success": True, "level": 2, "progress_mae": 0.30, "progress_smooth_mae": 0.15},
                    {"subtask_idx": 3, "success": False, "level": 3, "progress_mae": 0.50, "progress_smooth_mae": 0.25},
                ],
            )

            stats = module.collect_stats(str(root))

        self.assertEqual(stats["overall"], {"success": 3, "total": 5, "rate": 0.6})
        self.assertEqual(stats["by_subtask_idx"][1]["success"], 1)
        self.assertEqual(stats["by_subtask_idx"][1]["total"], 2)
        self.assertEqual(stats["by_subtask_idx"][1]["rate"], 0.5)
        self.assertEqual(stats["by_subtask_idx"][2]["success"], 2)
        self.assertEqual(stats["by_subtask_idx"][2]["total"], 2)
        self.assertEqual(stats["by_subtask_idx"][2]["rate"], 1.0)
        self.assertEqual(stats["by_subtask_idx"][3]["success"], 0)
        self.assertEqual(stats["by_subtask_idx"][3]["total"], 1)
        self.assertEqual(stats["by_subtask_idx"][3]["rate"], 0.0)
        self.assertEqual(stats["by_level"][1]["success"], 1)
        self.assertEqual(stats["by_level"][1]["total"], 2)
        self.assertAlmostEqual(stats["by_level"][1]["progress_mae"], 0.25)
        self.assertAlmostEqual(stats["by_level"][1]["progress_smooth_mae"], 0.125)
        self.assertAlmostEqual(stats["by_level"][2]["progress_mae"], 0.25)
        self.assertAlmostEqual(stats["by_level"][2]["progress_smooth_mae"], 0.125)


if __name__ == "__main__":
    unittest.main()
