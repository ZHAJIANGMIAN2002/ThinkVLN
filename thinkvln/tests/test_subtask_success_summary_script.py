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
                    {"subtask_idx": 1, "success": True},
                    {"subtask_idx": 1, "success": False},
                    {"subtask_idx": 2, "success": True},
                ],
            )
            _write_jsonl(
                root / "subtask_closed_loop_rank1.jsonl",
                [
                    {"subtask_idx": 2, "success": True},
                    {"subtask_idx": 3, "success": False},
                ],
            )

            stats = module.collect_stats(str(root))

        self.assertEqual(stats["overall"], {"success": 3, "total": 5, "rate": 0.6})
        self.assertEqual(stats["by_subtask_idx"][1], {"success": 1, "total": 2, "rate": 0.5})
        self.assertEqual(stats["by_subtask_idx"][2], {"success": 2, "total": 2, "rate": 1.0})
        self.assertEqual(stats["by_subtask_idx"][3], {"success": 0, "total": 1, "rate": 0.0})


if __name__ == "__main__":
    unittest.main()
