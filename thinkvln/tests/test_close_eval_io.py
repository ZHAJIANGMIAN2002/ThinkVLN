import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from thinkvln.eval.close_eval_utils import write_jsonl_record


class CloseEvalIOTest(unittest.TestCase):
    def test_write_jsonl_record_flushes_and_fsyncs_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            payload = {"episode_id": 7, "success": 1.0}
            with path.open("w+", encoding="utf-8") as handle, mock.patch("os.fsync") as fsync_mock:
                write_jsonl_record(handle, payload, sync_to_disk=True)
                handle.seek(0)
                self.assertEqual(handle.read(), json.dumps(payload) + "\n")
                fsync_mock.assert_called_once_with(handle.fileno())

    def test_write_jsonl_record_skips_fsync_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.jsonl"
            payload = {"episode_id": 9, "spl": 0.56}
            with path.open("w+", encoding="utf-8") as handle, mock.patch("os.fsync") as fsync_mock:
                write_jsonl_record(handle, payload, sync_to_disk=False)
                handle.seek(0)
                self.assertEqual(handle.read(), json.dumps(payload) + "\n")
                fsync_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
