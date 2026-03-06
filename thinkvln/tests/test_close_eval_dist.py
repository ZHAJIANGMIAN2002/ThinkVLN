import unittest
from datetime import timedelta
from unittest import mock

from thinkvln.eval import close_eval_dist


class CloseEvalDistTest(unittest.TestCase):
    def test_init_dist_mode_supports_custom_timeouts(self):
        with mock.patch.dict(
            "os.environ",
            {"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "1"},
            clear=False,
        ):
            with mock.patch.object(close_eval_dist.dist, "is_initialized", return_value=False), mock.patch.object(
                close_eval_dist.dist, "init_process_group"
            ) as init_pg_mock, mock.patch.object(close_eval_dist.dist, "new_group", return_value="scalar_group") as new_group_mock, mock.patch.object(
                close_eval_dist.torch.cuda, "set_device"
            ) as set_device_mock:
                rank, world_size, gpu = close_eval_dist.init_dist_mode(
                    timeout_minutes=42,
                    scalar_timeout_minutes=75,
                )

        self.assertEqual(rank, 2)
        self.assertEqual(world_size, 4)
        self.assertEqual(gpu, 1)
        init_pg_mock.assert_called_once_with(backend="nccl", timeout=timedelta(minutes=42))
        new_group_mock.assert_called_once_with(backend="gloo", timeout=timedelta(minutes=75))
        set_device_mock.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
