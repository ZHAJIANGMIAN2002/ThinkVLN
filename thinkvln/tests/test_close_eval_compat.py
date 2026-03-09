import unittest

from thinkvln.eval import close_eval
from thinkvln.eval.close_eval_cli import build_parser


class CloseEvalCompatibilityTest(unittest.TestCase):
    def test_facade_exports_expected_symbols(self):
        expected = [
            "VLNEvaluator",
            "eval",
            "evaluate",
            "build_subtask_spans",
            "timeline_progress",
            "compute_step_budget",
            "summarize_subtask_aggregation",
            "load_summary_full",
            "load_thinkvln_actor_model",
            "init_dist_mode",
            "get_rank",
            "get_world_size",
        ]
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue(hasattr(close_eval, name))

        self.assertTrue(callable(close_eval.eval))
        self.assertTrue(callable(close_eval.evaluate))

    def test_parser_flags_and_defaults(self):
        parser = build_parser()
        flags = set(parser._option_string_actions.keys())
        for flag in [
            "--model_path",
            "--model_type",
            "--ladder_mode",
            "--summary_full_path",
            "--target_episode_key",
            "--enable_step_debug",
            "--step_debug_format",
            "--sample_rate",
            "--subgoal_success_distance",
            "--subtask_step_budget_factor",
            "--habitat_config_path",
            "--eval_split",
            "--output_path",
            "--model_max_length",
            "--num_frames",
            "--num_future_steps",
            "--num_history",
            "--device",
            "--dist_timeout_minutes",
            "--scalar_dist_timeout_minutes",
        ]:
            with self.subTest(flag=flag):
                self.assertIn(flag, flags)

        args = parser.parse_args([])
        self.assertEqual(args.model_type, "thinkvln")
        self.assertEqual(args.ladder_mode, "subtask")
        self.assertEqual(
            parser._option_string_actions["--ladder_mode"].choices,
            ["subtask"],
        )
        self.assertEqual(args.habitat_config_path, "config/vln_r2r.yaml")
        self.assertEqual(args.eval_split, "val_unseen")
        self.assertEqual(args.output_path, "./results/env_eval")
        self.assertEqual(args.sample_rate, 1.0)
        self.assertEqual(args.model_max_length, 4096)
        self.assertEqual(args.memory_num_history_images, 6)
        self.assertEqual(args.done_threshold, 0.85)
        self.assertEqual(args.target_episode_key, "")
        self.assertEqual(args.enable_step_debug, False)
        self.assertEqual(args.step_debug_format, "none")


if __name__ == "__main__":
    unittest.main()
