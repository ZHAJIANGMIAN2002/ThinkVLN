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
            "oracle_subtask_at_step",
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
        ]:
            with self.subTest(flag=flag):
                self.assertIn(flag, flags)

        args = parser.parse_args([])
        self.assertEqual(args.model_type, "thinkvln")
        self.assertEqual(args.ladder_mode, "legacy")
        self.assertEqual(args.habitat_config_path, "config/vln_r2r.yaml")
        self.assertEqual(args.eval_split, "val_unseen")
        self.assertEqual(args.output_path, "./results/env_eval")
        self.assertEqual(args.sample_rate, 1.0)
        self.assertEqual(args.model_max_length, 4096)


if __name__ == "__main__":
    unittest.main()
