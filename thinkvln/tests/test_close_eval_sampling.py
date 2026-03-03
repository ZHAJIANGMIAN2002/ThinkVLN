import unittest

from thinkvln.eval.close_eval_utils import should_sample_episode


class CloseEvalSamplingTest(unittest.TestCase):
    def test_should_sample_episode_keeps_all_at_rate_one(self):
        for episode_id in range(20):
            self.assertTrue(should_sample_episode("scene_a", episode_id, 1.0))

    def test_should_sample_episode_is_deterministic(self):
        sample_a = [i for i in range(200) if should_sample_episode("scene_b", i, 0.3)]
        sample_b = [i for i in range(200) if should_sample_episode("scene_b", i, 0.3)]
        self.assertEqual(sample_a, sample_b)

    def test_should_sample_episode_reduces_episode_count(self):
        all_ids = list(range(200))
        sampled = [i for i in all_ids if should_sample_episode("scene_c", i, 0.2)]
        self.assertGreater(len(sampled), 0)
        self.assertLess(len(sampled), len(all_ids))


if __name__ == "__main__":
    unittest.main()
