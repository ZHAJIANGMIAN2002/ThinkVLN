import unittest

from thinkvln.eval.close_eval_utils import shard_items_round_robin


class CloseEvalShardingTest(unittest.TestCase):
    def test_round_robin_shard_balances_work(self):
        items = list(range(17))
        shards = [shard_items_round_robin(items, rank=i, world_size=4) for i in range(4)]
        merged = sorted(value for shard in shards for value in shard)

        self.assertEqual(merged, items)
        sizes = [len(shard) for shard in shards]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_round_robin_shard_single_rank(self):
        items = list(range(8))
        self.assertEqual(shard_items_round_robin(items, rank=0, world_size=1), items)


if __name__ == "__main__":
    unittest.main()
