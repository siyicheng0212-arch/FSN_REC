import unittest
import torch
from experiments.context_evidence.worker import sampling_orders
from experiments.context_evidence.multiseed import assignments, summary
from experiments.context_evidence.run import PAIRED_VARIANTS


class MultiSeedTest(unittest.TestCase):
    def test_pair_same_seed_gets_identical_orders(self):
        for seed in (42, 10, 1217):
            a, b = sampling_orders(seed, 47, 31)
            c, d = sampling_orders(seed, 47, 31)
            self.assertTrue(torch.equal(a, c) and torch.equal(b, d))
            self.assertEqual(a.shape, (30, 47))
            self.assertTrue(torch.equal(a.sort().values, torch.arange(47).expand(30, 47)))

    def test_new_seeds_really_change_sampling(self):
        for left, right in ((42, 10), (42, 1217), (10, 1217)):
            self.assertFalse(torch.equal(sampling_orders(left, 47, 31)[0], sampling_orders(right, 47, 31)[0]))

    def test_seed42_exact_original_generator_order(self):
        generator = torch.Generator().manual_seed(42)
        expected_a = torch.stack([torch.randperm(47, generator=generator) for _ in range(30)])
        expected_b = torch.stack([torch.randperm(31, generator=generator) for _ in range(30)])
        a, b = sampling_orders(42, 47, 31)
        self.assertTrue(torch.equal(a, expected_a) and torch.equal(b, expected_b))

    def test_distinct_four_card_registered_tasks_only(self):
        tasks = assignments([0, 1, 2, 3], ['a', 'b', 'c', 'd'])
        self.assertEqual([x[0] for x in tasks], [10, 10, 1217, 1217])
        self.assertEqual([x[1] for x in tasks], list(PAIRED_VARIANTS)*2)
        with self.assertRaises(ValueError):
            assignments([0, 0, 2, 3], ['a', 'b', 'c', 'd'])

    def test_paired_sample_standard_deviation(self):
        rows = {str(s): {v: {'metrics': {'macro_f1': .8+i*.01+j*.02, 'accuracy': .9}}
                         for j, v in enumerate(PAIRED_VARIANTS)} for i, s in enumerate((42, 10, 1217))}
        r = summary(rows)
        self.assertAlmostEqual(r[PAIRED_VARIANTS[0]]['macro_f1_sample_std'], .01)
        self.assertAlmostEqual(r['paired_F1_delta']['mean'], .02)


if __name__ == '__main__':
    unittest.main()
