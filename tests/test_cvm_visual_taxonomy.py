import unittest

import numpy as np

from cvm.visual_taxonomy import derive_visual_taxonomy


class VisualTaxonomyTests(unittest.TestCase):
    def test_partition_is_deterministic_and_matches_prescribed_sizes(self):
        proto = np.eye(7)
        features = np.repeat(proto, 3, axis=0)
        labels = np.repeat(np.arange(7), 3)
        result = derive_visual_taxonomy(features, labels)
        reversed_result = derive_visual_taxonomy(features[::-1], labels[::-1])
        self.assertEqual(result["groups"], reversed_result["groups"])
        self.assertEqual(result["group_sizes"], [1, 1, 3, 2])
        self.assertEqual(result["num_candidate_partitions"], 210)
        self.assertEqual(sorted(sum(result["groups"], [])), list(range(7)))

    def test_known_visual_clusters_are_recovered(self):
        prototypes = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                               [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        r = derive_visual_taxonomy(np.repeat(prototypes, 2, axis=0), np.repeat(np.arange(7), 2))
        self.assertEqual(r["groups"], [[0], [1], [2, 3, 4], [5, 6]])

    def test_missing_classes_and_zero_features_rejected(self):
        with self.assertRaises(ValueError):
            derive_visual_taxonomy(np.ones((12, 4)), np.repeat(np.arange(6), 2))
        with self.assertRaises(ValueError):
            derive_visual_taxonomy(np.zeros((14, 4)), np.repeat(np.arange(7), 2))


if __name__ == "__main__":
    unittest.main()
