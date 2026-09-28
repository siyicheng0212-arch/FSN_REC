import unittest

import numpy as np
import torch

from experiments.audit_calibration import (
    confusion_by_record,
    paired_record_bootstrap,
)


class CalibrationAuditTest(unittest.TestCase):
    def test_record_confusion_preserves_cluster_units(self):
        ids, confusion = confusion_by_record(
            ["video-a", "video-a", "video-b"],
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 2, 2]),
        )
        self.assertEqual(ids, ["video-a", "video-b"])
        self.assertEqual(int(confusion[0].sum()), 2)
        self.assertEqual(int(confusion[0, 1, 2]), 1)
        self.assertEqual(int(confusion[1, 2, 2]), 1)

    def test_identical_decisions_have_zero_bootstrap_delta(self):
        confusion = np.zeros((3, 2, 7, 7), dtype=np.int64)
        confusion[:, 0, 0, 0] = 2
        confusion[:, 1, 1, 0] = 1
        result = paired_record_bootstrap(
            confusion, confusion.copy(), replicates=30, seed=42
        )
        self.assertEqual(result["macro_f1_delta_percentile_95_interval"], [0.0, 0.0])
        self.assertEqual(result["accuracy_delta_percentile_95_interval"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
