import json
from pathlib import Path
import tempfile
import unittest

from cvm.report import collect_runs


class ReportTests(unittest.TestCase):
    def make_run(self, root, seed, status="completed", epoch=1, lr=.1):
        path = Path(root) / f"run{seed}_{epoch}"
        path.mkdir()
        config = {"seed": seed, "backbone": "r2plus1d_18", "mode": "flat", "protocol_sha256": "a" * 64,
                  "taxonomy_definition": {"groups": [[0], [6], [1, 2, 5], [3, 4]]},
                  "code_hash": "b" * 64, "head_lr": lr, "output": "/private/patient-alice"}
        result = {"status": status, "training_completed": status == "completed", "protocol_sha256": "a" * 64}
        report = {"metadata": {"protocol_sha256": "a" * 64}, "num_recording_groups": 7,
                  "methods": {"main": {"macro_f1": .5, "accuracy": .6, "per_class": [], "confusion_matrix": []}},
                  "ambiguous_five": {"methods": {"main": {"macro_f1": .4}}}}
        for name, data in (("config", config), ("result", result), ("val_report", report)):
            (path / f"{name}.json").write_text(json.dumps(data))
        return path

    def test_incomplete_runs_are_visible_and_private_path_not_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_run(tmp, 42)
            b = self.make_run(tmp, 2026, "interrupted")
            r = collect_runs([a, b])
            self.assertFalse(r["methods"][0]["all_declared_seeds_completed"])
            self.assertEqual(r["methods"][0]["missing_seeds"], [2027])
            self.assertIsNone(r["methods"][0]["seed_summary"]["val_macro_f1"]["sd"])
            self.assertNotIn("patient-alice", json.dumps(r))
            self.assertEqual(len(r["methods"][0]["runs"]), 2)

    def test_favorable_duplicate_seed_and_changed_settings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_run(tmp, 42)
            b = self.make_run(tmp, 42, epoch=2)
            with self.assertRaises(ValueError):
                collect_runs([a, b])
            c = self.make_run(tmp, 2026, lr=.2)
            with self.assertRaises(ValueError):
                collect_runs([a, c])


if __name__ == "__main__":
    unittest.main()
