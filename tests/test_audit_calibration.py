import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from experiments.audit_calibration import (
    confusion_by_record,
    paired_record_bootstrap,
    run,
)
from experiments.pilot_data import ClipRecord, EXPECTED_LABELS
from experiments.sequence_decoder import decode_sequences, fit_transition_prior


def record(clip_id, record_id, split, start, label):
    return ClipRecord(
        clip_id=clip_id,
        split=split,
        label_id=label,
        normalized_label=EXPECTED_LABELS[label],
        source_collection="unit",
        group_id=record_id,
        video_path="/tmp/unused-video.mp4",
        clip_start_sec=float(start),
        clip_end_sec=float(start + 1),
        clip_duration_sec=1.0,
        record_id=record_id,
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

    def test_end_to_end_audit_reproduces_stored_viterbi_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [
                record("clip-train-a", "train-video", "train", 0, 0),
                record("clip-train-b", "train-video", "train", 1, 1),
                record("clip-train-c", "train-video", "train", 2, 1),
            ]
            validation = [
                record("clip-val-a", "validation-video", "val", 0, 0),
                record("clip-val-b", "validation-video", "val", 1, 1),
                record("clip-val-c", "singleton", "val", 0, 6),
            ]
            train_path, val_path = root / "train.jsonl", root / "val.jsonl"
            for path, records in ((train_path, train), (val_path, validation)):
                path.write_text(
                    "".join(json.dumps(item.to_manifest_dict()) + "\n" for item in records),
                    encoding="utf-8",
                )
            from experiments.audit_calibration import file_sha256

            hashes = {"train": file_sha256(train_path), "val": file_sha256(val_path)}
            prior = fit_transition_prior(train)
            predictions, results = [], []
            for seed in (42, 123):
                logits = torch.tensor([
                    [2.0, 0.0, 0, 0, 0, 0, 0],
                    [0.0, 1.5 + seed / 1000, 0, 0, 0, 0, 0],
                    [0.0, 0, 0, 0, 0, 0, 2.0],
                ])
                decoded = decode_sequences(logits, validation, prior)
                prediction_path = root / f"prediction-{seed}.jsonl"
                prediction_path.write_text(
                    "".join(
                        json.dumps({
                            "clip_id": item.clip_id,
                            "target": item.label_id,
                            "visual_logits": logits[index].tolist(),
                            "sequence_prediction": int(decoded[index]),
                        }) + "\n"
                        for index, item in enumerate(validation)
                    ),
                    encoding="utf-8",
                )
                result_path = root / f"result-{seed}.json"
                result_path.write_text(json.dumps({
                    "manifest_sha256": hashes,
                    "transition_weight": 1.0,
                    "transition_prior": {"smoothing": 1.0},
                    "test_metrics": None,
                }), encoding="utf-8")
                predictions.append((seed, prediction_path))
                results.append((seed, result_path))
            output = root / "audit.json"
            audit = run(Namespace(
                train_manifest=train_path,
                validation_manifest=val_path,
                prediction=predictions,
                sequence_result=results,
                bootstrap_replicates=30,
                output=output,
            ))
            self.assertTrue(output.exists())
            self.assertEqual(audit["record_cluster_bootstrap"]["num_records"], 2)
            self.assertEqual(audit["seeds"], [42, 123])
            self.assertIsNone(audit["test_metrics"])


if __name__ == "__main__":
    unittest.main()
