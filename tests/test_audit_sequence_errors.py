import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiments.audit_calibration import file_sha256
from experiments.audit_sequence_errors import (
    directional_case,
    run,
    temporal_context,
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
        video_path="/tmp/private-video.mp4",
        clip_start_sec=float(start),
        clip_end_sec=float(start + 1),
        clip_duration_sec=1.0,
        record_id=record_id,
    )


class SequenceErrorAuditTest(unittest.TestCase):
    def test_directional_cases_and_boundaries(self):
        self.assertEqual(
            directional_case(4, 4, 3), "new_reperfusion_to_sweep"
        )
        self.assertEqual(
            directional_case(3, 4, 3), "fixed_sweep_to_reperfusion"
        )
        self.assertEqual(
            directional_case(3, 3, 4), "new_sweep_to_reperfusion"
        )
        self.assertEqual(
            directional_case(4, 3, 4), "fixed_reperfusion_to_sweep"
        )
        self.assertIsNone(directional_case(3, 3, 3))
        context = temporal_context([
            record("clip-a", "r", "val", 0, 3),
            record("clip-b", "r", "val", 2, 4),
        ])
        self.assertTrue(context["clip-b"]["target_action_onset"])
        self.assertEqual(context["clip-b"]["previous_gap_sec"], 1.0)

    def test_private_workflow_reproduces_saved_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [
                record(f"clip-train-{i}", "train-r", "train", i, label)
                for i, label in enumerate((3, 3, 3, 4, 4))
            ]
            validation = [
                record("clip-val-a", "val-r", "val", 0, 3),
                record("clip-val-b", "val-r", "val", 2, 4),
            ]
            train_path, val_path = root / "train.jsonl", root / "val.jsonl"
            for path, records in ((train_path, train), (val_path, validation)):
                path.write_text(
                    "".join(json.dumps(row.to_manifest_dict()) + "\n" for row in records),
                    encoding="utf-8",
                )
            hashes = {"train": file_sha256(train_path), "val": file_sha256(val_path)}
            prior = fit_transition_prior(train)
            predictions, results = [], []
            for seed in (42, 123):
                logits = torch.tensor([
                    [0, 0, 0, 4.0, 0, 0, 0],
                    [0, 0, 0, 1.1, 1.2, 0, 0],
                ], dtype=torch.float32)
                decoded = decode_sequences(logits, validation, prior)
                prediction_path = root / f"predictions-{seed}.jsonl"
                prediction_path.write_text(
                    "".join(
                        json.dumps({
                            "clip_id": row.clip_id,
                            "target": row.label_id,
                            "visual_prediction": int(logits[index].argmax()),
                            "sequence_prediction": int(decoded[index]),
                            "visual_logits": logits[index].tolist(),
                        }) + "\n"
                        for index, row in enumerate(validation)
                    ),
                    encoding="utf-8",
                )
                result_path = root / f"result-{seed}.json"
                result_path.write_text(json.dumps({
                    "seed": seed,
                    "manifest_sha256": hashes,
                    "transition_weight": 1.0,
                    "transition_prior": {"smoothing": 1.0},
                    "test_metrics": None,
                }), encoding="utf-8")
                predictions.append((seed, prediction_path))
                results.append((seed, result_path))
            arguments = Namespace(
                train_manifest=train_path,
                validation_manifest=val_path,
                prediction=predictions,
                sequence_result=results,
                output_dir=root / "private_error_audit",
                max_per_category=4,
                max_per_record=2,
            )
            summary = run(arguments)
            self.assertEqual(summary["seeds"], [42, 123])
            self.assertEqual(summary["manifest_sha256"], hashes)
            self.assertIsNone(summary["test_metrics"])
            self.assertTrue((arguments.output_dir / "private_review_queue.jsonl").exists())
            self.assertNotIn("video_path", summary)
            arguments.output_dir = root / "public_results" / "unsafe"
            with self.assertRaisesRegex(ValueError, "private clip-review output"):
                run(arguments)


if __name__ == "__main__":
    unittest.main()
