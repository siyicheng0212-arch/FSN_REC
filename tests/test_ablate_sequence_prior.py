import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiments.ablate_sequence_prior import (
    ablation_predictions,
    run as run_ablation,
    sha256,
    shuffled_training_records,
)
from experiments.compare_structured_decoders import run as run_comparison
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
        video_path="/tmp/unused.mp4",
        clip_start_sec=float(start),
        clip_end_sec=float(start + 1),
        clip_duration_sec=1.0,
        record_id=record_id,
    )


class SequencePriorAblationTest(unittest.TestCase):
    def test_uniform_prior_matches_independent_visual_decisions(self):
        training = [
            record("clip-t0", "train-record", "train", 0, 0),
            record("clip-t1", "train-record", "train", 1, 1),
            record("clip-t2", "train-record", "train", 2, 2),
        ]
        validation = [
            record("clip-v0", "validation-record", "val", 0, 0),
            record("clip-v1", "validation-record", "val", 1, 1),
            record("clip-v2", "validation-record", "val", 2, 2),
        ]
        logits = torch.tensor([
            [2.0, 1.0, 0, 0, 0, 0, 0],
            [1.0, 2.0, 0, 0, 0, 0, 0],
            [0.0, 1.0, 2.0, 0, 0, 0, 0],
        ])
        methods = ablation_predictions(logits, training, validation)
        self.assertEqual(
            methods["uniform_prior"].tolist(),
            methods["visual_only"].tolist(),
        )
        shuffled = shuffled_training_records(training)
        self.assertEqual(
            sorted(item.label_id for item in shuffled),
            sorted(item.label_id for item in training),
        )
        self.assertEqual(set(methods), {
            "visual_only", "uniform_prior", "start_prior_only",
            "transition_prior_only", "full_bigram", "shuffled_train_order",
            "shuffled_inference_order", "record_mean_logit_pooling",
        })

    def test_full_ablation_and_comparison_reproduce_frozen_v3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = [
                record(f"clip-train-{step}", "train-record", "train", step, label)
                for step, label in enumerate((0, 1, 2, 3, 3, 4, 4, 5))
            ]
            validation = [
                record(f"clip-val-{step}", "val-record", "val", step, label)
                for step, label in enumerate((0, 1, 2, 3, 3, 4, 4, 5))
            ]
            train_path, val_path = root / "train.jsonl", root / "val.jsonl"
            for path, records in ((train_path, training), (val_path, validation)):
                path.write_text(
                    "".join(json.dumps(item.to_manifest_dict()) + "\n" for item in records),
                    encoding="utf-8",
                )
            hashes = {"train": sha256(train_path), "val": sha256(val_path)}
            prior = fit_transition_prior(training)
            predictions, results = [], []
            for seed in (42, 123):
                logits = torch.full((len(validation), 7), -2.0)
                for index, row in enumerate(validation):
                    logits[index, row.label_id] = 2.0 + seed / 1000
                decoded = decode_sequences(logits, validation, prior)
                prediction_path = root / f"predictions-{seed}.jsonl"
                prediction_path.write_text(
                    "".join(
                        json.dumps({
                            "clip_id": row.clip_id,
                            "target": row.label_id,
                            "visual_logits": logits[index].tolist(),
                            "sequence_prediction": int(decoded[index]),
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
            common = dict(
                train_manifest=train_path,
                validation_manifest=val_path,
                prediction=predictions,
                sequence_result=results,
            )
            ablation = run_ablation(Namespace(**common, output=root / "ablation.json"))
            comparison = run_comparison(Namespace(
                **common, weight=1.0, smoothing=1.0,
                output=root / "comparison.json",
            ))
            self.assertEqual(ablation["manifest_sha256"], hashes)
            self.assertEqual(comparison["manifest_sha256"], hashes)
            self.assertEqual(ablation["seeds"], [42, 123])
            self.assertIsNone(comparison["test_metrics"])
            self.assertAlmostEqual(
                ablation["aggregate"]["full_bigram"]["macro_f1_mean"],
                comparison["aggregate"]["bigram"]["macro_f1_mean"],
            )
            self.assertTrue((root / "ablation.json").exists())
            self.assertTrue((root / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
