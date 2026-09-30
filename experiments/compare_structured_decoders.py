#!/usr/bin/env python3
"""Compare fixed train-only first-, second-, and segmental FSN decoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from experiments.metrics import compute_classification_metrics
from experiments.pilot_data import load_pilot_manifest
from experiments.sequence_decoder import (
    decode_sequences,
    fit_transition_prior,
    predictions_to_logits,
)
from experiments.structured_decoders import (
    decode_second_order_sequences,
    decode_segmental_sequences,
    fit_second_order_prior,
    fit_segmental_prior,
)


def _read_predictions(path: Path) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {row["clip_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate clip ids in {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(records, targets, predictions):
    metadata = [
        {
            "clip_id": record.clip_id,
            "source_collection": record.source_collection,
            "clip_duration_sec": record.clip_duration_sec,
        }
        for record in records
    ]
    return compute_classification_metrics(
        predictions_to_logits(predictions), targets, metadata
    )


def _compact(metrics: dict, visual_macro: float) -> dict:
    block = metrics["all"]
    confusion = block["confusion_matrix"]
    return {
        "accuracy": block["accuracy"],
        "macro_f1": block["macro_f1"],
        "weighted_f1": block["weighted_f1"],
        "delta_macro_f1_vs_visual": block["macro_f1"] - visual_macro,
        "per_class_f1": [item["f1"] for item in block["per_class"]],
        "sweep_to_reperfusion": confusion[3][4],
        "reperfusion_to_sweep": confusion[4][3],
        "duration_le_10s": metrics["slices"]["duration_le_10s"],
        "duration_gt_10s": metrics["slices"]["duration_gt_10s"],
    }


def run(args: argparse.Namespace) -> dict:
    if args.weight < 0 or args.smoothing <= 0:
        raise ValueError("weight must be non-negative and smoothing positive")
    train = load_pilot_manifest(args.train_manifest)
    validation = load_pilot_manifest(args.validation_manifest)
    if {record.group_id for record in train} & {
        record.group_id for record in validation
    }:
        raise ValueError("train and validation groups overlap")
    prediction_paths = dict(args.prediction)
    result_paths = dict(args.sequence_result)
    if len(prediction_paths) < 2 or set(prediction_paths) != set(result_paths):
        raise ValueError("prediction and result seed sets must match")
    manifest_hashes = {
        "train": _sha256(args.train_manifest),
        "val": _sha256(args.validation_manifest),
    }
    first = fit_transition_prior(train, args.smoothing)
    second = fit_second_order_prior(train, args.smoothing)
    segmental = fit_segmental_prior(train, args.smoothing)
    per_seed = {}
    for seed, path in sorted(prediction_paths.items()):
        saved = json.loads(result_paths[seed].read_text(encoding="utf-8"))
        if saved.get("test_metrics") is not None:
            raise ValueError("independent test metrics must be null")
        if saved.get("seed") != seed or saved.get("manifest_sha256") != manifest_hashes:
            raise ValueError("seed or frozen manifest hash mismatch")
        if (
            saved.get("transition_weight") != args.weight
            or saved.get("transition_prior", {}).get("smoothing") != args.smoothing
        ):
            raise ValueError("stored transition settings mismatch")
        rows = _read_predictions(path)
        if set(rows) != {record.clip_id for record in validation}:
            raise ValueError(f"prediction clip set does not match validation: {path}")
        ordered = [rows[record.clip_id] for record in validation]
        if any(
            int(row["target"]) != record.label_id
            for row, record in zip(ordered, validation)
        ):
            raise ValueError("prediction targets differ from manifest")
        logits = torch.tensor(
            [row["visual_logits"] for row in ordered],
            dtype=torch.float32,
        )
        if not torch.isfinite(logits).all():
            raise ValueError("visual logits must be finite")
        targets = torch.tensor([record.label_id for record in validation])
        visual_prediction = logits.argmax(dim=1)
        visual = _metric(validation, targets, visual_prediction)
        visual_macro = visual["all"]["macro_f1"]
        decoded = {
            "bigram": decode_sequences(logits, validation, first, args.weight),
            "trigram": decode_second_order_sequences(
                logits, validation, second, args.weight
            ),
            "semi_markov": decode_segmental_sequences(
                logits, validation, segmental, args.weight
            ),
        }
        if any(
            int(row["sequence_prediction"]) != int(prediction)
            for row, prediction in zip(ordered, decoded["bigram"])
        ):
            raise ValueError("stored FSN-v3 predictions do not reproduce")
        per_seed[str(seed)] = {
            "visual": _compact(visual, visual_macro),
            "models": {
                name: {
                    **_compact(_metric(validation, targets, prediction), visual_macro),
                    "changed_predictions": int(
                        (prediction != visual_prediction).sum()
                    ),
                }
                for name, prediction in decoded.items()
            },
        }
    for seed_result in per_seed.values():
        bigram = seed_result["models"]["bigram"]["macro_f1"]
        for name in ("trigram", "semi_markov"):
            seed_result["models"][name]["delta_macro_f1_vs_bigram"] = (
                seed_result["models"][name]["macro_f1"] - bigram
            )
    aggregate = {}
    for name in ("bigram", "trigram", "semi_markov"):
        macro = [item["models"][name]["macro_f1"] for item in per_seed.values()]
        delta_visual = [
            item["models"][name]["delta_macro_f1_vs_visual"]
            for item in per_seed.values()
        ]
        aggregate[name] = {
            "macro_f1_mean": statistics.mean(macro),
            "macro_f1_sample_std": statistics.stdev(macro),
            "delta_vs_visual_mean": statistics.mean(delta_visual),
            "all_seeds_improve_visual": min(delta_visual) >= 0,
        }
        if name != "bigram":
            delta_bigram = [
                item["models"][name]["delta_macro_f1_vs_bigram"]
                for item in per_seed.values()
            ]
            aggregate[name].update(
                {
                    "delta_vs_bigram_mean": statistics.mean(delta_bigram),
                    "all_seeds_improve_bigram": min(delta_bigram) >= 0,
                }
            )
    result = {
        "schema_version": "fsn-structured-decoder-comparison-1.0",
        "fit": "training labels only",
        "evaluation_role": "exploratory_validation_not_independent_test",
        "weight": args.weight,
        "smoothing": args.smoothing,
        "manifest_sha256": manifest_hashes,
        "seeds": sorted(prediction_paths),
        "max_train_segment_length": segmental.max_duration,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "test_metrics": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_prediction(value: str) -> tuple[int, Path]:
    try:
        seed, path = value.split("=", 1)
        return int(seed), Path(path).resolve()
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("prediction must be SEED=PATH") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", type=parse_prediction, required=True
    )
    parser.add_argument(
        "--sequence-result", action="append", type=parse_prediction, required=True
    )
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.validation_manifest = args.validation_manifest.resolve()
    args.output = args.output.resolve()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
