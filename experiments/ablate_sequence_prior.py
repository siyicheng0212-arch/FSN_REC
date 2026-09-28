#!/usr/bin/env python3
"""Ablate the train-only FSN procedural prior on frozen visual logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from experiments.metrics import compute_classification_metrics
from experiments.pilot_data import EXPECTED_LABELS, ClipRecord, load_pilot_manifest
from experiments.sequence_decoder import (
    TransitionPrior,
    _ordered_groups,
    decode_sequences,
    fit_transition_prior,
    predictions_to_logits,
)


PERMUTATION_SEED = 20260928


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_shuffle_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{PERMUTATION_SEED}|{salt}|{value}".encode()).hexdigest()


def shuffled_training_records(records: list[ClipRecord]) -> list[ClipRecord]:
    """Destroy transition order while retaining every record's class counts."""
    altered = list(records)
    for indices in _ordered_groups(records):
        labels = [records[index].label_id for index in indices]
        permuted = sorted(
            zip(indices, labels),
            key=lambda pair: fixed_shuffle_key(records[pair[0]].clip_id, "train"),
        )
        for original_index, (_, new_label) in zip(indices, permuted):
            altered[original_index] = replace(
                records[original_index],
                label_id=new_label,
                normalized_label=EXPECTED_LABELS[new_label],
            )
    return altered


def shuffled_inference_records(records: list[ClipRecord]) -> list[ClipRecord]:
    """Keep visual evidence fixed while changing the order seen by Viterbi."""
    altered = list(records)
    for indices in _ordered_groups(records):
        shuffled = sorted(
            indices,
            key=lambda index: fixed_shuffle_key(records[index].clip_id, "infer"),
        )
        for new_rank, index in enumerate(shuffled):
            altered[index] = replace(
                records[index],
                clip_start_sec=float(new_rank),
                clip_end_sec=float(new_rank + 1),
            )
    return altered


def uniformized_prior(
    prior: TransitionPrior, *, initial: bool, transition: bool
) -> TransitionPrior:
    return TransitionPrior(
        initial_probability=(
            torch.full((7,), 1 / 7, dtype=torch.float64)
            if initial else prior.initial_probability
        ),
        transition_probability=(
            torch.full((7, 7), 1 / 7, dtype=torch.float64)
            if transition else prior.transition_probability
        ),
        smoothing=prior.smoothing,
        training_sequences=prior.training_sequences,
        training_clips=prior.training_clips,
    )


def record_mean_logit_pooling(
    logits: torch.Tensor, records: list[ClipRecord]
) -> torch.Tensor:
    prediction = logits.argmax(dim=1).clone()
    for indices in _ordered_groups(records):
        index = torch.tensor(indices, dtype=torch.long)
        prediction[index] = logits[index].mean(dim=0).argmax()
    return prediction


def ablation_predictions(
    logits: torch.Tensor,
    training: list[ClipRecord],
    validation: list[ClipRecord],
) -> dict[str, torch.Tensor]:
    prior = fit_transition_prior(training, smoothing=1.0)
    shuffled_prior = fit_transition_prior(
        shuffled_training_records(training), smoothing=1.0
    )
    return {
        "visual_only": logits.argmax(dim=1),
        "uniform_prior": decode_sequences(
            logits, validation,
            uniformized_prior(prior, initial=True, transition=True),
            weight=1.0,
        ),
        "start_prior_only": decode_sequences(
            logits, validation,
            uniformized_prior(prior, initial=False, transition=True),
            weight=1.0,
        ),
        "transition_prior_only": decode_sequences(
            logits, validation,
            uniformized_prior(prior, initial=True, transition=False),
            weight=1.0,
        ),
        "full_bigram": decode_sequences(logits, validation, prior, weight=1.0),
        "shuffled_train_order": decode_sequences(
            logits, validation, shuffled_prior, weight=1.0
        ),
        "shuffled_inference_order": decode_sequences(
            logits, shuffled_inference_records(validation), prior, weight=1.0
        ),
        "record_mean_logit_pooling": record_mean_logit_pooling(logits, validation),
    }


def _metrics(
    predictions: torch.Tensor,
    records: list[ClipRecord],
    targets: torch.Tensor,
) -> dict:
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


def named_path(value: str) -> tuple[int, Path]:
    try:
        seed, path = value.split("=", 1)
        return int(seed), Path(path).resolve()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected SEED=PATH") from exc


def run(args: argparse.Namespace) -> dict:
    training = load_pilot_manifest(args.train_manifest)
    validation = load_pilot_manifest(args.validation_manifest)
    if {record.group_id for record in training} & {
        record.group_id for record in validation
    }:
        raise ValueError("train and validation groups overlap")
    paths = dict(args.prediction)
    results = dict(args.sequence_result)
    if len(paths) < 2 or set(paths) != set(results):
        raise ValueError("prediction and result seed sets must match")
    manifest_hashes = {
        "train": sha256(args.train_manifest),
        "val": sha256(args.validation_manifest),
    }
    target = torch.tensor([record.label_id for record in validation])
    expected = {record.clip_id for record in validation}
    per_seed = {}
    for seed in sorted(paths):
        saved = json.loads(results[seed].read_text(encoding="utf-8"))
        if saved.get("test_metrics") is not None:
            raise ValueError("independent test metrics must be null")
        if saved["manifest_sha256"] != manifest_hashes:
            raise ValueError("frozen manifest hash mismatch")
        prediction_rows = [
            json.loads(line)
            for line in paths[seed].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {row["clip_id"]: row for row in prediction_rows}
        if len(by_id) != len(prediction_rows) or set(by_id) != expected:
            raise ValueError("prediction clip identifiers differ from validation")
        ordered = [by_id[record.clip_id] for record in validation]
        if any(int(row["target"]) != record.label_id for row, record in zip(ordered, validation)):
            raise ValueError("prediction targets differ from manifest")
        logits = torch.tensor(
            [row["visual_logits"] for row in ordered], dtype=torch.float32
        )
        if not torch.isfinite(logits).all():
            raise ValueError("visual logits must be finite")
        methods = ablation_predictions(logits, training, validation)
        if any(
            int(row["sequence_prediction"]) != int(prediction)
            for row, prediction in zip(ordered, methods["full_bigram"])
        ):
            raise ValueError("stored FSN-v3 predictions do not reproduce")
        per_seed[str(seed)] = {}
        for name, prediction in methods.items():
            metrics = _metrics(prediction, validation, target)
            confusion = metrics["all"]["confusion_matrix"]
            per_seed[str(seed)][name] = {
                "accuracy": metrics["all"]["accuracy"],
                "macro_f1": metrics["all"]["macro_f1"],
                "weighted_f1": metrics["all"]["weighted_f1"],
                "per_class_f1": [
                    item["f1"] for item in metrics["all"]["per_class"]
                ],
                "sweep_to_reperfusion": confusion[3][4],
                "reperfusion_to_sweep": confusion[4][3],
                "changed_from_visual": int(
                    (prediction != methods["visual_only"]).sum()
                ),
                "corrected_from_visual": int(
                    ((prediction == target) & (methods["visual_only"] != target)).sum()
                ),
                "harmed_from_visual": int(
                    ((prediction != target) & (methods["visual_only"] == target)).sum()
                ),
            }
    aggregate = {}
    for name in next(iter(per_seed.values())):
        macro = [per_seed[str(seed)][name]["macro_f1"] for seed in sorted(paths)]
        base = [
            per_seed[str(seed)]["visual_only"]["macro_f1"]
            for seed in sorted(paths)
        ]
        aggregate[name] = {
            "macro_f1_mean": statistics.mean(macro),
            "macro_f1_sample_std": statistics.stdev(macro),
            "delta_vs_visual_mean": statistics.mean(
                score - reference for score, reference in zip(macro, base)
            ),
        }
    output = {
        "schema_version": "fsn-first-order-ablation-1.0",
        "fit": "train labels only",
        "evaluation_role": "exploratory_validation_not_independent_test",
        "transition_weight": 1.0,
        "smoothing": 1.0,
        "permutation_seed": PERMUTATION_SEED,
        "manifest_sha256": manifest_hashes,
        "seeds": sorted(paths),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "test_metrics": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=named_path, required=True)
    parser.add_argument("--sequence-result", action="append", type=named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.validation_manifest = args.validation_manifest.resolve()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
