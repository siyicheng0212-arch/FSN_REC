#!/usr/bin/env python3
"""Audit FSN visual and sequence probabilities with record-cluster uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from experiments.pilot_data import load_pilot_manifest
from experiments.probability_metrics import probability_metrics
from experiments.sequence_decoder import (
    decode_sequences,
    fit_transition_prior,
    sequence_marginal_probabilities,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_path(value: str) -> tuple[int, Path]:
    try:
        seed, path = value.split("=", 1)
        return int(seed), Path(path).resolve()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected SEED=PATH") from exc


def read_prediction_rows(path: Path) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["clip_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate clip identifiers in {path.name}")
    return by_id


def confusion_by_record(
    record_ids: list[str], targets: torch.Tensor, predictions: torch.Tensor
) -> tuple[list[str], np.ndarray]:
    ids = sorted(set(record_ids))
    locations = {record_id: index for index, record_id in enumerate(ids)}
    confusion = np.zeros((len(ids), 7, 7), dtype=np.int64)
    for record_id, target, prediction in zip(
        record_ids, targets.tolist(), predictions.tolist()
    ):
        confusion[locations[record_id], target, prediction] += 1
    return ids, confusion


def _macro_f1(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(7, dtype=float),
        where=denominator != 0,
    )
    return float(f1.mean())


def _accuracy(confusion: np.ndarray) -> float:
    return float(np.trace(confusion) / confusion.sum())


def paired_record_bootstrap(
    visual: np.ndarray,
    sequence: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 20260928,
) -> dict:
    """Resample source records jointly across seeds and both model decisions.

    This estimates held-out-record variation conditional on these three fitted
    seeds.  It does not describe variation from training new seeds or new
    institutions, and the validation set was used for checkpoint selection.
    """
    if visual.shape != sequence.shape or visual.ndim != 4:
        raise ValueError("confusion arrays must match [seeds, records, 7, 7]")
    if visual.shape[2:] != (7, 7) or replicates <= 0:
        raise ValueError("invalid confusion shape or replicate count")
    rng = np.random.default_rng(seed)
    macro_deltas = np.empty(replicates)
    accuracy_deltas = np.empty(replicates)
    for draw in range(replicates):
        sampled = rng.integers(0, visual.shape[1], size=visual.shape[1])
        visual_draw = visual[:, sampled].sum(axis=1)
        sequence_draw = sequence[:, sampled].sum(axis=1)
        macro_deltas[draw] = np.mean([
            _macro_f1(seq) - _macro_f1(base)
            for base, seq in zip(visual_draw, sequence_draw)
        ])
        accuracy_deltas[draw] = np.mean([
            _accuracy(seq) - _accuracy(base)
            for base, seq in zip(visual_draw, sequence_draw)
        ])
    return {
        "unit": "source_record",
        "num_records": visual.shape[1],
        "num_model_seeds": visual.shape[0],
        "replicates": replicates,
        "random_seed": seed,
        "macro_f1_delta_percentile_95_interval": np.quantile(
            macro_deltas, [0.025, 0.975]
        ).tolist(),
        "accuracy_delta_percentile_95_interval": np.quantile(
            accuracy_deltas, [0.025, 0.975]
        ).tolist(),
        "macro_f1_positive_fraction": float((macro_deltas > 0).mean()),
        "conditional_scope": (
            "Record variation on the checkpoint-selected validation set; "
            "training-seed and external-site uncertainty are not included"
        ),
    }


def _mean_std(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def run(args: argparse.Namespace) -> dict:
    train = load_pilot_manifest(args.train_manifest)
    validation = load_pilot_manifest(args.validation_manifest)
    prior = fit_transition_prior(train, smoothing=1.0)
    paths = dict(args.prediction)
    result_paths = dict(args.sequence_result)
    if set(paths) != set(result_paths) or len(paths) < 2:
        raise ValueError("prediction and sequence-result seeds must match")
    labels = torch.tensor([record.label_id for record in validation], dtype=torch.long)
    record_ids = [record.record_id or record.clip_id for record in validation]
    expected_clips = {record.clip_id for record in validation}
    validation_hash = file_sha256(args.validation_manifest)
    train_hash = file_sha256(args.train_manifest)
    seed_results = {}
    visual_groups = []
    sequence_groups = []
    for seed in sorted(paths):
        existing = json.loads(result_paths[seed].read_text(encoding="utf-8"))
        if existing.get("test_metrics") is not None:
            raise ValueError("independent test metrics must be null")
        if existing["manifest_sha256"] != {
            "train": train_hash, "val": validation_hash
        }:
            raise ValueError("result manifest hash mismatch")
        if existing["transition_weight"] != 1.0 or (
            existing["transition_prior"]["smoothing"] != 1.0
        ):
            raise ValueError("transition settings differ from frozen protocol")
        rows = read_prediction_rows(paths[seed])
        if set(rows) != expected_clips:
            raise ValueError("prediction clip set differs from frozen validation")
        ordered = [rows[record.clip_id] for record in validation]
        if any(int(row["target"]) != int(target) for row, target in zip(ordered, labels)):
            raise ValueError("prediction targets differ from validation manifest")
        logits = torch.tensor([row["visual_logits"] for row in ordered], dtype=torch.double)
        visual_prediction = logits.argmax(dim=1)
        sequence_prediction = decode_sequences(logits, validation, prior, weight=1.0)
        if any(
            int(row["sequence_prediction"]) != int(prediction)
            for row, prediction in zip(ordered, sequence_prediction)
        ):
            raise ValueError("stored Viterbi predictions do not reproduce")
        visual_probability = torch.softmax(logits, dim=1)
        marginal_probability = sequence_marginal_probabilities(
            logits, validation, prior, weight=1.0
        )
        visual_metrics = probability_metrics(
            visual_probability, labels, visual_prediction
        )
        sequence_metrics = probability_metrics(
            marginal_probability, labels, sequence_prediction
        )
        def probability_slices(indices: list[int]) -> dict:
            if not indices:
                return {"num_samples": 0, "visual": None, "sequence": None}
            selected = torch.tensor(indices, dtype=torch.long)
            return {
                "num_samples": len(indices),
                "visual": probability_metrics(
                    visual_probability[selected],
                    labels[selected],
                    visual_prediction[selected],
                ),
                "sequence": probability_metrics(
                    marginal_probability[selected],
                    labels[selected],
                    sequence_prediction[selected],
                ),
            }

        duration_slices = {
            "duration_le_10s": probability_slices([
                index for index, record in enumerate(validation)
                if record.clip_duration_sec <= 10.0
            ]),
            "duration_gt_10s": probability_slices([
                index for index, record in enumerate(validation)
                if record.clip_duration_sec > 10.0
            ]),
        }
        sources = sorted(set(record.source_collection for record in validation))
        source_slices = {
            source: probability_slices([
                index for index, record in enumerate(validation)
                if record.source_collection == source
            ])
            for source in sources
        }
        ids, visual_cm = confusion_by_record(record_ids, labels, visual_prediction)
        sequence_ids, sequence_cm = confusion_by_record(
            record_ids, labels, sequence_prediction
        )
        if ids != sequence_ids:
            raise RuntimeError("record identities differ between paired decisions")
        visual_groups.append(visual_cm)
        sequence_groups.append(sequence_cm)
        seed_results[str(seed)] = {
            "visual": visual_metrics,
            "sequence": sequence_metrics,
            "slices": {
                **duration_slices,
                "source_collection": source_slices,
            },
            "delta": {
                "decision_accuracy": (
                    sequence_metrics["decision_accuracy"]
                    - visual_metrics["decision_accuracy"]
                ),
                "top_label_ece": (
                    sequence_metrics["top_label_ece"]
                    - visual_metrics["top_label_ece"]
                ),
                "nll": sequence_metrics["nll"] - visual_metrics["nll"],
                "multiclass_brier": (
                    sequence_metrics["multiclass_brier"]
                    - visual_metrics["multiclass_brier"]
                ),
            },
        }
    means = {}
    for model in ("visual", "sequence"):
        for field in (
            "decision_accuracy", "top_label_ece", "nll",
            "multiclass_brier", "mean_selected_probability",
            "mean_predictive_entropy", "area_under_risk_coverage_curve",
        ):
            means[f"{model}_{field}"] = _mean_std([
                seed_results[str(seed)][model][field] for seed in sorted(paths)
            ])
    output = {
        "schema_version": "fsn-probability-and-cluster-audit-1.0",
        "protocol": "fixed train-only transition prior; 823 validation clips; no independent test",
        "manifest_sha256": {"train": train_hash, "val": validation_hash},
        "calibration_bins": 10,
        "seeds": sorted(paths),
        "per_seed": seed_results,
        "three_seed_summary": means,
        "record_cluster_bootstrap": paired_record_bootstrap(
            np.stack(visual_groups),
            np.stack(sequence_groups),
            replicates=args.bootstrap_replicates,
        ),
        "probability_interpretation": (
            "Visual probabilities are softmax logits. Sequence probabilities "
            "are forward-backward marginals; decision labels remain Viterbi. "
            "ECE uses the marginal probability of the selected decision label."
        ),
        "test_metrics": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=named_path, required=True)
    parser.add_argument("--sequence-result", action="append", type=named_path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.validation_manifest = args.validation_manifest.resolve()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
