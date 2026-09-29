#!/usr/bin/env python3
"""Evaluate fixed dual-rate fusion and v3 decoding on matched private logits.

This is a diagnostic with no fitted fusion weights. The input rows contain
clip identities and must stay private; only aggregate metrics are written.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from experiments.audit_focus_selection import frozen_hashes, sha256_file
from experiments.metrics import compute_classification_metrics
from experiments.pilot_data import load_pilot_manifest
from experiments.sequence_decoder import (
    decode_sequences,
    fit_transition_prior,
    predictions_to_logits,
)


SEEDS = (42, 123, 2026)
ARMS = ("uniform", "three_windows")


def _read_rows(path: Path, records: list) -> torch.Tensor:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["clip_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != {record.clip_id for record in records}:
        raise ValueError("private focus rows do not match the frozen validation clips")
    ordered = [by_id[record.clip_id] for record in records]
    if any(row["target"] != record.label_id for row, record in zip(ordered, records)):
        raise ValueError("private focus targets differ from frozen manifest")
    logits = torch.tensor([row["visual_logits"] for row in ordered], dtype=torch.float64)
    if logits.shape != (len(records), 7) or not torch.isfinite(logits).all():
        raise ValueError("private visual logits must be finite [N,7]")
    if any(row["prediction"] != int(logits[index].argmax()) for index, row in enumerate(ordered)):
        raise ValueError("stored visual predictions do not match logits")
    return logits


def _compact(metrics: dict) -> dict:
    block = metrics["all"]
    confusion = block["confusion_matrix"]
    return {
        "macro_f1": block["macro_f1"],
        "accuracy": block["accuracy"],
        "weighted_f1": block["weighted_f1"],
        "per_class_f1": [row["f1"] for row in block["per_class"]],
        "sweep_to_reperfusion": confusion[3][4],
        "reperfusion_to_sweep": confusion[4][3],
        "duration_le_10s_present_class_macro_f1": metrics["slices"]["duration_le_10s"]["present_class_macro_f1"],
        "duration_gt_10s_present_class_macro_f1": metrics["slices"]["duration_gt_10s"]["present_class_macro_f1"],
        "source_accuracy": {
            source: block["accuracy"]
            for source, block in metrics["slices"]["source_collection"].items()
        },
    }


def _metrics(logits_or_labels: torch.Tensor, targets: torch.Tensor, records: list,
             *, labels: bool = False) -> dict:
    metadata = [
        {"source_collection": record.source_collection,
         "clip_duration_sec": record.clip_duration_sec}
        for record in records
    ]
    logits = predictions_to_logits(logits_or_labels) if labels else logits_or_labels
    return _compact(compute_classification_metrics(logits, targets, metadata))


def run(args: argparse.Namespace) -> dict:
    expected = frozen_hashes(args.protocol)
    for split, sha in expected.items():
        if sha256_file(args.manifest_dir / f"{split}.jsonl") != sha:
            raise ValueError(f"{split} manifest hash mismatch")
    train = load_pilot_manifest(args.manifest_dir / "train.jsonl")
    validation = load_pilot_manifest(args.manifest_dir / "val.jsonl")
    if {row.group_id for row in train} & {row.group_id for row in validation}:
        raise ValueError("train and validation groups overlap")
    if len(validation) != 823:
        raise ValueError("expected 823 validation clips")
    targets = torch.tensor([row.label_id for row in validation], dtype=torch.long)
    prior = fit_transition_prior(train, smoothing=1.0)
    per_seed = {}
    for seed in SEEDS:
        logits = {
            arm: _read_rows(
                args.focus_dir / f"{arm}_seed{seed}" / "private_focus_rows.jsonl",
                validation,
            )
            for arm in ARMS
        }
        probabilities = [torch.softmax(logits[arm], dim=1) for arm in ARMS]
        fixed_fusion = (probabilities[0] + probabilities[1]) * 0.5
        fused_logits = fixed_fusion.clamp_min(1e-12).log()
        candidates = {**logits, "equal_probability_fusion": fused_logits}
        result = {}
        for name, scores in candidates.items():
            visual = _metrics(scores, targets, validation)
            decoded = decode_sequences(scores, validation, prior, weight=1.0)
            sequence = _metrics(decoded, targets, validation, labels=True)
            result[name] = {"visual": visual, "v3_sequence": sequence}
        per_seed[str(seed)] = result
    aggregate = {}
    for name in (*ARMS, "equal_probability_fusion"):
        aggregate[name] = {}
        for stage in ("visual", "v3_sequence"):
            aggregate[name][stage] = {
                key: {
                    "mean": statistics.mean(
                        per_seed[str(seed)][name][stage][key] for seed in SEEDS
                    ),
                    "per_seed": [
                        per_seed[str(seed)][name][stage][key] for seed in SEEDS
                    ],
                }
                for key in ("macro_f1", "accuracy", "sweep_to_reperfusion", "reperfusion_to_sweep")
            }
    paired_delta = {
        stage: {
            "macro_f1_dense_minus_uniform": [
                per_seed[str(seed)]["three_windows"][stage]["macro_f1"]
                - per_seed[str(seed)]["uniform"][stage]["macro_f1"]
                for seed in SEEDS
            ],
            "macro_f1_fusion_minus_uniform": [
                per_seed[str(seed)]["equal_probability_fusion"][stage]["macro_f1"]
                - per_seed[str(seed)]["uniform"][stage]["macro_f1"]
                for seed in SEEDS
            ],
        }
        for stage in ("visual", "v3_sequence")
    }
    summary = {
        "schema_version": "fsn-fixed-dual-rate-fusion-diagnostic-1.0",
        "purpose": "diagnostic only; no trainable fusion parameters or validation-tuned weight",
        "fusion": "arithmetic mean of uniform and three-window softmax probabilities",
        "sequence_prior": "train labels only; Laplace 1.0; Viterbi weight 1.0",
        "manifest_sha256": expected,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_delta": paired_delta,
        "test_metrics": None,
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if any(private in serialized for private in ("/root/", "/Users/", "autodl-tmp", "clip-")):
        raise ValueError("public diagnostic contains private source details")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/motion_sampling_protocol.json"))
    parser.add_argument("--focus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        name: {
            stage: result["aggregate"][name][stage]["macro_f1"]["mean"]
            for stage in ("visual", "v3_sequence")
        }
        for name in (*ARMS, "equal_probability_fusion")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
