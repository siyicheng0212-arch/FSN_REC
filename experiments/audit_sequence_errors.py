#!/usr/bin/env python3
"""Build a PRIVATE clip-review queue before changing the FSN sequence model.

Input predictions and the review queue contain clip identities and media paths.
Keep all outputs on the data server; never publish them in GitHub results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.audit_calibration import file_sha256, named_path, read_prediction_rows
from experiments.pilot_data import ClipRecord, load_pilot_manifest
from experiments.sequence_decoder import (
    _ordered_groups,
    decode_sequences,
    fit_transition_prior,
)


REVIEW_CATEGORIES = (
    "new_reperfusion_to_sweep",
    "fixed_sweep_to_reperfusion",
    "new_sweep_to_reperfusion",
    "fixed_reperfusion_to_sweep",
    "persistent_reperfusion_to_sweep",
    "persistent_sweep_to_reperfusion",
)


def directional_case(target: int, visual: int, sequence: int) -> str | None:
    if target == 4 and visual == 4 and sequence == 3:
        return "new_reperfusion_to_sweep"
    if target == 3 and visual == 4 and sequence == 3:
        return "fixed_sweep_to_reperfusion"
    if target == 3 and visual == 3 and sequence == 4:
        return "new_sweep_to_reperfusion"
    if target == 4 and visual == 3 and sequence == 4:
        return "fixed_reperfusion_to_sweep"
    if target == 4 and visual == 3 and sequence == 3:
        return "persistent_reperfusion_to_sweep"
    if target == 3 and visual == 4 and sequence == 4:
        return "persistent_sweep_to_reperfusion"
    return None


def temporal_context(records: list[ClipRecord]) -> dict[str, dict]:
    context = {}
    for indices in _ordered_groups(records):
        for position, index in enumerate(indices):
            record = records[index]
            previous = records[indices[position - 1]] if position else None
            following = (
                records[indices[position + 1]]
                if position + 1 < len(indices)
                else None
            )
            context[record.clip_id] = {
                "position_in_record": position,
                "record_length": len(indices),
                "previous_label": previous.label_id if previous else None,
                "next_label": following.label_id if following else None,
                "target_action_onset": (
                    previous is not None and previous.label_id != record.label_id
                ),
                "target_action_offset": (
                    following is not None and following.label_id != record.label_id
                ),
                "previous_gap_sec": (
                    record.clip_start_sec - previous.clip_end_sec
                    if previous else None
                ),
            }
    return context


def _safe_output_dir(output_dir: Path) -> None:
    if {"public_results", "reports", ".git"} & set(output_dir.parts):
        raise ValueError("private clip-review output must not be in public_results, reports, or .git")


def _select_review_queue(
    per_clip: dict[str, dict], *, max_per_category: int, max_per_record: int
) -> list[dict]:
    selected = []
    selected_ids = set()
    for category in REVIEW_CATEGORIES:
        candidates = [
            row for row in per_clip.values()
            if row["case_votes"].get(category, 0)
        ]
        candidates.sort(key=lambda row: (
            -row["case_votes"][category],
            -row["mean_visual_target_probability"],
            row["clip_id"],
        ))
        per_record = Counter()
        for row in candidates:
            if row["clip_id"] in selected_ids:
                continue
            if per_record[row["record_id"]] >= max_per_record:
                continue
            selected.append({**row, "review_category": category})
            selected_ids.add(row["clip_id"])
            per_record[row["record_id"]] += 1
            if per_record.total() >= max_per_category:
                break
    return selected


def run(args: argparse.Namespace) -> dict:
    if args.max_per_category < 1 or args.max_per_record < 1:
        raise ValueError("review queue limits must be positive")
    _safe_output_dir(args.output_dir)
    train = load_pilot_manifest(args.train_manifest)
    validation = load_pilot_manifest(args.validation_manifest)
    if {row.group_id for row in train} & {row.group_id for row in validation}:
        raise ValueError("train and validation groups overlap")
    prediction_paths = dict(args.prediction)
    result_paths = dict(args.sequence_result)
    if (
        len(prediction_paths) < 2
        or len(prediction_paths) != len(args.prediction)
        or len(result_paths) != len(args.sequence_result)
        or set(prediction_paths) != set(result_paths)
    ):
        raise ValueError("at least two unique, matched seed inputs are required")
    manifest_hashes = {
        "train": file_sha256(args.train_manifest),
        "val": file_sha256(args.validation_manifest),
    }
    prior = fit_transition_prior(train, smoothing=1.0)
    labels = torch.tensor([row.label_id for row in validation], dtype=torch.long)
    expected_ids = {row.clip_id for row in validation}
    context = temporal_context(validation)
    per_clip: dict[str, dict] = {}
    per_seed = {}
    for seed in sorted(prediction_paths):
        saved = json.loads(result_paths[seed].read_text(encoding="utf-8"))
        if saved.get("test_metrics") is not None:
            raise ValueError("independent test metrics must be null")
        if saved.get("seed") != seed or saved.get("manifest_sha256") != manifest_hashes:
            raise ValueError("seed or manifest hash mismatch")
        if (
            saved.get("transition_weight") != 1.0
            or saved.get("transition_prior", {}).get("smoothing") != 1.0
        ):
            raise ValueError("stored v3 transition settings mismatch")
        rows = read_prediction_rows(prediction_paths[seed])
        if set(rows) != expected_ids:
            raise ValueError("prediction clip set differs from manifest")
        ordered = [rows[row.clip_id] for row in validation]
        if any(int(row["target"]) != int(target) for row, target in zip(ordered, labels)):
            raise ValueError("prediction targets differ from manifest")
        logits = torch.tensor([row["visual_logits"] for row in ordered], dtype=torch.float64)
        if logits.shape != (len(validation), 7) or not torch.isfinite(logits).all():
            raise ValueError("visual logits must be finite [N, 7]")
        visual = logits.argmax(dim=1)
        sequence = decode_sequences(logits, validation, prior, weight=1.0)
        if any(
            int(row["visual_prediction"]) != int(prediction)
            or int(row["sequence_prediction"]) != int(decoded)
            for row, prediction, decoded in zip(ordered, visual, sequence)
        ):
            raise ValueError("saved predictions do not reproduce")
        probabilities = torch.softmax(logits, dim=1)
        counts = Counter()
        for index, record in enumerate(validation):
            target = int(labels[index])
            visual_label = int(visual[index])
            sequence_label = int(sequence[index])
            case = directional_case(target, visual_label, sequence_label)
            counts["visual_sweep_to_reperfusion"] += int(
                target == 3 and visual_label == 4
            )
            counts["sequence_sweep_to_reperfusion"] += int(
                target == 3 and sequence_label == 4
            )
            counts["visual_reperfusion_to_sweep"] += int(
                target == 4 and visual_label == 3
            )
            counts["sequence_reperfusion_to_sweep"] += int(
                target == 4 and sequence_label == 3
            )
            counts["visual_correct_sequence_wrong"] += int(
                visual_label == target and sequence_label != target
            )
            counts["visual_wrong_sequence_correct"] += int(
                visual_label != target and sequence_label == target
            )
            if case is None:
                continue
            counts[case] += 1
            counts[f"{case}_at_action_onset"] += int(
                context[record.clip_id]["target_action_onset"]
            )
            gap = context[record.clip_id]["previous_gap_sec"]
            counts[f"{case}_gap_gt_1s"] += int(gap is not None and gap > 1.0)
            counts[f"{case}_gap_gt_10s"] += int(gap is not None and gap > 10.0)
            item = per_clip.setdefault(record.clip_id, {
                "clip_id": record.clip_id,
                "record_id": record.record_id or record.clip_id,
                "source_collection": record.source_collection,
                "video_path": record.video_path,
                "clip_start_sec": record.clip_start_sec,
                "clip_end_sec": record.clip_end_sec,
                "clip_duration_sec": record.clip_duration_sec,
                "target": target,
                **context[record.clip_id],
                "case_votes": {},
                "seed_details": [],
            })
            item["case_votes"][case] = item["case_votes"].get(case, 0) + 1
            item["seed_details"].append({
                "seed": seed,
                "case": case,
                "visual_prediction": visual_label,
                "sequence_prediction": sequence_label,
                "visual_target_probability": float(probabilities[index, target]),
                "visual_sweep_probability": float(probabilities[index, 3]),
                "visual_reperfusion_probability": float(probabilities[index, 4]),
                "visual_logit_margin_sweep_minus_reperfusion": float(
                    logits[index, 3] - logits[index, 4]
                ),
            })
        per_seed[str(seed)] = dict(sorted(counts.items()))
    for item in per_clip.values():
        item["mean_visual_target_probability"] = sum(
            detail["visual_target_probability"] for detail in item["seed_details"]
        ) / len(item["seed_details"])
    queue = _select_review_queue(
        per_clip,
        max_per_category=args.max_per_category,
        max_per_record=args.max_per_record,
    )
    category_counts = Counter(row["review_category"] for row in queue)
    summary = {
        "schema_version": "fsn-private-sequence-error-audit-1.0",
        "privacy": "clip review queue and paths stay on the data server; do not publish",
        "manifest_sha256": manifest_hashes,
        "seeds": sorted(prediction_paths),
        "per_seed": per_seed,
        "unique_reviewable_clips": len(per_clip),
        "review_queue_count": len(queue),
        "review_queue_by_category": dict(sorted(category_counts.items())),
        "test_metrics": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "private_review_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=named_path, required=True)
    parser.add_argument("--sequence-result", action="append", type=named_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-category", type=int, default=12)
    parser.add_argument("--max-per-record", type=int, default=2)
    args = parser.parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.validation_manifest = args.validation_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
