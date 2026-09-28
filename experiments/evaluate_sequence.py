#!/usr/bin/env python3
"""Evaluate the frozen original AdaFocus checkpoint with FSN sequence decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.full_data import FullClipDataset
from experiments.metrics import compute_classification_metrics
from experiments.sequence_decoder import (
    decode_sequences,
    fit_transition_prior,
    predictions_to_logits,
)
from experiments.train_adafocus import make_model, metadata_rows, seed_all


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_support_macro(left: dict, right: dict) -> dict:
    ids = [
        index
        for index, (a, b) in enumerate(zip(left["per_class"], right["per_class"]))
        if a["support"] > 0 and b["support"] > 0
    ]
    if not ids:
        return {"class_ids": [], "left": 0.0, "right": 0.0}
    return {
        "class_ids": ids,
        "left": sum(left["per_class"][index]["f1"] for index in ids) / len(ids),
        "right": sum(right["per_class"][index]["f1"] for index in ids) / len(ids),
    }


@torch.no_grad()
def infer(model, dataset, batch_size: int, workers: int, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    logits, targets, metadata = [], [], []
    model.eval()
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(video)["logits"]
        logits.append(output.float().cpu())
        targets.append(batch["label"].cpu())
        metadata.extend(metadata_rows(batch))
    return torch.cat(logits), torch.cat(targets), metadata


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("sequence evaluation requires CUDA")
    if args.transition_weight < 0 or args.smoothing <= 0:
        raise ValueError("invalid transition hyperparameters")
    device = torch.device("cuda:0")
    seed_all(args.seed)
    train_manifest = args.manifest_dir / "train.jsonl"
    val_manifest = args.manifest_dir / "val.jsonl"
    train_dataset = FullClipDataset(train_manifest, args.cache_dir, 36, 224)
    val_dataset = FullClipDataset(val_manifest, args.cache_dir, 36, 224)
    prior = fit_transition_prior(train_dataset.records, args.smoothing)

    model_args = argparse.Namespace(
        variant="original",
        checkpoint=args.official_checkpoint,
        allow_random_init=False,
        seed=args.seed,
    )
    model, official_report = make_model(model_args, device)
    checkpoint = torch.load(
        args.visual_checkpoint, map_location=device, weights_only=False
    )
    state = {key: value for key, value in checkpoint["model"].items() if key != "class_weights"}
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"trained checkpoint mismatch: missing={loaded.missing_keys}, "
            f"unexpected={loaded.unexpected_keys}"
        )
    visual_logits, targets, metadata = infer(
        model, val_dataset, args.batch_size, args.workers, device
    )
    baseline = compute_classification_metrics(visual_logits, targets, metadata)
    decoded = decode_sequences(
        visual_logits, val_dataset.records, prior, args.transition_weight
    )
    sequence = compute_classification_metrics(
        predictions_to_logits(decoded), targets, metadata
    )
    short_long_common = common_support_macro(
        sequence["slices"]["duration_le_10s"],
        sequence["slices"]["duration_gt_10s"],
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    base_prediction = visual_logits.argmax(dim=1)
    for index, record in enumerate(val_dataset.records):
        rows.append(
            {
                "clip_id": record.clip_id,
                "record_id": record.record_id or record.clip_id,
                "clip_start_sec": record.clip_start_sec,
                "target": int(targets[index]),
                "visual_prediction": int(base_prediction[index]),
                "sequence_prediction": int(decoded[index]),
                "visual_logits": [float(value) for value in visual_logits[index]],
            }
        )
    with (output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "schema_version": "fsn-procedure-aware-decoder-1.0",
        "protocol": "train-only transition prior; original-test-as-validation; no independent test",
        "seed": args.seed,
        "transition_weight": args.transition_weight,
        "transition_prior": prior.to_dict(),
        "manifest_sha256": {
            "train": sha256(train_manifest),
            "val": sha256(val_manifest),
        },
        "official_checkpoint": str(args.official_checkpoint),
        "official_checkpoint_sha256": sha256(args.official_checkpoint),
        "visual_checkpoint": str(args.visual_checkpoint),
        "visual_checkpoint_sha256": sha256(args.visual_checkpoint),
        "visual_checkpoint_epoch": checkpoint["epoch"],
        "visual_checkpoint_validation_macro_f1": checkpoint["val_macro_f1"],
        "official_load_report": official_report,
        "baseline_metrics": baseline,
        "sequence_metrics": sequence,
        "delta_macro_f1": sequence["all"]["macro_f1"] - baseline["all"]["macro_f1"],
        "delta_accuracy": sequence["all"]["accuracy"] - baseline["all"]["accuracy"],
        "changed_predictions": int((decoded != base_prediction).sum()),
        "short_long_common_support_macro_f1": short_long_common,
        "test_metrics": None,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for field in ("manifest_dir", "cache_dir", "official_checkpoint", "visual_checkpoint", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
