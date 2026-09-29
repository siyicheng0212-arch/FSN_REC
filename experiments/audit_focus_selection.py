#!/usr/bin/env python3
"""Audit where a trained AdaFocus model actually sampled each FSN clip.

The private rows file contains clip identities and predictions. Keep it on the
data server. The summary contains only aggregate statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.full_data import FullClipDataset, requested_timestamps
from experiments.metrics import compute_classification_metrics
from experiments.sequence_decoder import predictions_to_logits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes(protocol_path: Path) -> dict[str, str]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    hashes = {
        "train": protocol["train_manifest_sha256"],
        "val": protocol["validation_manifest_sha256"],
    }
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in hashes.values()):
        raise ValueError("invalid frozen manifest hash in protocol")
    return hashes


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"support": 0}
    fields = (
        "selected_span_fraction", "largest_selected_gap_fraction",
        "adjacent_selected_pairs", "selected_first_third",
        "selected_middle_third", "selected_last_third",
    )
    return {
        "support": len(rows),
        "accuracy": sum(row["prediction"] == row["target"] for row in rows) / len(rows),
        **{f"mean_{field}": statistics.mean(row[field] for row in rows) for field in fields},
    }


def summarize(
    rows: list[dict], sampling: str, seed: int, expected_hashes: dict[str, str]
) -> dict:
    slices = {
        "all": rows,
        "duration_le_3s": [row for row in rows if row["duration_sec"] <= 3.0],
        "duration_3_to_10s": [row for row in rows if 3.0 < row["duration_sec"] <= 10.0],
        "duration_gt_10s": [row for row in rows if row["duration_sec"] > 10.0],
        "sweeping": [row for row in rows if row["target"] == 3],
        "reperfusion": [row for row in rows if row["target"] == 4],
        "lishui": [row for row in rows if row["source"] == "lishui"],
        "lishui_long": [row for row in rows if row["source"] == "lishui" and row["duration_sec"] > 10.0],
        "reperfusion_wrong_as_sweep": [row for row in rows if row["target"] == 4 and row["prediction"] == 3],
        "sweeping_wrong_as_reperfusion": [row for row in rows if row["target"] == 3 and row["prediction"] == 4],
    }
    return {
        "schema_version": "fsn-focus-time-audit-1.0",
        "sampling": sampling,
        "seed": seed,
        "num_frames": 36,
        "num_focus_frames": 12,
        "manifest_sha256": expected_hashes,
        "slices": {name: _stats(subset) for name, subset in slices.items()},
        "private_rows_not_for_publication": True,
        "test_metrics": None,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference is required for this audit")
    if args.sampling not in ("uniform", "three_windows"):
        raise ValueError("invalid sampling mode")
    if {"public_results", "reports", ".git"} & set(args.output_dir.parts):
        raise ValueError("private focus rows must stay outside public result directories")
    expected_hashes = frozen_hashes(args.protocol)
    for split, expected in expected_hashes.items():
        current = sha256_file(args.manifest_dir / f"{split}.jsonl")
        if current != expected:
            raise ValueError(f"{split} manifest hash mismatch")
    checkpoint = torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)
    stored = checkpoint.get("split_audit", {}).get("manifest_sha256")
    if stored != expected_hashes:
        raise ValueError("visual checkpoint manifest hash mismatch")
    if checkpoint.get("args", {}).get("sampling") != args.sampling:
        raise ValueError("visual checkpoint sampling mode mismatch")
    if checkpoint.get("args", {}).get("seed") != args.seed:
        raise ValueError("visual checkpoint seed mismatch")
    dataset = FullClipDataset(
        args.manifest_dir / "val.jsonl", args.cache_dir, 36, 224, args.sampling
    )
    record_by_id = {record.clip_id: record for record in dataset.records}
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    device = torch.device("cuda:0")
    from experiments.train_adafocus import make_model

    model_args = argparse.Namespace(
        variant="original", checkpoint=args.official_checkpoint,
        allow_random_init=False, seed=args.seed,
    )
    model, _ = make_model(model_args, device)
    state = {key: value for key, value in checkpoint["model"].items() if key != "class_weights"}
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("trained checkpoint state mismatch")
    model.eval()
    rows = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(video)
        logits = output["logits"].float().cpu()
        focus = output["eval_outputs"][-1].view(len(batch["clip_id"]), 12).cpu()
        offsets = torch.arange(len(batch["clip_id"])).unsqueeze(1) * 36
        indices = focus - offsets
        if ((indices < 0) | (indices >= 36)).any():
            raise RuntimeError("focus indices outside candidate range")
        for index, clip_id in enumerate(batch["clip_id"]):
            record = record_by_id[clip_id]
            selected = indices[index].tolist()
            if selected != sorted(selected) or len(set(selected)) != 12:
                raise RuntimeError("focus indices are not unique and ordered")
            requested = requested_timestamps(record, 36, args.sampling)
            times = [requested[position] for position in selected]
            duration = record.clip_duration_sec
            gaps = [right - left for left, right in zip(times, times[1:])]
            prediction = int(logits[index].argmax())
            rows.append({
                "clip_id": clip_id,
                "record_id": record.record_id,
                "source": record.source_collection,
                "target": record.label_id,
                "prediction": prediction,
                "visual_logits": [float(value) for value in logits[index]],
                "duration_sec": duration,
                "selected_indices": selected,
                "requested_selected_timestamps_sec": times,
                "selected_span_fraction": (times[-1] - times[0]) / duration,
                "largest_selected_gap_fraction": max(gaps) / duration,
                "adjacent_selected_pairs": sum(right - left == 1 for left, right in zip(selected, selected[1:])),
                "selected_first_third": sum(position < 12 for position in selected),
                "selected_middle_third": sum(12 <= position < 24 for position in selected),
                "selected_last_third": sum(position >= 24 for position in selected),
            })
    if len(rows) != 823:
        raise RuntimeError(f"expected 823 validation clips, found {len(rows)}")
    summary = summarize(rows, args.sampling, args.seed, expected_hashes)
    metrics = compute_classification_metrics(
        predictions_to_logits(row["prediction"] for row in rows),
        torch.tensor([row["target"] for row in rows]),
        [
            {"source_collection": row["source"], "clip_duration_sec": row["duration_sec"]}
            for row in rows
        ],
    )
    macro_f1 = metrics["all"]["macro_f1"]
    checkpoint_f1 = float(checkpoint["val_macro_f1"])
    if abs(macro_f1 - checkpoint_f1) > 0.005:
        raise RuntimeError(
            f"checkpoint prediction mismatch: {macro_f1:.5f} vs {checkpoint_f1:.5f}"
        )
    summary["macro_f1_reproduced"] = macro_f1
    summary["checkpoint_macro_f1"] = checkpoint_f1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_focus_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "private_focus_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/motion_sampling_protocol.json"),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--sampling", choices=("uniform", "three_windows"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
