#!/usr/bin/env python3
"""Export sanitized FSN experiment summaries that are safe to commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIVATE_MARKERS = ("/root/", "/Users/", "autodl-tmp", "Backup Plus")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def assert_public(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_public(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_public(item, f"{location}[{index}]")
    elif isinstance(value, str) and any(marker in value for marker in PRIVATE_MARKERS):
        raise ValueError(f"private path marker at {location}: {value}")


def best_history_row(result: dict[str, Any]) -> dict[str, Any]:
    best_name = result["best_epoch"]
    phase, epoch = best_name.rsplit("_", 1)
    matches = [
        row
        for row in result["history"]
        if row["phase"] == phase and row["epoch"] == int(epoch)
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve best epoch {best_name}")
    return matches[0]


def clean_load_report(report: dict[str, Any]) -> dict[str, Any]:
    if "official_checkpoint" in report:
        return {
            "official_checkpoint": clean_load_report(report["official_checkpoint"]),
            "shared_tensors_loaded": report["shared_tensors_loaded"],
            "new_module_tensors": report["new_module_tensors"],
        }
    return {
        key: value
        for key, value in report.items()
        if key != "checkpoint"
    }


def clean_visual_result(path: Path, display_name: str) -> dict[str, Any]:
    result = read_json(path)
    if result.get("test_metrics") is not None:
        raise ValueError(f"test metrics must be null: {path}")
    best = best_history_row(result)
    return {
        "display_name": display_name,
        "variant": result["variant"],
        "seed": result["seed"],
        "trainable_parameters": result["trainable_parameters"],
        "split_audit": result["split_audit"],
        "class_weight_mode": result["class_weight_mode"],
        "class_weights": result["class_weights"],
        "load_report": clean_load_report(result["load_report"]),
        "optimizer_groups": result["optimizer_groups"],
        "best_epoch": result["best_epoch"],
        "best_validation_macro_f1": result["best_val_macro_f1"],
        "best_validation_metrics": best["val_metrics"],
        "epochs_completed": {
            "head_warmup": sum(
                row["phase"] == "head_warmup" for row in result["history"]
            ),
            "finetune": sum(
                row["phase"] == "finetune" for row in result["history"]
            ),
        },
        "test_metrics": None,
    }


def clean_sequence_result(path: Path) -> dict[str, Any]:
    result = read_json(path)
    if result.get("test_metrics") is not None:
        raise ValueError(f"test metrics must be null: {path}")
    cleaned = {
        key: value
        for key, value in result.items()
        if key not in {
            "official_checkpoint", "visual_checkpoint", "official_load_report"
        }
    }
    cleaned["official_load_report"] = clean_load_report(
        result["official_load_report"]
    )
    return cleaned


def clean_three_seed(path: Path) -> dict[str, Any]:
    result = read_json(path)
    if result.get("test_metrics") is not None:
        raise ValueError("three-seed test metrics must be null")
    for row in result["per_seed"]:
        row.pop("path", None)
    return result


def clean_phase_audit(path: Path) -> dict[str, Any]:
    result = read_json(path)
    return {
        key: result[key]
        for key in (
            "schema_version", "checkpoint_epoch",
            "checkpoint_validation_macro_f1", "phase_bins",
            "decision_thresholds", "summary", "decision",
        )
        if key in result
    }


def clean_spatial_audit(path: Path) -> dict[str, Any]:
    result = read_json(path)
    return {
        key: result[key]
        for key in ("schema_version", "checkpoint_epoch", "summary")
        if key in result
    }


def export(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    visual = [
        clean_visual_result(path, name)
        for name, path in args.visual_result
    ]
    sequence = [clean_sequence_result(path) for path in args.sequence_result]
    three_seed = clean_three_seed(args.three_seed_summary)
    audits = {
        "phase_coverage": clean_phase_audit(args.phase_audit),
        "spatial_patch": clean_spatial_audit(args.spatial_audit),
        "checkpoint_reproduction": read_json(args.reproduction_audit),
        "completion": read_json(args.completion_audit),
    }
    payloads = {
        "visual_model_comparison.json": {
            "schema_version": "fsn-public-visual-comparison-1.0",
            "protocol_warning": (
                "823 held-out clips are validation; no independent test exists"
            ),
            "models": visual,
        },
        "sequence_per_seed.json": {
            "schema_version": "fsn-public-sequence-results-1.0",
            "results": sequence,
        },
        "three_seed_summary.json": three_seed,
        "audit_summary.json": audits,
    }
    for filename, payload in payloads.items():
        assert_public(payload)
        write_json(args.output / filename, payload)
    manifest = {
        "schema_version": "fsn-public-results-manifest-1.0",
        "files": sorted(payloads),
        "excluded": [
            "videos and decoded frame caches",
            "raw annotations and manifests containing local paths",
            "clip-level prediction JSONL",
            "training logs and model checkpoints",
        ],
        "privacy_check": "no known absolute private path markers",
        "test_metrics_policy": "must remain null",
    }
    assert_public(manifest)
    write_json(args.output / "manifest.json", manifest)


def named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
        return name, Path(path).resolve()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual-result", action="append", type=named_path, required=True
    )
    parser.add_argument(
        "--sequence-result", action="append", type=Path, required=True
    )
    parser.add_argument("--three-seed-summary", type=Path, required=True)
    parser.add_argument("--phase-audit", type=Path, required=True)
    parser.add_argument("--spatial-audit", type=Path, required=True)
    parser.add_argument("--reproduction-audit", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.sequence_result = [path.resolve() for path in args.sequence_result]
    for field in (
        "three_seed_summary", "phase_audit", "spatial_audit",
        "reproduction_audit", "completion_audit", "output",
    ):
        setattr(args, field, getattr(args, field).resolve())
    return args


if __name__ == "__main__":
    export(parse_args())
