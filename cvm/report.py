"""Collect complete, predeclared training seeds without publishing private rows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Sequence


def _load(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def _stat(values: list[float]) -> dict:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite summary metric")
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n_seeds": len(values)}


def collect_runs(run_directories: Sequence[Path], expected_seeds: Sequence[int] = (42, 2026, 2027)) -> dict:
    expected = tuple(int(seed) for seed in expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected seeds must be distinct and nonempty")
    grouped: dict[tuple, list[dict]] = {}
    protocol_shas = set()
    for directory in run_directories:
        config = _load(directory / "config.json")
        result = _load(directory / "result.json")
        protocol_sha = config["protocol_sha256"]
        if result.get("protocol_sha256") != protocol_sha:
            raise ValueError("run result and configuration protocol differ")
        protocol_shas.add(protocol_sha)
        taxonomy = config["taxonomy_definition"]
        key = (config["backbone"], config["mode"], tuple(tuple(g) for g in taxonomy["groups"]))
        record = {"seed": int(config["seed"]), "status": result["status"],
                  "training_completed": bool(result.get("training_completed")),
                  "best_epoch": result.get("best_epoch"), "seconds": result.get("seconds"),
                  "parameters_trainable_finetune": result.get("parameters_trainable_finetune"),
                  "peak_cuda_allocated_bytes": result.get("peak_cuda_allocated_bytes"),
                  "git_commit": config.get("git_commit"), "code_hash": config.get("code_hash"),
                  "pretrained_sha256": config.get("checkpoint_sha256"),
                  "hyperparameters": {k: config.get(k) for k in ("batch_size", "accum_steps", "epochs", "warmup_epochs", "patience",
                  "optimizer", "backbone_lr", "head_lr", "warmup_lr", "weight_decay", "class_weights", "clip_grad", "augmentation",
                  "main_decoder", "aux_weight", "group_weight", "conditional_weight", "preprocessing")}}
        if record["seed"] not in expected:
            raise ValueError("undeclared training seed; declare it before summarizing")
        if record["training_completed"]:
            if result["status"] not in ("completed", "early_stopped"):
                raise ValueError("completion flag conflicts with run status")
            report = _load(directory / "val_report.json")
            if report["metadata"]["protocol_sha256"] != protocol_sha:
                raise ValueError("prediction report protocol mismatch")
            main = report["methods"]["main"]
            five = report["ambiguous_five"]["methods"]["main"]
            record.update({"val_macro_f1": main["macro_f1"], "val_accuracy": main["accuracy"],
                           "val_five_macro_f1": five["macro_f1"], "per_class": main["per_class"],
                           "confusion_matrix": main["confusion_matrix"],
                           "evaluation_metadata": report["metadata"], "num_recording_groups": report["num_recording_groups"]})
        grouped.setdefault(key, []).append(record)
    if not grouped or len(protocol_shas) != 1:
        raise ValueError("all collected runs must use one frozen protocol")
    output = []
    for (backbone, mode, groups), records in sorted(grouped.items()):
        if len({r["seed"] for r in records}) != len(records):
            raise ValueError("duplicate seed for one method; cannot select a favorable rerun")
        if len({json.dumps(r["hyperparameters"], sort_keys=True) for r in records}) != 1:
            raise ValueError("different hyperparameters cannot be pooled as seed replication")
        if len({r["code_hash"] for r in records}) != 1:
            raise ValueError("different implementations cannot be pooled as seed replication")
        completed = [r for r in records if r["training_completed"]]
        metrics = {key: _stat([r[key] for r in completed]) for key in
                   ("val_macro_f1", "val_accuracy", "val_five_macro_f1")} if completed else None
        output.append({"backbone": backbone, "mode": mode, "taxonomy_groups": [list(g) for g in groups],
                       "expected_seeds": list(expected), "missing_seeds": sorted(set(expected) - {r["seed"] for r in records}),
                       "all_declared_seeds_completed": len(completed) == len(expected),
                       "seed_summary": metrics, "runs": sorted(records, key=lambda r: r["seed"])})
    return {"schema_version": "fsn-cvm-seed-report-1", "protocol_sha256": next(iter(protocol_shas)),
            "split": "validation", "methods": output,
            "interpretation": "Validation selected the checkpoint. These are development estimates, not an independent test or a significance claim. Report every declared seed.",
            "privacy": {"aggregate_only": True, "raw_paths": False, "clip_ids": False, "recording_ids": False}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-seeds", default="42,2026,2027")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_runs(args.runs, [int(seed) for seed in args.expected_seeds.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
