"""Execute only predeclared paired analyses from private prediction artifacts.

No model is trained, selected, or evaluated here. Frozen plan/aggregate guards
are applied first, then original recording groups are resampled in each pair.
Missing jobs remain missing; they are never scored as zero or intersected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .analysis import compare_predictions, read_predictions
from .report import collect_plans


def _filename(identifier: str, seed: int) -> str:
    # Do not serialize an arbitrary identifier/path into an output filename.
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
        raise ValueError("invalid declared contrast identifier")
    return f"{identifier}__seed{seed}.json"


def run_pairs(plan_paths: Sequence[Path], output: Path, *, replicates: int = 2000, seed: int = 42) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("paired output already exists; no overwrite or resume")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    summary = collect_plans(plan_paths)
    locations = {}
    for path in plan_paths:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
        for job in plan["jobs"]:
            key = (job["configuration_id"], int(job["seed"]))
            value = (Path(job["output"]), job.get("task_type", "train"))
            if key in locations and locations[key] != value:
                raise ValueError("duplicate declared seed has multiple output locations")
            locations[key] = value
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    # Write a status artifact before processing: an exception cannot look complete.
    status_path = output / "paired_status.json"
    result = {"schema_version": "fsn-cvm-declared-pairs-1", "status": "running",
              "protocol_sha256": summary["protocol_sha256"], "plan_sha256": summary["plan_sha256"],
              "split": "validation", "replicates": replicates, "bootstrap_seed": seed,
              "pairs": completed, "training_or_inference_performed": False,
              "interpretation": "Checkpoint-conditional recording-group intervals; distinct from training-seed SD. Reused validation is exploratory.",
              "privacy": {"aggregate_only": True, "private_paths": False, "clip_ids": False, "recording_ids": False}}
    def save_status():
        status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_status()
    try:
        for contrast in summary["comparison_rows"]:
            available = set(contrast["completed_paired_seeds"])
            for training_seed in contrast["declared_seeds"]:
                record = {"contrast_id": contrast["contrast_id"], "training_seed": training_seed,
                          "status": "missing_or_incomplete", "aggregate_file": None}
                if training_seed in available:
                    pairs = []
                    for key in ("baseline_configuration_id", "candidate_configuration_id"):
                        directory, task = locations[(contrast[key], training_seed)]
                        predictions = directory / ("predictions.jsonl" if task == "evaluate" else "val_predictions.jsonl")
                        pairs.append(read_predictions(predictions, directory / "prediction_metadata.json"))
                    probe = contrast["comparison_kind"] == "same_checkpoint_probe"
                    frames = "frames" in contrast.get("varying_factors", [])
                    kwargs = {"baseline_metadata": pairs[0][1], "candidate_metadata": pairs[1][1],
                              "bootstrap_replicates": replicates, "seed": seed,
                              "same_checkpoint_probe": probe, "frame_sensitivity": frames}
                    report = compare_predictions(pairs[0][0], pairs[1][0], **kwargs)
                    # Add the fair true-group interval only when both heads were trained.
                    if not probe and not frames and "flat" in pairs[0][1].get("trained_heads", []) and {"group", "conditional"}.issubset(pairs[1][1].get("trained_heads", [])):
                        oracle = compare_predictions(pairs[0][0], pairs[1][0], baseline_method="oracle_flat", candidate_method="oracle_hierarchy", **kwargs)
                        report["fair_oracle_paired_uncertainty"] = oracle["paired_uncertainty"]
                    filename = _filename(contrast["contrast_id"], training_seed)
                    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                    (output / filename).write_text(serialized, encoding="utf-8")
                    record.update(status="completed", aggregate_file=filename,
                                  aggregate_sha256=hashlib.sha256(serialized.encode()).hexdigest())
                completed.append(record)
                save_status()
        result["status"] = "completed_with_missing_pairs" if any(item["status"] != "completed" for item in completed) else "completed"
        save_status()
    except BaseException:
        result["status"] = "failed_or_interrupted"
        save_status()
        raise
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG, not an additional training seed")
    args = parser.parse_args(argv)
    try:
        result = run_pairs(args.plans, args.output, replicates=args.replicates, seed=args.seed)
    except (ValueError, OSError, KeyError):
        raise SystemExit("paired analysis failed: check the frozen plan and private artifact consistency; no fitting or inference was performed") from None
    print(json.dumps({"status": result["status"], "pairs": len(result["pairs"]), "training_or_inference_performed": False}))


if __name__ == "__main__":
    main()
