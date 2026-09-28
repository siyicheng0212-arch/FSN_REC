#!/usr/bin/env python3
"""Aggregate matched baseline/sequence metrics across formal FSN seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


# Two-sided Student-t 97.5% quantiles for small seed counts.  Three seeds use
# df=2 and must be reported as descriptive evidence, not strong significance.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _mean_std_interval(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one value is required")
    mean = statistics.mean(values)
    if len(values) == 1:
        return {"values": values, "mean": mean, "sample_std": None,
                "t_95_interval": None}
    std = statistics.stdev(values)
    df = len(values) - 1
    critical = T_975.get(df, 1.96)
    margin = critical * std / math.sqrt(len(values))
    return {
        "values": values,
        "mean": mean,
        "sample_std": std,
        "t_95_interval": [mean - margin, mean + margin],
        "degrees_of_freedom": df,
        "caution": "only three seeds; interval is descriptive and unstable"
        if len(values) == 3 else None,
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two seed results are required")
    rows = []
    manifest_hashes = None
    transition_config = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("test_metrics") is not None:
            raise ValueError(f"independent test metrics are not allowed: {path}")
        current_hashes = data.get("manifest_sha256")
        current_config = (data.get("transition_weight"), data["transition_prior"]["smoothing"])
        if manifest_hashes is None:
            manifest_hashes = current_hashes
            transition_config = current_config
        if current_hashes != manifest_hashes:
            raise ValueError("seed results use different manifests")
        if current_config != transition_config:
            raise ValueError("seed results use different transition settings")
        baseline = data["baseline_metrics"]["all"]
        sequence = data["sequence_metrics"]["all"]
        baseline_cm = baseline["confusion_matrix"]
        sequence_cm = sequence["confusion_matrix"]
        duration = {}
        for slice_name in ("duration_le_10s", "duration_gt_10s"):
            base_slice = data["baseline_metrics"]["slices"][slice_name]
            sequence_slice = data["sequence_metrics"]["slices"][slice_name]
            duration[slice_name] = {
                "support": base_slice["num_samples"],
                "baseline_accuracy": base_slice["accuracy"],
                "sequence_accuracy": sequence_slice["accuracy"],
                "baseline_macro_f1": base_slice["macro_f1"],
                "sequence_macro_f1": sequence_slice["macro_f1"],
                "baseline_present_class_macro_f1": base_slice[
                    "present_class_macro_f1"
                ],
                "sequence_present_class_macro_f1": sequence_slice[
                    "present_class_macro_f1"
                ],
            }
        rows.append({
            "path": str(path),
            "seed": int(data["seed"]),
            "baseline_macro_f1": baseline["macro_f1"],
            "sequence_macro_f1": sequence["macro_f1"],
            "delta_macro_f1": sequence["macro_f1"] - baseline["macro_f1"],
            "baseline_accuracy": baseline["accuracy"],
            "sequence_accuracy": sequence["accuracy"],
            "delta_accuracy": sequence["accuracy"] - baseline["accuracy"],
            "baseline_per_class_f1": [item["f1"] for item in baseline["per_class"]],
            "sequence_per_class_f1": [item["f1"] for item in sequence["per_class"]],
            "baseline_sweep_to_reperfusion": baseline_cm[3][4],
            "sequence_sweep_to_reperfusion": sequence_cm[3][4],
            "baseline_reperfusion_to_sweep": baseline_cm[4][3],
            "sequence_reperfusion_to_sweep": sequence_cm[4][3],
            "duration_slices": duration,
            "changed_predictions": data["changed_predictions"],
        })
    rows.sort(key=lambda row: row["seed"])
    delta_macro = [row["delta_macro_f1"] for row in rows]
    delta_accuracy = [row["delta_accuracy"] for row in rows]
    summary = {
        "schema_version": "fsn-procedure-aware-three-seed-summary-1.0",
        "num_seeds": len(rows),
        "seeds": [row["seed"] for row in rows],
        "manifest_sha256": manifest_hashes,
        "transition_weight": transition_config[0],
        "smoothing": transition_config[1],
        "per_seed": rows,
        "baseline_macro_f1": _mean_std_interval(
            [row["baseline_macro_f1"] for row in rows]
        ),
        "sequence_macro_f1": _mean_std_interval(
            [row["sequence_macro_f1"] for row in rows]
        ),
        "delta_macro_f1": _mean_std_interval(delta_macro),
        "baseline_accuracy": _mean_std_interval(
            [row["baseline_accuracy"] for row in rows]
        ),
        "sequence_accuracy": _mean_std_interval(
            [row["sequence_accuracy"] for row in rows]
        ),
        "delta_accuracy": _mean_std_interval(delta_accuracy),
        "acceptance": {
            "mean_delta_macro_f1_at_least_0.01": statistics.mean(delta_macro) >= 0.01,
            "no_seed_macro_f1_regression": min(delta_macro) >= 0.0,
            "mean_accuracy_non_decreasing": statistics.mean(delta_accuracy) >= 0.0,
        },
        "test_metrics": None,
    }
    summary["acceptance"]["passed"] = all(summary["acceptance"].values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([path.resolve() for path in args.results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
