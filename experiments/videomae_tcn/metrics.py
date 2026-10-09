"""Seven-class, paired and identity-free stratified summaries."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from experiments.relation.evaluate_suite import classification_metrics, _directions, _paired


def block(labels, predictions, a_predictions):
    result = classification_metrics(labels, predictions)
    result["paired_vs_A"] = _paired(labels, predictions, a_predictions)
    result["sweep_reperfusion"] = _directions(labels, predictions)
    present = [row["f1"] for row in result["per_class"] if row["support"] > 0]
    result["present_class_macro_f1"] = float(np.mean(present)) if present else None
    result["full_seven_class_macro_f1"] = result["macro_f1"]
    result["all_seven_classes_present"] = len(present) == 7
    return result


def summarize(private_predictions, metadata_rows, val_segments):
    metadata = {row["clip_id"]: row for row in metadata_rows if row["split"] == "val"}
    by_clip = {row["clip_id"]: row for row in private_predictions}
    if len(by_clip) != len(private_predictions) or by_clip.keys() != metadata.keys():
        raise ValueError("each val clip must be evaluated exactly once")
    isolated = {s.clip_ids[0] for s in val_segments if len(s.clip_ids) == 1}
    labels, a, tcn = (np.array([row[key] for row in private_predictions], dtype=np.int64)
                      for key in ("label_id", "a_prediction", "tcn_prediction"))
    for row in private_predictions:
        if row["label_id"] != metadata[row["clip_id"]]["label_id"]:
            raise ValueError("prediction label differs from sealed metadata")
    variants = {"VideoMAE_only": block(labels, a, a), "VideoMAE_TCN": block(labels, tcn, a)}
    slices = {name: {} for name in ("source_kind", "duration_seconds", "context")}
    conditions = {
        "source_kind": {kind: np.array([metadata[row["clip_id"]]["source_kind"] == kind
                                        for row in private_predictions])
                        for kind in ("clinical", "network")},
        "duration_seconds": {name: np.array([lo <= metadata[row["clip_id"]]["duration"] < hi
                                               for row in private_predictions])
                             for name, lo, hi in (("lt_5", 0, 5), ("5_to_lt_15", 5, 15),
                                                 ("ge_15", 15, float("inf")))},
        "context": {"singleton": np.array([row["clip_id"] in isolated for row in private_predictions]),
                    "has_candidate_neighbor": np.array([row["clip_id"] not in isolated
                                                         for row in private_predictions])}}
    for family, choices in conditions.items():
        for name, mask in choices.items():
            slices[family][name] = {"VideoMAE_only": block(labels[mask], a[mask], a[mask]),
                                    "VideoMAE_TCN": block(labels[mask], tcn[mask], a[mask])}
    recordings = defaultdict(list)
    for position, row in enumerate(private_predictions):
        recordings[metadata[row["clip_id"]]["recording_key"]].append(position)
    # Each record is an aggregate statistical unit; export no keys/IDs.
    record_metrics = [block(labels[p], tcn[p], a[p]) for p in recordings.values()]
    record_a = [block(labels[p], a[p], a[p]) for p in recordings.values()]
    recording_aggregate = {
        "recording_count": len(recordings), "not_patient_independence": True,
        "unweighted_mean_record_accuracy": {
            "VideoMAE_only": float(np.mean([m["accuracy"] for m in record_a])),
            "VideoMAE_TCN": float(np.mean([m["accuracy"] for m in record_metrics]))},
        "unweighted_mean_present_class_macro_f1": {
            "VideoMAE_only": float(np.mean([m["present_class_macro_f1"] for m in record_a])),
            "VideoMAE_TCN": float(np.mean([m["present_class_macro_f1"] for m in record_metrics]))},
        "recordings_with_corrections": sum(m["paired_vs_A"]["improved"] > 0 for m in record_metrics),
        "recordings_with_harm": sum(m["paired_vs_A"]["worsened"] > 0 for m in record_metrics)}
    deltas = [{"label_id": old["label_id"], "support": old["support"],
               "f1_change": new["f1"] - old["f1"],
               "macro_f1_contribution": (new["f1"] - old["f1"]) / 7}
              for old, new in zip(variants["VideoMAE_only"]["per_class"],
                                  variants["VideoMAE_TCN"]["per_class"])]
    return {"variants": variants, "slices": slices, "recording_aggregate": recording_aggregate,
            "per_class_changes": deltas, "six_sample_class_caution": {
                "label_id": 6, "support": variants["VideoMAE_only"]["per_class"][6]["support"],
                "interpretation": "report complete seven-class metrics; small support is unstable"}}
