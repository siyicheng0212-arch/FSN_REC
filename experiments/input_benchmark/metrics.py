"""Paired recording-cluster development intervals, never patient/test claims."""
import numpy as np

from .model import CONFIG


def confusion_f1(matrix):
    true, predicted, tp = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    denominator = true + predicted
    return float(np.divide(2 * tp, denominator, out=np.zeros(7), where=denominator > 0).mean())


def recording_interval(labels, current, control, clip_ids, metadata, replicates=None):
    """Resample the same whole recordings for both models; private keys never return."""
    n = len(labels)
    if len(current) != n or len(control) != n or len(clip_ids) != n or len(set(clip_ids)) != n:
        raise ValueError("paired bootstrap requires identical unique clip coverage")
    by_id = {r["clip_id"]: r for r in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("recording metadata contains duplicate clip identities")
    if any(np.asarray(x).dtype.kind not in "iu" or np.any((np.asarray(x) < 0) | (np.asarray(x) >= 7))
           for x in (labels, current, control)):
        raise ValueError("bootstrap requires integer predictions in the fixed seven-class space")
    records = {}
    for i, clip in enumerate(clip_ids):
        row = by_id[clip]
        if row["split"] != "val" or not isinstance(row.get("recording_key"), str) or not row["recording_key"]:
            raise ValueError("recording bootstrap requires sealed val identities")
        records.setdefault(row["recording_key"], []).append(i)
    if len(records) < 2:
        return {"identifiable": False, "recording_count": len(records), "reason": "fewer_than_two_recordings"}
    matrices = []
    for indices in records.values():
        indices = np.array(indices)
        matrices.append([np.bincount(np.asarray(labels)[indices] * 7 + np.asarray(pred)[indices], minlength=49).reshape(7, 7)
                         for pred in (current, control)])
    matrices = np.asarray(matrices)
    count = CONFIG["record_bootstrap"]["replicates"] if replicates is None else replicates
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("bootstrap needs at least two replicates")
    rng = np.random.default_rng(CONFIG["record_bootstrap"]["seed"])
    differences = []
    for _ in range(count):
        sampled = matrices[rng.integers(0, len(matrices), len(matrices))].sum(0)
        differences.append(confusion_f1(sampled[0]) - confusion_f1(sampled[1]))
    full = matrices.sum(0)
    return {"identifiable": True, "recording_count": len(records), "replicates": count,
            "seed": CONFIG["record_bootstrap"]["seed"], "paired_macro_f1_difference": confusion_f1(full[0]) - confusion_f1(full[1]),
            "percentile_95_interval": np.quantile(differences, [.025, .975]).tolist(),
            "scope": "conditional_recording_resampling_of_repeated_development_val_not_confirmatory_significance",
            "patient_independence_verified": False, "rare_class_may_be_absent_in_resamples": True,
            "seven_classes_always_retained": True}
