"""Read-only release audit: aggregate math, actual budgets, allowlist and hashes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_KEYS = {
    "schema", "snapshot_beijing_time", "selection_metric", "independent_test",
    "patient_independence_verified", "seed", "train_count", "val_count",
    "class_names", "selection_order", "selected_temporal", "confusion_convention",
    "variants", "paired_vs_original_ASFormer", "paired_vs_previous_interval",
    "source_summary_sha256",
}
METRIC_KEYS = {"accuracy", "confusion_matrix", "macro_f1", "micro_f1",
               "weighted_f1", "per_class"}
CLASS_KEYS = {"class_id", "class_name", "f1", "precision", "recall", "support"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    require(isinstance(a, (int, float)) and not isinstance(a, bool)
            and math.isfinite(a) and abs(a-b) < 1e-12, "metric does not match counts")


def validate_metrics(m, names, expected_count=None):
    require(set(m) == METRIC_KEYS, "unexpected metric fields")
    cm = m["confusion_matrix"]
    require(len(cm) == 7 and all(len(row) == 7 for row in cm),
            "seven-by-seven confusion required")
    require(all(type(v) is int and v >= 0 for row in cm for v in row),
            "aggregate integer counts required")
    total = sum(map(sum, cm))
    require(total > 0, "nonempty metrics")
    if expected_count is not None:
        require(total == expected_count, "validation count mismatch")
    require(len(m["per_class"]) == 7, "complete seven-class metrics required")
    f1s, weighted = [], 0.0
    for i, item in enumerate(m["per_class"]):
        require(set(item) == CLASS_KEYS and item["class_id"] == i
                and item["class_name"] == names[i], "class order changed")
        support, predicted, tp = sum(cm[i]), sum(row[i] for row in cm), cm[i][i]
        require(item["support"] == support, "class support mismatch")
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        close(item["precision"], precision)
        close(item["recall"], recall)
        close(item["f1"], f1)
        f1s.append(f1)
        weighted += f1*support
    accuracy = sum(cm[i][i] for i in range(7))/total
    close(m["accuracy"], accuracy)
    close(m["micro_f1"], accuracy)
    close(m["macro_f1"], sum(f1s)/7)
    close(m["weighted_f1"], weighted/total)
    return total


def validate_results(result, curves):
    require(set(result) == ROOT_KEYS, "unexpected public result fields")
    require(result["independent_test"] is False
            and result["patient_independence_verified"] is False,
            "unsupported independence claim")
    require(result["seed"] == 42 and result["train_count"] == 7306
            and result["val_count"] == 888, "registered counts/seed changed")
    order = result["selection_order"]
    require(list(result["variants"]) == order
            and list(curves["variants"]) == order, "candidate registry mismatch")
    require(curves["optimizer_steps_are_per_epoch_not_cumulative"] is True,
            "step semantics must be explicit")
    for name in order:
        variant = result["variants"][name]
        require(set(variant) == {"metrics", "resources", "completion"},
                "unexpected per-variant fields")
        validate_metrics(variant["metrics"], result["class_names"], 888)
        curve = curves["variants"][name]["curve"]
        resource = variant["resources"]
        require(len(curve) == resource["epochs_completed"], "history length mismatch")
        require([r[0] for r in curve] == list(range(1, len(curve)+1)),
                "history epochs not consecutive")
        require(all(len(r) == 5 and all(isinstance(v, (int, float))
                and not isinstance(v, bool) and math.isfinite(v) for v in r)
                and r[1] >= 0 and 0 <= r[2] <= 1 and 0 <= r[3] <= 1
                and r[4] == 192 for r in curve), "invalid epoch aggregate")
        require(sum(r[4] for r in curve) == resource["optimizer_steps"],
                "complete-epoch step ledger mismatch")
        best = max(curve, key=lambda r: r[2])
        require(best[0] == resource["best_epoch"], "best epoch mismatch")
        close(best[2], variant["metrics"]["macro_f1"])
        close(best[3], variant["metrics"]["accuracy"])
        if name in {"asformer_interval", "asformer_interval_self_reference"}:
            previous, stale, first_plateau = -1.0, 0, None
            for row in curve:
                if row[2] > previous:
                    previous, stale = row[2], 0
                else:
                    stale += 1
                if stale == 10:
                    first_plateau = row[0]
                    break
            require(first_plateau == len(curve), "first patience10 plateau mismatch")
    selected = max(order, key=lambda n: result["variants"][n]["metrics"]["macro_f1"])
    require(selected == result["selected_temporal"], "selection mismatch")
    pair = result["paired_vs_original_ASFormer"]
    old = result["variants"]["asformer"]["metrics"]["confusion_matrix"]
    new = result["variants"]["asformer_interval_self_reference"]["metrics"]["confusion_matrix"]
    old_correct, new_correct = sum(old[i][i] for i in range(7)), sum(new[i][i] for i in range(7))
    require(new_correct-old_correct == pair["improved"]-pair["worsened"],
            "paired change totals mismatch")
    require(pair["changed_count"] == pair["improved"]+pair["worsened"]
            +pair["wrong_to_different_wrong"], "paired changed total mismatch")
    return selected


def check_privacy_text(text):
    # Construct deployment patterns without embedding deployment addresses here.
    forbidden = (
        "/" + "root/", "/" + "Users/", "autodl" + "-tmp",
        "seeta" + "cloud", "fsn-" + "autodl", "GPU-" + "b301",
        "BEGIN " + "PRIVATE KEY",
    )
    require(not any(v in text for v in forbidden), "private deployment text rejected")
    require("\x00" not in text, "binary content rejected")


def scan_metrics(value, names):
    if isinstance(value, dict):
        if "confusion_matrix" in value:
            validate_metrics(value, names)
        else:
            for v in value.values():
                scan_metrics(v, names)
    elif isinstance(value, list):
        for v in value:
            scan_metrics(v, names)


def audit_release(root=ROOT):
    manifest = json.loads((root/"reports/public_release_manifest.json").read_text())
    allowed = set(manifest["files"]) | set(manifest["unhashed_audit_files"])
    actual = {
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        and not any(part in {".pytest_cache", "__pycache__"} for part in p.parts)
    }
    require(actual == allowed, "release file allowlist mismatch")
    for name in sorted(actual):
        path = root/name
        require(not path.is_symlink(), "symlinks rejected")
        raw = path.read_bytes()
        check_privacy_text(raw.decode("utf-8"))
        if name in manifest["files"]:
            require(hashlib.sha256(raw).hexdigest() == manifest["files"][name],
                    "release content changed after sealing")
    result = json.loads((root/"reports/results.json").read_text())
    curves = json.loads((root/"reports/epoch_curves.json").read_text())
    selected = validate_results(result, curves)
    scan_metrics(json.loads((root/"reports/context_diagnostics.json").read_text()),
                 result["class_names"])
    return {"passed": True, "files": len(actual), "accepted_candidates": 8,
            "selected_temporal": selected,
            "aggregate_counts_and_curves_consistent": True,
            "deployment_pattern_scan_passed": True,
            "exact_file_allowlist_and_hashes_verified": True,
            "privacy_scope": "code_synthetic_tests_and_deidentified_aggregates_only",
            "human_content_review_still_required": True}


if __name__ == "__main__":
    print(json.dumps(audit_release(), ensure_ascii=False, sort_keys=True))
