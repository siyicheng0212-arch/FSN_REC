"""Freeze a matched-size taxonomy from training-only feature prototypes.

This is a data-derived control, not a proposed novel clustering algorithm.
No validation/test scores or confusion matrices are accepted by this CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def derive_visual_taxonomy(features: np.ndarray, labels: np.ndarray) -> dict:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or y.ndim != 1 or x.shape[1] < 1:
        raise ValueError("expected features [N,D] and labels [N]")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError("non-finite features or non-integer labels")
    if set(y.tolist()) != set(range(7)):
        raise ValueError("training features must cover exactly seven classes")
    counts = [int((y == c).sum()) for c in range(7)]
    if min(counts) < 2:
        raise ValueError("at least two training examples per class required")
    sample_norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (sample_norms <= 1e-12).any():
        raise ValueError("zero feature vector")
    x = x / sample_norms
    prototypes = np.stack([x[y == c].mean(axis=0) for c in range(7)])
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("degenerate class prototype")
    similarities = (prototypes / norms) @ (prototypes / norms).T
    candidates = []
    # The two singleton groups are unordered: 21 * 10 = 210 partitions.
    for singles in itertools.combinations(range(7), 2):
        remaining = sorted(set(range(7)) - set(singles))
        for triple in itertools.combinations(remaining, 3):
            pair = tuple(sorted(set(remaining) - set(triple)))
            groups = ((singles[0],), (singles[1],), tuple(triple), pair)
            score = sum(float(similarities[a, b]) for g in groups for a, b in itertools.combinations(g, 2))
            candidates.append((score, groups))
    score, groups = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return {"name": "train_feature_prototypes_v1", "groups": [list(g) for g in groups],
            "method": "maximum sum of within-group cosine similarities of class-balanced training prototypes",
            "group_sizes": [1, 1, 3, 2], "num_candidate_partitions": len(candidates),
            "objective": score, "training_class_counts": counts,
            "not_claimed": "not a novel clustering algorithm; not an independent test result"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    if meta.get("split") != "train" or not meta.get("protocol_sha256") or not meta.get("training_manifest_sha256"):
        raise SystemExit("metadata must identify a frozen training split, protocol and manifest SHA")
    data = np.load(args.features, allow_pickle=False)
    actual_features_sha = hashlib.sha256(args.features.read_bytes()).hexdigest()
    if meta.get("features_sha256") and meta["features_sha256"] != actual_features_sha:
        raise SystemExit("exported feature file does not match its provenance SHA")
    result = derive_visual_taxonomy(data["features"], data["labels"])
    result["provenance"] = {k: meta[k] for k in ("split", "protocol_sha256", "training_manifest_sha256")}
    result["protocol_sha256"] = meta["protocol_sha256"]
    result["training_manifest_sha256"] = meta["training_manifest_sha256"]
    result["class_names"] = ["消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定"]
    result["features_sha256"] = actual_features_sha
    result["feature_encoder_selection_exposure"] = meta.get("feature_encoder_selection_exposure", "not_declared")
    result["provenance"]["feature_encoder_selection_exposure"] = result["feature_encoder_selection_exposure"]
    result["metadata_sha256"] = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
