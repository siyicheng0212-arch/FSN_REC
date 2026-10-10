"""Read-only validation of the existing VideoMAE export and candidate layout."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from experiments.aligned_protocol import DATA_PROTOCOL, EXPECTED_COUNTS, EXPECTED_MANIFEST_SHA
from experiments.relation.data import load_edges, load_feature_index, read_jsonl, sha256_file
from experiments.relation.evaluate_suite import classification_metrics
from experiments.relation.protocol import SOURCE_FILES, validate_source_artifacts
from experiments.relation.tcn_baseline import TCNConfig, build_segments
from experiments.relation.videomae2_followup import SCHEMA as EXPORT_SCHEMA


SCHEMA = "fsn-videomae-plain-tcn-v1"
TRAINING = {"seed": 42, "epochs": 30, "patience": 5, "batch_size_segments": 16,
            "hidden_dim": 64, "blocks": 3, "dropout": 0.1,
            "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.01,
            "clip_grad_norm": 5.0, "loss": "unweighted_valid_clip_cross_entropy"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _export_seal(index_path, index, source_dir):
    seal = _json(index_path.parent / "protocol.json")
    required = {"schema": EXPORT_SCHEMA, "status": "complete", "A_frozen": True,
                "index_sha256": sha256_file(index_path),
                "checkpoint_sha256": index.checkpoint_sha256,
                "source_audit_sha256": sha256_file(source_dir / "audit.json"),
                "data_protocol": DATA_PROTOCOL, "split_counts": EXPECTED_COUNTS,
                "test_read": False, "val_per_clip_A_logits_equal_separate_best_replay": True,
                "global_tokens": "last_block_spatial_mean_per_sampled_tubelet",
                "local_tokens": "last_block_spatial_max_per_sampled_tubelet"}
    required.update({split + "_manifest_sha256": value
                     for split, value in EXPECTED_MANIFEST_SHA.items()})
    for key, expected in required.items():
        actual = seal.get(key)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            raise ValueError(f"VideoMAE export seal disagrees: {key}")
    if any(tuple(index.shapes[name]) != shape for name, shape in {
            "global_tokens": (8, 768), "local_tokens": (8, 768),
            "global_positions": (8,), "local_positions": (8,)}.items()):
        raise ValueError("expected existing eight-tubelet ViT-B/16 features at 768 channels")
    # Every clip must have a numeric-file seal, not just the first clip or index.
    if any(not row.get("feature_sha256") for row in index.clips.values()):
        raise ValueError("every frozen clip feature requires a SHA256 seal")
    return seal


def _disjoint_recordings(metadata):
    for field in ("group_id", "source_video_key", "recording_key"):
        groups = {split: set() for split in ("train", "val")}
        for row in metadata:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing recording identity field: {field}")
            groups[row["split"]].add(value)
        if groups["train"] & groups["val"]:
            raise ValueError(f"train/val overlap in {field}; preserve scene for audit")


def _candidate_time_checks(metadata, edges, audit):
    if audit.get("max_gap_seconds") != 5.0:
        raise ValueError("sealed candidate protocol must retain its fixed five-second cutoff")
    by_id = {row["clip_id"]: row for row in metadata}
    for edge in edges:
        left, right = (by_id[edge[key]] for key in ("left_clip_id", "right_clip_id"))
        times = [row.get(key) for row in (left, right)
                 for key in ("clip_start_sec", "clip_end_sec")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) for value in times):
            raise ValueError("candidate edge has missing/nonfinite time coordinates")
        ls, le, rs, re = times
        if ls < 0 or rs < 0 or le <= ls or re <= rs or rs <= ls or not 0 <= rs - le <= 5:
            raise ValueError("candidate edge has overlapping, reversed or excessive-gap intervals")


def prepare(index_path, source_dir):
    """Return the original model's Segment objects; never load A or read test."""
    index_path, source_dir = Path(index_path).resolve(), Path(source_dir).resolve()
    index = load_feature_index(index_path)  # hashes and reads all numeric features
    source = validate_source_artifacts(source_dir, index, source_dir / "edges.jsonl")
    counts = {split: sum(row["split"] == split for row in index.clips.values())
              for split in ("train", "val")}
    if counts != EXPECTED_COUNTS:
        raise ValueError("actual feature index must contain exactly train7372/val823")
    seal = _export_seal(index_path, index, source_dir)
    metadata = read_jsonl(source_dir / "manifest_metadata.jsonl")
    _disjoint_recordings(metadata)
    edges = load_edges(source_dir / "edges.jsonl", index)
    _candidate_time_checks(metadata, edges, source)
    segments = {split: build_segments(
        index, metadata, read_jsonl(source_dir / f"chains_{split}.jsonl"), edges, split)
        for split in ("train", "val")}
    for split, items in segments.items():
        if sum(len(item.clip_ids) for item in items) != counts[split]:
            raise ValueError("candidate segments lost clips")
    labels = np.concatenate([s.labels for s in segments["val"]])
    predictions = np.concatenate([s.a_logits.argmax(1) for s in segments["val"]])
    replay = classification_metrics(labels, predictions)
    exported = seal.get("A_val_metrics", {})
    # compute_classification_metrics (exporter) uses confusion_matrix/num_samples.
    for key in ("accuracy", "macro_f1"):
        value = exported.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or \
                not math.isfinite(value) or not math.isclose(replay[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"VideoMAE A-only replay differs from sealed export: {key}")
    if replay["confusion"] != exported.get("confusion_matrix") or \
            exported.get("num_samples") != counts["val"]:
        raise ValueError("VideoMAE A-only confusion/count differs from sealed export")
    paths = {"index": index_path, "export_protocol": index_path.parent / "protocol.json",
             "source_audit": source_dir / "audit.json"}
    paths.update({name: source_dir / name for name in SOURCE_FILES})
    input_sha = {name: sha256_file(path) for name, path in paths.items()}
    config = TCNConfig(index.global_dim, index.local_dim, TRAINING["hidden_dim"],
                       TRAINING["blocks"], TRAINING["dropout"])
    model_config = asdict(config)
    # Older sealed VideoMAE runs omitted the dilation field because the default
    # schedule was implicit. Keep their evidence payload byte-for-byte stable;
    # an explicit local-TCN schedule is recorded by the training checkpoint.
    if config.dilations is None:
        model_config.pop("dilations", None)
    info = {"counts": counts, "manifest_sha256": source["manifest_sha256"],
            "A_checkpoint_sha256": index.checkpoint_sha256, "input_sha256": input_sha,
            "model_config": model_config, "A_only": replay,
            "segments": {split: {"count": len(items),
                        "singleton_clips": sum(len(s.clip_ids) == 1 for s in items),
                        "maximum_length": max(len(s.clip_ids) for s in items)}
                         for split, items in segments.items()}}
    return segments, metadata, info
