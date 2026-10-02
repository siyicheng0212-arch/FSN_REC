"""Deterministic train-only anchor views for context-utility experiments.

An entry covers an original structural segment exactly once. Its normal view
keeps every original label. Its optional perturbed view replaces paired frozen
visual features/A logits, and may be supervised *only at anchor_positions*.
Plans contain private clip identifiers: save runtime plans outside git.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random

import torch

from experiments.fsn_tcn.hard_negatives import validate_metadata
from experiments.fsn_tcn.io import validate_chains


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def plan_hash(plan):
    """Hash the complete ordered private plan, including donor/anchor choices."""
    return _digest(plan)


def _segments(bundle):
    metadata = validate_metadata(bundle.metadata)
    if "train" not in bundle.chains:
        raise ValueError("bundle has no train chains")
    validate_chains(bundle.chains["train"], metadata, "train")
    segments = []
    for chain in bundle.chains["train"]:
        ids = chain["ordered_clip_ids"]
        start = 0
        for position in range(1, len(ids) + 1):
            if position == len(ids) or not chain["eligible"][position - 1]:
                segment = list(ids[start:position])
                segments.append({"segment_id": "segment-" + _digest(segment),
                                 "ordered_clip_ids": segment})
                start = position
    return metadata, sorted(segments, key=lambda entry: entry["segment_id"])


def _eligible_donor(anchor, donor):
    return (anchor["split"] == donor["split"] == "train"
            and anchor["source_kind"] == donor["source_kind"] == "clinical"
            and anchor["source_collection"] == donor["source_collection"]
            and all(anchor[key] != donor[key]
                    for key in ("group_id", "recording_key", "source_video_key")))


def build_epoch_plan(bundle, *, seed: int, epoch: int,
                     anchors_per_segment: int = 1, max_neighbors: int = 2):
    """Build label-blind independent-RNG views; singleton/network views are clean.

    ``max_neighbors`` is the largest left/right clip offset, not a sample count.
    Zero constructed replacements are returned explicitly in the audit; the
    formal caller must reject that condition rather than silently alter data.
    """
    _integer(seed, "seed")
    _integer(epoch, "epoch")
    _integer(anchors_per_segment, "anchors_per_segment", minimum=1)
    _integer(max_neighbors, "max_neighbors", minimum=1)
    metadata, segments = _segments(bundle)
    pools = defaultdict(list)
    for clip, row in sorted(metadata.items()):
        if row["split"] == "train" and row["source_kind"] == "clinical":
            pools[row["source_collection"]].append(clip)
    rng = random.Random(int(_digest(["context-utility-plan-v1", seed, epoch]), 16))
    counters = Counter()
    by_source = Counter()
    plan = []
    for segment in segments:
        ids = segment["ordered_clip_ids"]
        source = metadata[ids[0]]
        candidates = [clip for clip in pools[source["source_collection"]]
                      if _eligible_donor(source, metadata[clip])]
        counters["segments"] += 1
        counters["singleton_segments"] += len(ids) == 1
        counters["clinical_segments"] += source["source_kind"] == "clinical"
        counters["network_segments"] += source["source_kind"] == "network"
        counters["eligible_replacement_segments"] += len(ids) > 1 and bool(candidates)
        counters["clinical_segments_without_donor"] += source["source_kind"] == "clinical" and not candidates
        positions = list(range(len(ids)))
        rng.shuffle(positions)
        anchors = sorted(positions[:min(len(ids), anchors_per_segment)])
        neighbor_positions = sorted({position for anchor in anchors
                                     for position in range(max(0, anchor - max_neighbors),
                                                           min(len(ids), anchor + max_neighbors + 1))
                                     if position not in anchors})
        replacements = []
        if source["source_kind"] == "clinical":
            counters["requested_clinical_replacements"] += len(neighbor_positions)
            for position in neighbor_positions:
                if not candidates:
                    counters["unfilled_replacements"] += 1
                    continue
                donor = candidates[rng.randrange(len(candidates))]
                replacements.append({"position": position, "donor_clip_id": donor})
                by_source[source["source_collection"]] += 1
        counters["anchor_positions"] += len(anchors)
        counters["actual_replacements"] += len(replacements)
        counters["segments_with_replacements"] += bool(replacements)
        counters["clean_view_segments"] += not replacements
        plan.append(dict(segment, anchor_positions=anchors, replacements=replacements))
    rng.shuffle(plan)
    validate_plan(bundle, plan, max_neighbors=max_neighbors)
    audit = {key: counters[key] for key in (
        "segments", "singleton_segments", "clinical_segments", "network_segments",
        "eligible_replacement_segments", "clinical_segments_without_donor", "anchor_positions",
        "requested_clinical_replacements", "actual_replacements", "unfilled_replacements",
        "segments_with_replacements", "clean_view_segments")}
    audit.update({"version": "anchor_neighbor_replacement_v1", "seed": seed, "epoch": epoch,
                  "anchors_per_segment": anchors_per_segment, "max_neighbors": max_neighbors,
                  "original_train_clip_coverage": sum(len(e["ordered_clip_ids"]) for e in plan),
                  "original_train_clips_exactly_once": True,
                  "replacements_by_collection": dict(sorted(by_source.items())),
                  "train_only": True, "action_labels_used": False,
                  "has_constructed_replacements": counters["actual_replacements"] > 0,
                  "plan_sha256": plan_hash(plan),
                  "limitation": "Constructed unrelated neighbors are not verified real edit boundaries."})
    return plan, audit


def validate_plan(bundle, plan, *, max_neighbors=None):
    """Reject missing/repeated clips, structural bridges, and invalid donors."""
    metadata, segments = _segments(bundle)
    if max_neighbors is not None:
        _integer(max_neighbors, "max_neighbors", minimum=1)
    if not isinstance(plan, list):
        raise ValueError("plan must be a list")
    canonical = {entry["segment_id"]: entry["ordered_clip_ids"] for entry in segments}
    seen = set()
    for entry in plan:
        if not isinstance(entry, dict):
            raise ValueError("plan entry must be an object")
        segment_id = entry.get("segment_id")
        if not isinstance(segment_id, str) or segment_id not in canonical or segment_id in seen:
            raise ValueError("unknown or repeated original structural segment")
        seen.add(segment_id)
        ids = entry.get("ordered_clip_ids")
        if ids != canonical[segment_id]:
            raise ValueError("plan changed clip coverage, split, ordering or structural cuts")
        anchors = entry.get("anchor_positions")
        if (not isinstance(anchors, list) or not anchors or any(type(p) is not int for p in anchors)
                or len(anchors) != len(set(anchors)) or any(p < 0 or p >= len(ids) for p in anchors)):
            raise ValueError("anchor positions must be nonempty unique in-segment integers")
        replacements = entry.get("replacements")
        if not isinstance(replacements, list):
            raise ValueError("replacements must be a list")
        changed_positions = set()
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise ValueError("replacement must be an object")
            position, donor_id = replacement.get("position"), replacement.get("donor_clip_id")
            if (type(position) is not int or position < 0 or position >= len(ids)
                    or position in anchors or position in changed_positions):
                raise ValueError("replacement position must be unique and may not replace an anchor")
            if max_neighbors is not None and not any(abs(position - anchor) <= max_neighbors for anchor in anchors):
                raise ValueError("replacement lies outside configured anchor neighborhood")
            changed_positions.add(position)
            if not isinstance(donor_id, str) or donor_id not in metadata:
                raise ValueError("unknown donor clip")
            if not all(_eligible_donor(metadata[ids[anchor]], metadata[donor_id]) for anchor in anchors):
                raise ValueError("donor must be train-only clinical, same collection, distinct group/record/video")
    if seen != set(canonical):
        raise ValueError("each original train structural segment/clip must occur exactly once")
    return True


def apply_view(features, a_logits, entry, read_clip):
    """Pairwise frozen feature/logit replacement; never reads or returns labels."""
    if (not isinstance(features, torch.Tensor) or not isinstance(a_logits, torch.Tensor)
            or features.ndim != 2 or a_logits.ndim != 2
            or len(features) != len(a_logits) or a_logits.shape[1] != 7
            or not features.is_floating_point() or not a_logits.is_floating_point()
            or not torch.isfinite(features).all() or not torch.isfinite(a_logits).all()):
        raise ValueError("features[T,F] and finite floating A logits[T,7] are required")
    if not isinstance(entry, dict) or len(entry.get("ordered_clip_ids", [])) != len(features):
        raise ValueError("entry clip count differs from feature/logit sequence")
    anchors = entry.get("anchor_positions")
    if (not isinstance(anchors, list) or not anchors or any(type(p) is not int for p in anchors)
            or len(anchors) != len(set(anchors)) or any(p < 0 or p >= len(features) for p in anchors)):
        raise ValueError("invalid anchor positions")
    replacements = entry.get("replacements")
    if not isinstance(replacements, list):
        raise ValueError("replacements must be a list")
    output_features, output_logits = features.clone(), a_logits.clone()
    seen = set()
    for row in replacements:
        if not isinstance(row, dict):
            raise ValueError("replacement must be an object")
        position, donor_id = row.get("position"), row.get("donor_clip_id")
        if (type(position) is not int or position < 0 or position >= len(features)
                or position in anchors or position in seen or not isinstance(donor_id, str)):
            raise ValueError("invalid, repeated or anchor replacement position")
        seen.add(position)
        pair = read_clip(donor_id)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("read_clip must return paired donor features and A logits")
        donor_features = torch.as_tensor(pair[0], device=features.device)
        donor_logits = torch.as_tensor(pair[1], device=a_logits.device)
        if (donor_features.shape != features.shape[1:] or donor_logits.shape != (7,)
                or not donor_features.is_floating_point() or not donor_logits.is_floating_point()
                or not torch.isfinite(donor_features).all() or not torch.isfinite(donor_logits).all()):
            raise ValueError("donor feature/logit shape, dtype or finite-value mismatch")
        donor_features = donor_features.to(dtype=features.dtype)
        donor_logits = donor_logits.to(dtype=a_logits.dtype)
        if not torch.isfinite(donor_features).all() or not torch.isfinite(donor_logits).all():
            raise ValueError("donor feature/logit conversion produced nonfinite values")
        output_features[position] = donor_features
        output_logits[position] = donor_logits
    return output_features, output_logits, list(anchors)
