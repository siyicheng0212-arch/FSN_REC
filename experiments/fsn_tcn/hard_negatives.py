"""Train-only, within-source cross-record negatives for visual relation fitting.

These pairs are deliberately *not* legitimate temporal candidates. Keep them in
the relation-training table and never pass them to the formal chain builder.
No action labels are inspected when selecting endpoints.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import re


ORIGIN = "synthetic_cross_record_v1"


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _sha(row, key):
    value = _text(row, key)
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"{key} must be a lowercase SHA256")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def validate_metadata(metadata):
    """Copy and validate train/val metadata without reading action labels.

    Recording, group and source-video identities may never cross splits. Unknown
    sources and test rows are rejected rather than silently filtered out.
    """
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("metadata must be a nonempty clip_id -> row dictionary")
    result, identities = {}, {key: {} for key in ("group_id", "recording_key", "source_video_key")}
    for clip, raw in sorted(metadata.items()):
        if not isinstance(clip, str) or not clip.strip() or not isinstance(raw, dict):
            raise ValueError("metadata requires nonempty clip IDs and object rows")
        row = dict(raw)
        if _text(row, "clip_id") != clip:
            raise ValueError("metadata dictionary key differs from row clip_id")
        group = _text(row, "group_id")
        collection = _text(row, "source_collection")
        recording, video = _sha(row, "recording_key"), _sha(row, "source_video_key")
        if row.get("split") not in {"train", "val"}:
            raise ValueError("metadata split must be train or val; test is not read")
        if row.get("source_kind") not in {"clinical", "network"}:
            raise ValueError("metadata source_kind must be clinical or network")
        if "timebase_key" in row:
            _sha(row, "timebase_key")
        start = _number(row.get("clip_start_sec"), "clip_start_sec")
        end = _number(row.get("clip_end_sec"), "clip_end_sec")
        if start < 0 or end <= start:
            raise ValueError("clip interval must satisfy 0 <= start < end")
        for field in ("duration", "clip_duration_sec"):
            if field in row and not math.isclose(_number(row[field], field), end - start, abs_tol=1e-6):
                raise ValueError(f"{field} does not match clip interval")
        signature = (row["split"], row["source_kind"], collection)
        for key, identity in (("group_id", group), ("recording_key", recording), ("source_video_key", video)):
            previous = identities[key].setdefault(identity, signature)
            if previous[0] != signature[0]:
                raise ValueError(f"train/val {key} leakage")
            if previous != signature:
                raise ValueError(f"inconsistent source metadata for {key}")
        row.update(clip_start_sec=start, clip_end_sec=end)
        result[clip] = row
    return result


def _same_record(left, right):
    keys = ("group_id", "recording_key", "source_video_key", "source_collection", "source_kind")
    return all(left[key] == right[key] for key in keys) and left.get("timebase_key") == right.get("timebase_key")


def _different_record(left, right):
    return all(left[key] != right[key] for key in ("group_id", "recording_key", "source_video_key"))


def validate_original_edges(edges, metadata):
    """Require original within-record candidates; synthetic pairs stay separate."""
    if not isinstance(edges, list):
        raise ValueError("original_edges must be a list")
    output, pairs, identifiers = [], set(), set()
    for raw in edges:
        if not isinstance(raw, dict):
            raise ValueError("edge rows must be objects")
        row = dict(raw)
        if str(row.get("label_origin", "")).startswith("synthetic_"):
            raise ValueError("synthetic pairs cannot be supplied as original candidates")
        left_id, right_id = _text(row, "left_clip_id"), _text(row, "right_clip_id")
        if left_id == right_id or (left_id, right_id) in pairs:
            raise ValueError("original edges must be unique directed, non-self pairs")
        if left_id not in metadata or right_id not in metadata:
            raise ValueError("original edge has an unknown endpoint")
        left, right = metadata[left_id], metadata[right_id]
        if row.get("split") not in {"train", "val"} or left["split"] != row["split"] or right["split"] != row["split"]:
            raise ValueError("original edge endpoint split mismatch")
        if not _same_record(left, right) or right["clip_start_sec"] < left["clip_end_sec"]:
            raise ValueError("original candidate edge crosses a record or nonoverlap boundary")
        if row.get("status") not in {"C", "D", "U"}:
            raise ValueError("original edge status must be C, D or U")
        if row.get("source_kind", left["source_kind"]) != left["source_kind"]:
            raise ValueError("original edge source kind mismatch")
        if row["status"] == "C" and left["source_kind"] != "clinical":
            raise ValueError("C edge must use clinical source metadata")
        if "edge_id" in row:
            identifier = _text(row, "edge_id")
            if identifier in identifiers:
                raise ValueError("duplicate original edge_id")
            identifiers.add(identifier)
        pairs.add((left_id, right_id))
        output.append(row)
    return output


def build_hard_negatives(metadata, original_edges, *, seed=42, ratio=1.0):
    """Generate D pairs from clinical train C left endpoints only.

    The requested total is ``ceil(number_of_train_C * ratio)``. Requests are
    allocated deterministically across sorted positives. If eligible distinct
    candidates run out, the audit explicitly reports unfilled requests.
    """
    checked = validate_metadata(metadata)
    edges = validate_original_edges(original_edges, checked)
    rng = random.Random(_seed(seed))
    ratio = _number(ratio, "ratio")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    positives = sorted((row for row in edges if row["split"] == "train" and row["status"] == "C"),
                       key=lambda row: (row["left_clip_id"], row["right_clip_id"]))
    if not positives:
        raise ValueError("no clinical train C edges available for hard-negative anchors")
    pools = defaultdict(list)
    for clip, row in checked.items():
        if row["split"] == "train" and row["source_kind"] == "clinical":
            pools[row["source_collection"]].append(clip)
    existing = {(row["left_clip_id"], row["right_clip_id"]) for row in edges}
    negatives, skipped, insufficient = [], 0, 0
    requested_by_source, generated_by_source = Counter(), Counter()
    for index, positive in enumerate(positives):
        left_id = positive["left_clip_id"]
        left = checked[left_id]
        requested = math.ceil((index + 1) * ratio) - math.ceil(index * ratio)
        collection = left["source_collection"]
        requested_by_source[collection] += requested
        candidates = [clip for clip in pools[collection]
                      if _different_record(left, checked[clip]) and (left_id, clip) not in existing]
        rng.shuffle(candidates)
        selected = candidates[:requested]
        skipped += requested - len(selected)
        insufficient += len(selected) < requested
        for right_id in selected:
            pair = (left_id, right_id)
            digest = hashlib.sha256(json.dumps(["train", *pair], separators=(",", ":")).encode()).hexdigest()
            negatives.append({
                "edge_id": "hard-edge-" + digest[:24], "left_clip_id": left_id, "right_clip_id": right_id,
                "split": "train", "status": "D", "label_origin": ORIGIN,
                "source_kind": "clinical", "source_collection": collection,
                "reason": "same_source_collection_different_record_group_and_video",
                "synthetic_pair": True, "eligible_for_formal_chain": False,
            })
            existing.add(pair)
            generated_by_source[collection] += 1
    if not negatives:
        raise ValueError("no hard negatives could be generated: need distinct clinical records within one source collection")
    audit = {
        "label_origin": ORIGIN, "seed": seed, "ratio": ratio,
        "positive_train_anchors": len(positives), "requested_negatives": math.ceil(len(positives) * ratio),
        "generated_negatives": len(negatives), "skipped_requests": skipped,
        "anchors_with_insufficient_candidates": insufficient,
        "requested_by_source": dict(sorted(requested_by_source.items())),
        "generated_by_source": dict(sorted(generated_by_source.items())),
        "train_only": True, "actions_used_to_construct_pairs": False,
        "unique_directed_pairs": True, "formal_chain_candidates": False,
        "same_source_collection": True, "distinct_record_group_and_video": True,
        "limitation": "Synthetic cross-record negatives are known constructed disconnections, not annotations of real edit boundaries.",
    }
    return negatives, audit
