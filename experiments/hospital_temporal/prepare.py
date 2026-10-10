"""CPU-only structural-chain audit, not temporal training or R+D preparation.

Reuse the established candidate rules, but do not generate source-policy
targets, source gates, relation negatives or a transition table. Source names
are aggregate reporting fields only. A new A trained without Menzhen is still
required before the subsequent temporal models can be trained fairly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.hospital_holdout.prepare import (
    EXPECTED_COUNTS, OLD_SHA, encoded, rows, sha, validate_partition,
)
from experiments.hospital_holdout.video_data import manifest_rows
from experiments.relation.source_edges import _records

SCHEMA = "fsn-hospital-temporal-chain-audit-v1"


def build_chains(records, max_gap_seconds=5.0):
    """Match the old structural masks, without its C/D policy supervision.

    Nested overlaps are detected using the covered end, not just the preceding
    clip's end. No clip is removed or skipped to bridge a structural cut.
    Action labels and source names do not determine eligible connections.
    """
    if isinstance(max_gap_seconds, bool) or not isinstance(max_gap_seconds, (int, float)) \
            or not math.isfinite(max_gap_seconds) or max_gap_seconds != 5.0:
        raise ValueError("the existing structural cutoff is fixed at five seconds")
    records = _records(records)
    if any(row.get("time_unit_verified") is not True for row in records):
        raise ValueError("all clip timestamp units must have been verified")
    buckets = defaultdict(list)
    for row in records:
        key = (row["split"], row["group_id"], row["video_path"],
               row["timebase_id"], row["record_id"] or "")
        buckets[key].append(row)
    chains, segments, excluded = [], {"train": [], "val": []}, Counter()
    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda row: (
            row["clip_start_sec"], row["clip_end_sec"], row["clip_id"]))
        starts = Counter(row["clip_start_sec"] for row in ordered)
        covered_end, masks, reasons = ordered[0]["clip_end_sec"], [], []
        current = [ordered[0]["clip_id"]]
        for left, right in zip(ordered, ordered[1:]):
            if starts[left["clip_start_sec"]] > 1 or starts[right["clip_start_sec"]] > 1:
                reason = "duplicate_start"
            elif right["clip_start_sec"] < covered_end:
                reason = "overlap_or_nested_interval"
            elif right["clip_start_sec"] - left["clip_end_sec"] > max_gap_seconds:
                reason = "gap_exceeds_cutoff"
            else:
                reason = "ordered_nonoverlap_candidate"
            flag = reason == "ordered_nonoverlap_candidate"
            masks.append(flag)
            reasons.append(reason)
            covered_end = max(covered_end, right["clip_end_sec"])
            if flag:
                current.append(right["clip_id"])
            else:
                excluded[(key[0], reason)] += 1
                segments[key[0]].append(current)
                current = [right["clip_id"]]
        segments[key[0]].append(current)
        identity = hashlib.sha256(encoded(list(key))).hexdigest()
        chains.append({"chain_id": identity, "split": key[0],
                       "ordered_clip_ids": [row["clip_id"] for row in ordered],
                       "eligible": masks, "candidate_reasons": reasons})
    for split in ("train", "val"):
        expected = {row["clip_id"] for row in records if row["split"] == split}
        actual = [clip for segment in segments[split] for clip in segment]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("segments must partition all clips exactly once")
    return records, chains, segments, excluded


def aggregate(records, chains, segments, excluded):
    by_id = {row["clip_id"]: row for row in records}
    result = {}
    for split in ("train", "val"):
        subset = [row for row in records if row["split"] == split]
        lengths = [len(segment) for segment in segments[split]]
        neighbor_ids = {clip for segment in segments[split] if len(segment) > 1 for clip in segment}
        sources = {}
        for source in sorted({row["source_collection"] for row in subset}):
            source_rows = [row for row in subset if row["source_collection"] == source]
            source_segments = [segment for segment in segments[split]
                               if by_id[segment[0]]["source_collection"] == source]
            sources[source] = {"clips": len(source_rows),
                "recordings": len({row["record_id"] for row in source_rows}),
                "groups": len({row["group_id"] for row in source_rows}),
                "videos": len({row["video_path"] for row in source_rows}),
                "candidate_edges": sum(len(segment) - 1 for segment in source_segments),
                "neighbor_clips": sum(row["clip_id"] in neighbor_ids for row in source_rows),
                "isolated_clips": sum(row["clip_id"] not in neighbor_ids for row in source_rows),
                "class_counts": [sum(row["label_id"] == label for row in source_rows) for label in range(7)]}
        result[split] = {"clips": len(subset), "recording_chains": sum(
            chain["split"] == split for chain in chains), "segments": len(lengths),
            "candidate_edges": sum(length - 1 for length in lengths),
            "neighbor_clips": len(neighbor_ids), "isolated_clips": lengths.count(1),
            "segment_length_histogram": dict(sorted(Counter(lengths).items())),
            "segment_length_mean": statistics.mean(lengths),
            "segment_length_median": statistics.median(lengths),
            "segment_length_max": max(lengths),
            "class_neighbor_counts": [sum(row["label_id"] == label and row["clip_id"] in neighbor_ids
                                          for row in subset) for label in range(7)],
            "excluded_adjacencies": {reason: count for (role, reason), count in sorted(excluded.items())
                                     if role == split}, "sources": sources}
    return result


def audit(protocol_dir, historical_train=None):
    protocol_dir = Path(protocol_dir).resolve()
    train, protocol = manifest_rows(protocol_dir, "train")
    val, _ = manifest_rows(protocol_dir, "val")
    validate_partition(train, val, EXPECTED_COUNTS)
    if protocol.get("video_byte_overlap") != 0 or protocol.get("R_D_enabled") is not False:
        raise ValueError("the video-byte seal must pass and R+D must remain disabled")
    records, chains, segments, excluded = build_chains(train + val)
    overlap = None
    if historical_train is not None:
        if sha(historical_train) != OLD_SHA["train"]:
            raise ValueError("historical manifest differs from the sealed former training set")
        old_ids = {row["clip_id"] for row in rows(historical_train)}
        overlap = len(old_ids & {row["clip_id"] for row in val})
    summary = {"schema": SCHEMA, "status": "data_audit_complete_A_features_required",
        "counts": EXPECTED_COUNTS, "manifest_sha256": protocol["manifest_sha256"],
        "protocol_sha256": sha(protocol_dir / "protocol.json"),
        "splits": aggregate(records, chains, segments, excluded),
        "historical_training_clips_in_new_Menzhen_validation": overlap,
        "max_gap_seconds": 5, "source_used_to_enable_connections": False,
        "actions_used_to_construct_connections": False, "R_D_enabled": False,
        "temporal_training_started": False,
        "Menzhen_role": "development_validation_for_checkpoint_and_model_selection",
        "independent_test_evaluated": False, "patient_independence_verified": False,
        "requires_new_A_trained_without_Menzhen": True,
        "warning": "Structural candidates are not verified real continuity or editing boundaries. "
                   "Recording/group independence is not patient independence."}
    private = {"schema": SCHEMA, "chains": chains, "segments": segments,
               "contains_private_clip_identifiers": True, "R_D_enabled": False}
    return summary, private


def write_audit(output, protocol_dir, summary, private):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    for protected in (repo, Path(protocol_dir).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("audit output must be private and outside code/input directories")
    if output.exists():
        raise ValueError("new output required; never overwrite or resume an old audit")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("public_safe_summary.json", summary), ("private_chains.json", private)):
        with (output / name).open("xb") as stream:
            stream.write(encoded(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--historical-train-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true", help="write a new private CPU audit, not training")
    args = parser.parse_args(argv)
    summary, private = audit(args.protocol_dir, args.historical_train_manifest)
    if args.write:
        if args.output is None:
            parser.error("--write requires a new --output outside the repository")
        write_audit(args.output, args.protocol_dir, summary, private)
    print(json.dumps({**summary, "read_only": not args.write}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
