"""Source-rule targets and candidate chains, without claiming human continuity GT.

The user confirms clinical recordings are continuous and network recordings are
not approved for process decoding. C/D below therefore encode this *policy*.
They do not establish whether every network pair is physically discontinuous.
Action labels are copied for later evaluation and never determine a connection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from experiments.aligned_protocol import DATA_PROTOCOL, validate_manifest_protocol
from experiments.pilot_data import _parse_record, _read_jsonl_rows

LABEL_ORIGIN = "source_rule_v1"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _records(records):
    rows, clips, groups, videos, recordings, group_sources = [], set(), {}, {}, {}, {}
    for item in records:
        row = dict(item.to_manifest_dict() if hasattr(item, "to_manifest_dict") else item)
        for key in ("clip_id", "group_id", "source_collection", "video_path"):
            _text(row, key)
        clip = row["clip_id"]
        if clip in clips:
            raise ValueError(f"duplicate clip_id: {clip}")
        clips.add(clip)
        split = row.get("split")
        if not isinstance(split, str) or split not in {"train", "val"}:
            raise ValueError("split must be train or val; test is not read")
        group = row["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError("train/val group leakage")
        groups[group] = split
        if row["video_path"] in videos and videos[row["video_path"]] != split:
            raise ValueError("train/val source video leakage, even under different group IDs")
        videos[row["video_path"]] = split
        record_id = row.get("record_id")
        if record_id is not None:
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("record_id must be nonempty when provided")
            if record_id in recordings and recordings[record_id] != split:
                raise ValueError("train/val record_id leakage")
            recordings[record_id] = split
        row["record_id"] = record_id
        if group in group_sources and group_sources[group] != row["source_collection"]:
            raise ValueError("mixed source_collection within a recording group")
        group_sources[group] = row["source_collection"]
        label = row.get("label_id")
        if isinstance(label, bool) or not isinstance(label, int) or label not in range(7):
            raise ValueError("label_id must be an integer in [0,6]")
        start = _number(row.get("clip_start_sec"), "clip_start_sec")
        end = _number(row.get("clip_end_sec"), "clip_end_sec")
        if start < 0 or end <= start:
            raise ValueError("clip interval must satisfy 0 <= start < end")
        if "clip_duration_sec" in row:
            duration = _number(row["clip_duration_sec"], "clip_duration_sec")
            if not math.isclose(duration, end - start, abs_tol=1e-6, rel_tol=0):
                raise ValueError("clip_duration_sec does not match interval")
        timebase = row.get("timebase_id", "source_video_seconds")
        if not isinstance(timebase, str) or not timebase.strip():
            raise ValueError("timebase_id must be a nonempty string when provided")
        row.update(clip_start_sec=start, clip_end_sec=end,
                   clip_duration_sec=end - start, timebase_id=timebase)
        rows.append(row)
    if not rows:
        raise ValueError("no clips supplied")
    return rows


def source_inventory(records):
    """Counts only; do not expose source paths or per-clip identifiers."""
    rows = _records(records)
    sources = sorted({row["source_collection"] for row in rows})
    return {source: {split: {
        "clips": sum(r["source_collection"] == source and r["split"] == split for r in rows),
        "groups": len({r["group_id"] for r in rows if r["source_collection"] == source and r["split"] == split})
    } for split in ("train", "val")} for source in sources}


def build_source_relations(records, source_map, *, max_gap_seconds=5.0):
    """Return edges, chains, metadata and audit; never modify or resplit inputs.

    ``eligible`` means a structural candidate, including network pairs. It is
    not the C/D target and does not prove clinical continuity. Masks prevent
    crossing overlap, duplicate-start and large-gap ambiguities. No skipped
    clip is used to bridge a cut. Each clip occurs in exactly one chain.
    """
    max_gap = _number(max_gap_seconds, "max_gap_seconds")
    if max_gap < 0:
        raise ValueError("max_gap_seconds must be nonnegative")
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("source_map must be an explicit nonempty object")
    for key, value in source_map.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or value not in {"clinical", "network"}:
            raise ValueError("source_map requires exact nonempty source_collection keys and clinical/network values")
    rows = _records(records)
    unknown = sorted({r["source_collection"] for r in rows} - source_map.keys())
    if unknown:
        raise ValueError(f"unmapped source_collection values: {unknown}")
    buckets = defaultdict(list)
    metadata = []
    for row in rows:
        kind = source_map[row["source_collection"]]
        key = (row["split"], row["group_id"], row["video_path"], row["timebase_id"], row["record_id"] or "")
        buckets[key].append(row)
        metadata.append({k: row[k] for k in ("clip_id", "group_id", "split", "source_collection", "label_id",
                                              "clip_start_sec", "clip_end_sec", "clip_duration_sec")} | {
            "source_kind": kind, "source_video_key": _sha(row["video_path"]),
            "timebase_key": _sha(row["timebase_id"]), "duration": row["clip_duration_sec"],
            "recording_key": _sha(row["record_id"] or row["group_id"]),
        })
    edges, chains, cuts = [], {"train": [], "val": []}, Counter()
    for key in sorted(buckets):
        split, group, video, timebase, record_id = key
        ordered = sorted(buckets[key], key=lambda r: (r["clip_start_sec"], r["clip_end_sec"], r["clip_id"]))
        kind = source_map[ordered[0]["source_collection"]]
        starts = Counter(r["clip_start_sec"] for r in ordered)
        masks, reasons = [], []
        covered_end = ordered[0]["clip_end_sec"]
        for left, right in zip(ordered, ordered[1:]):
            gap = right["clip_start_sec"] - left["clip_end_sec"]
            if starts[left["clip_start_sec"]] > 1 or starts[right["clip_start_sec"]] > 1:
                reason = "duplicate_start"
            elif right["clip_start_sec"] < covered_end:
                reason = "overlap_or_nested_interval"
            elif gap > max_gap:
                reason = "gap_exceeds_cutoff"
            else:
                reason = "ordered_nonoverlap_candidate"
            eligible = reason == "ordered_nonoverlap_candidate"
            masks.append(eligible)
            reasons.append(reason)
            covered_end = max(covered_end, right["clip_end_sec"])
            if not eligible:
                cuts[reason] += 1
                continue
            pair = json.dumps([split, left["clip_id"], right["clip_id"]], ensure_ascii=False, separators=(",", ":"))
            edges.append({
                "edge_id": "edge-" + _sha(pair)[:24], "left_clip_id": left["clip_id"],
                "right_clip_id": right["clip_id"], "split": split,
                "status": "C" if kind == "clinical" else "D", "label_origin": LABEL_ORIGIN,
                "source_kind": kind, "gap_seconds": gap,
                "reason": "continuous_clinical_source_rule" if kind == "clinical" else "source_disallowed_for_flow",
            })
        chain_key = json.dumps([split, group, video, timebase, record_id], ensure_ascii=False, separators=(",", ":"))
        chains[split].append({
            "chain_id": "chain-" + _sha(chain_key)[:24], "group_id": group, "split": split,
            "source_kind": kind, "source_video_key": _sha(video), "timebase_key": _sha(timebase),
            "recording_key": _sha(record_id or group),
            "ordered_clip_ids": [r["clip_id"] for r in ordered], "eligible": masks,
            "candidate_reasons": reasons,
        })
    edge_counts = {split: {status: sum(e["split"] == split and e["status"] == status for e in edges)
                          for status in ("C", "D", "U")} for split in ("train", "val")}
    audit = {
        "label_origin": LABEL_ORIGIN, "label_policy": LABEL_ORIGIN, "source_map": dict(sorted(source_map.items())),
        "unused_source_map_keys": sorted(source_map.keys() - {r["source_collection"] for r in rows}),
        "source_inventory": source_inventory(rows), "edge_counts": edge_counts,
        "clip_counts": {split: sum(r["split"] == split for r in rows) for split in ("train", "val")},
        "chain_counts": {split: len(chains[split]) for split in ("train", "val")},
        "excluded_adjacencies": dict(sorted(cuts.items())), "max_gap_seconds": max_gap,
        "max_gap_status": "engineering_cutoff_not_clinically_verified",
        "actions_used_to_construct_edges": False, "synthetic_negatives": False,
        "ordering_basis": "source_annotation_seconds_not_verified_frame_pts",
        "grouping_basis_counts": dict(Counter(str(r.get("grouping_basis", "not_declared")) for r in rows)),
        "grouping_confidence_counts": dict(Counter(str(r.get("grouping_confidence", "not_declared")) for r in rows)),
        "patient_disjointness_verified": False,
        "warning": "Targets are source-confounded policy supervision, not human-verified continuity ground truth. "
                   "Network D means flow is disallowed by source policy; it does not prove discontinuity. "
                   "Compare learned R against deterministic source gating.",
    }
    return {"edges": edges, "chains": chains, "metadata": sorted(metadata, key=lambda r: r["clip_id"]), "audit": audit}


def load_fixed_manifests(manifest_dir):
    """Formal CLI rejects changed/subset manifests before creating output."""
    root = Path(manifest_dir)
    rows, counts, hashes = [], {}, {}
    for split in ("train", "val"):
        path = root / f"{split}.jsonl"
        split_rows = []
        for _, raw in _read_jsonl_rows(path):
            if "inner_split_role" in raw:
                raise ValueError("full 7372/823 manifests must not use an internal split role")
            _parse_record(raw, expected_split=split, require_video=False)
            if not raw.get("group_id"):
                raise ValueError("explicit recording group_id is required")
            split_rows.append(raw)
        counts[split], hashes[split] = len(split_rows), _file_sha(path)
        rows.extend(split_rows)
    validate_manifest_protocol(counts, hashes)
    _records(rows)
    return rows, hashes


def _write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--inventory", action="store_true", help="print source counts, without constructing targets")
    parser.add_argument("--source-map", help="JSON object: exact source_collection -> clinical/network")
    parser.add_argument("--output", help="new private output directory; existing directories are rejected")
    parser.add_argument("--max-gap-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    rows, hashes = load_fixed_manifests(args.manifest_dir)
    if args.inventory:
        if args.source_map or args.output:
            parser.error("--inventory does not accept --source-map or --output")
        print(json.dumps({"data_protocol": DATA_PROTOCOL, "manifest_sha256": hashes,
                          "source_inventory": source_inventory(rows)}, ensure_ascii=False, indent=2))
        return
    if not args.source_map or not args.output:
        parser.error("build requires --source-map and --output")
    source_map = json.loads(Path(args.source_map).read_text(encoding="utf-8"))
    result = build_source_relations(rows, source_map, max_gap_seconds=args.max_gap_seconds)
    result["audit"].update(data_protocol=DATA_PROTOCOL, manifest_sha256=hashes,
                           counts=result["audit"]["clip_counts"],
                           source_map_sha256=_file_sha(args.source_map))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "edges.jsonl", result["edges"])
    _write_jsonl(output / "manifest_metadata.jsonl", result["metadata"])
    for split in ("train", "val"):
        _write_jsonl(output / f"chains_{split}.jsonl", result["chains"][split])
    result["audit"]["files_sha256"] = {name: _file_sha(output / name) for name in (
        "edges.jsonl", "chains_train.jsonl", "chains_val.jsonl", "manifest_metadata.jsonl")}
    (output / "audit.json").write_text(json.dumps(result["audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "clip_counts": result["audit"]["clip_counts"],
                      "edge_counts": result["audit"]["edge_counts"],
                      "warning": result["audit"]["warning"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
