"""Audit timestamps and derive train7307/Menzhen-dev888; default read-only.

This entry point does NOT invoke the general random-splitting dataset builder.
Private unit evidence is keyed by annotation SHA256, never by guessed magnitude.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from data_tools import build_dataset as parser

SCHEMA = "fsn-menzhen-development-7307-888-v1"
OLD_SHA = {
    "train": "093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3",
    "val": "ad785ec18b14b63616582a143384fac649e69e27d142dfcf87c6852f9d3c6f02",
}
EXPECTED_COUNTS = {"train": 7307, "val": 888}
OLD_COUNTS = {"train": 7372, "val": 823}
TRAIN_SOURCES = {"fsn", "bilibili", "dy", "ks", "youtube", "lishui"}
AUTHORIZED_TERMINAL_CORRECTIONS = 4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, separators=(",", ":")) + "\n").encode()


def resolve(path, data_root):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path(data_root) / p).resolve()


def paired_tsv_unit(text):
    """Infer only by equality to an explicit clock column, never magnitude."""
    candidates = {"seconds", "milliseconds"}
    checked = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        c = line.split("\t")
        if len(c) < 9 or ":" not in c[2] or ":" not in c[4]:
            return None
        try:
            reference = (parser.parse_time(c[2]), parser.parse_time(c[4]))
            for unit in tuple(candidates):
                measured = (parser.parse_time(c[3], unit), parser.parse_time(c[5], unit))
                if max(abs(a - b) for a, b in zip(measured, reference)) > 0.001001:
                    candidates.remove(unit)
        except ValueError:
            return None
        checked += 1
    return next(iter(candidates)) if checked and len(candidates) == 1 else None


def verified_unit(record, text, evidence):
    automatic = paired_tsv_unit(text) if record["input_format"] == "tsv_rows" else None
    item = evidence.get(record["source_txt_sha256"])
    if item is None:
        return automatic, "paired_clock_columns" if automatic else "clock_only_or_missing_evidence"
    if not isinstance(item, dict) or item.get("unit") not in {"seconds", "milliseconds"} \
            or item.get("evidence_type") not in {"verified_elan_export", "data_owner_confirmation"} \
            or not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
        raise ValueError("unit evidence needs explicit unit, evidence_type and explanation")
    if automatic and automatic != item["unit"]:
        raise ValueError("declared unit contradicts paired TSV clocks")
    return item["unit"], item["evidence_type"]


def audit_video(video, historical):
    """Re-probe current bytes; a documented old zero is missing, not equality.

    Positive historical sizes remain strict guards. Only the known no-moov
    zero-size sentinel can be re-sealed, and only after a successful new probe.
    This does not establish identity with an unavailable historical file hash.
    """
    video = Path(video)
    if not video.is_file() or video.stat().st_size <= 0:
        raise ValueError("original video is absent or empty")
    actual_size = video.stat().st_size
    prior_size = historical.get("file_size_bytes")
    missing = prior_size == 0 and historical.get("metadata_reader") == "iso_bmff_fallback_no_moov"
    if not missing and actual_size != prior_size:
        raise ValueError("original video differs from positive historical size")
    actual = parser.parse_video_metadata(video, compute_sha256=False)
    duration = actual.get("duration_sec")
    if actual.get("file_size_bytes") != actual_size or duration is None \
            or not math.isfinite(float(duration)) or float(duration) <= 0 \
            or not all(isinstance(actual.get(k), int) and actual[k] > 0 for k in ("width", "height")):
        raise ValueError("current video duration/geometry could not be verified; no historical fallback")
    return {"video_path": str(video), "historical_size_bytes": prior_size,
            "actual_size_bytes": actual_size, "historical_size_missing": missing,
            "size_check": "missing_no_moov_sentinel_reprobed" if missing else "historical_size_equal",
            "historical_video_sha_available": bool(historical.get("sha256")),
            "historical_video_sha256": historical.get("sha256"),
            "current_metadata": actual}


def capped_terminal_end(start, end, container_duration, decoded_end):
    """Owner-authorized terminal-only corrections, never arbitrary clipping."""
    if not all(math.isfinite(float(v)) for v in (start,end,container_duration,decoded_end)) \
            or start < 0 or end <= start or min(container_duration,decoded_end) <= start:
        raise ValueError("terminal correction has no valid video interval")
    corrected = min(float(container_duration),float(decoded_end))
    if end <= container_duration + .001 or not 0 < end-corrected <= .5:
        raise ValueError("boundary is not one of the authorized small terminal overruns")
    return corrected


def decoded_terminal_end(video):
    import numpy as np
    from decord import VideoReader, cpu
    reader = VideoReader(str(video),num_threads=1,ctx=cpu(0))
    if len(reader) < 1:
        raise ValueError("terminal correction needs a decodable final frame")
    pts=np.asarray(reader.get_frame_timestamp(len(reader)-1),dtype=float).reshape(-1)
    if pts.shape!=(2,) or not np.isfinite(pts).all() or pts[1]<=pts[0]:
        raise ValueError("final-frame PTS is invalid")
    frame=reader[len(reader)-1].asnumpy()
    if frame.ndim!=3 or frame.shape[-1]!=3:
        raise ValueError("final frame could not be decoded as RGB")
    return float(pts[1])


def validate_partition(train, test, expected=EXPECTED_COUNTS):
    if {"train": len(train), "val": len(test)} != expected:
        raise ValueError("measured counts changed; do not force them to 7307/888")
    for name, subset in (("train", train), ("val", test)):
        if len({r["clip_id"] for r in subset}) != len(subset):
            raise ValueError("duplicate clip ID")
        if set(r["label_id"] for r in subset) != set(range(7)):
            raise ValueError("seven-class coverage is incomplete")
        for r in subset:
            source = r["source_collection"].casefold()
            if (name == "val" and source != "menzhen") or \
                    (name == "train" and source not in TRAIN_SOURCES):
                raise ValueError("unknown source or Menzhen contamination in training")
            start, end = float(r["clip_start_sec"]), float(r["clip_end_sec"])
            if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start:
                raise ValueError("invalid clip boundaries")
    for field in ("clip_id", "record_id", "group_id", "video_path"):
        if {r[field] for r in train} & {r[field] for r in test}:
            raise ValueError(f"train/Menzhen validation overlap in {field}")


def prepare(manifest_dir, records_path, events_path, data_root, unit_map=None,
            clip_terminal_overrun=False):
    old = []
    for split, count in OLD_COUNTS.items():
        path = Path(manifest_dir) / f"{split}.jsonl"
        if sha(path) != OLD_SHA[split]:
            raise ValueError("source manifest SHA mismatch")
        subset = rows(path)
        if len(subset) != count or any(r.get("split") != split for r in subset):
            raise ValueError("source manifest count/split mismatch")
        old.extend(subset)
    record_rows, event_rows = rows(records_path), rows(events_path)
    records = {r["record_id"]: r for r in record_rows}
    old_events = {r["event_id"]: r for r in event_rows}
    if len(records) != len(record_rows) or len(old_events) != len(event_rows):
        raise ValueError("duplicate metadata ID")
    evidence = {} if unit_map is None else json.loads(Path(unit_map).read_text())
    if not isinstance(evidence, dict):
        raise ValueError("unit map must be a SHA256-keyed object")
    events, annotation_audit, blockers = {}, [], []
    for record_id in sorted({r["record_id"] for r in old}):
        record = records[record_id]
        path = resolve(record["source_txt_path"], data_root)
        try:
            if sha(path) != record["source_txt_sha256"]:
                raise ValueError("annotation SHA mismatch")
            text = path.read_text(encoding=record["text_encoding"])
            unit, basis = verified_unit(record, text, evidence)
            parsed = parser.parse_events(text, record["input_format"], unit)
            if not parsed:
                raise ValueError("no parsed events")
            events[record_id] = parsed
            annotation_audit.append({"record_id": record_id, "annotation_sha256": sha(path),
                                     "numeric_unit": unit, "basis": basis,
                                     "event_count": len(parsed)})
        except (ValueError, OSError, UnicodeError) as error:
            blockers.append({"record_id": record_id, "annotation_sha256": record.get("source_txt_sha256"),
                             "input_format": record["input_format"], "reason": str(error)})
    if blockers:
        return {"status": "blocked", "files_created": False, "model_evaluation": False,
                "annotations_verified": len(annotation_audit), "blocked_annotations": len(blockers),
                "blocked_formats": dict(Counter(r["input_format"] for r in blockers))}, \
            {"blockers": blockers, "verified": annotation_audit}, None
    derived, changed, corrections = {"train": [], "val": []}, 0, []
    video_audit = {}
    for row in old:
        record = records[row["record_id"]]
        match = re.search(r"-e(\d+)$", row["event_id"])
        if not match:
            raise ValueError("event ID does not identify its original event")
        event = events[row["record_id"]][int(match.group(1))]
        old_event = old_events[row["event_id"]]
        if max(abs(row["clip_start_sec"] - old_event["start_sec"]),
               abs(row["clip_end_sec"] - old_event["end_sec"])) > 1e-6:
            raise ValueError("source clip is not a whole original event; need verified subclip mapping")
        if event["label_id"] != row["label_id"] or event["supervision_status"] != "included":
            raise ValueError("label/eligibility changed; revise protocol rather than force counts")
        start, end = event["start_sec"], event["end_sec"]
        original_annotation_end = end
        # Do not silently expand an old shortened/materialized subclip.
        metadata = record.get("video_metadata") or {}
        video = resolve(row["video_path"], data_root)
        if str(video) not in video_audit:
            video_audit[str(video)] = audit_video(video, metadata)
        elif video_audit[str(video)]["historical_size_bytes"] != metadata.get("file_size_bytes"):
            raise ValueError("contradictory historical metadata for the same video")
        duration = video_audit[str(video)]["current_metadata"]["duration_sec"]
        if end > float(duration) + 0.001:
            if not clip_terminal_overrun:
                raise ValueError("annotation boundary exceeds verified video duration; explicit owner authorization required")
            decoded_end = decoded_terminal_end(video)
            end = capped_terminal_end(start,end,float(duration),decoded_end)
            corrections.append({"clip_id":row["clip_id"],"annotation_sha256":record["source_txt_sha256"],
                "original_annotation_end_sec":original_annotation_end,"derived_end_sec":end,
                "container_duration_sec":duration,"decoded_video_end_sec":decoded_end,
                "removed_terminal_seconds":original_annotation_end-end,
                "basis":"Owner explicitly authorized four terminal-only corrections on 2026-10-10"})
        target = "val" if row["source_collection"].casefold() == "menzhen" else "train"
        difference = max(abs(start - row["clip_start_sec"]), abs(end - row["clip_end_sec"]))
        changed += difference > 1e-6
        updated = {**row, "split": target, "original_split": row["split"],
                   "clip_start_sec": start, "clip_end_sec": end, "clip_duration_sec": end - start,
                   "original_annotation_start_sec":event["start_sec"],
                   "original_annotation_end_sec":original_annotation_end,
                   "terminal_correction_seconds":original_annotation_end-end,
                   "video_path": str(resolve(row["video_path"], data_root)),
                   "derived_protocol": SCHEMA, "time_unit_verified": True,
                   "time_audit_annotation_sha256": record["source_txt_sha256"]}
        derived[target].append(updated)
    if clip_terminal_overrun and len(corrections)!=AUTHORIZED_TERMINAL_CORRECTIONS:
        raise ValueError("authorized four terminal corrections do not match measured count")
    validate_partition(derived["train"], derived["val"],expected=EXPECTED_COUNTS)
    protocol = {"schema": SCHEMA, "counts": EXPECTED_COUNTS, "source_manifest_sha256": OLD_SHA,
                "records_sha256": sha(records_path), "timestamp_audit_passed": True,
                "events_sha256": sha(events_path),
                "unit_map_sha256": None if unit_map is None else sha(unit_map),
                "changed_boundaries": changed, "selection": "best_finetune_Menzhen888_macro_f1",
                "videos_reprobed": len(video_audit),
                "historical_video_sizes_missing": sum(v["historical_size_missing"] for v in video_audit.values()),
                "terminal_boundary_policy": "owner_authorized_four_small_tail_clips" if clip_terminal_overrun else "strict",
                "terminal_corrections":len(corrections),
                "patient_independence_verified": False, "R_D_enabled": False,
                "Menzhen_role": "hospital-held-out development validation used for checkpoint/model selection",
                "independent_test_evaluated": False,
                "historical_Menzhen_development_use": True,
                "old_cache_reuse_allowed": False,
                "class_counts": {k: dict(Counter(r["label_id"] for r in v)) for k, v in derived.items()}}
    return {"status": "verified", "counts": EXPECTED_COUNTS,
            "changed_boundaries": changed, "annotations_verified": len(annotation_audit),
            "videos_reprobed": len(video_audit),
            "historical_video_sizes_missing": protocol["historical_video_sizes_missing"],
            "terminal_corrections":len(corrections),
            "files_created": False, "model_evaluation": False}, \
        {"annotations": annotation_audit, "videos": list(video_audit.values()),
         "terminal_boundary_corrections":corrections,"protocol": protocol}, derived


def write_prepared(output, manifest_dir, records_path, detail, derived):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    for protected in (repo, Path(manifest_dir).resolve(), Path(records_path).resolve().parent):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("private output must be outside source/metadata")
    if output.exists() or derived is None:
        raise ValueError("output exists or timestamp audit is blocked")
    # Complete input and byte-duplicate audit before creating any output.
    video_hashes = {}
    for subset in derived.values():
        for row in subset:
            p = row["video_path"]
            if p not in video_hashes:
                video_hashes[p] = sha(p)
                recorded = next(v for v in detail["videos"] if v["video_path"] == p)
                historical_sha = recorded.get("historical_video_sha256")
                # Current probe intentionally does not hash; validate historical
                # bytes only when a historical checksum was actually recorded.
                if historical_sha and video_hashes[p] != historical_sha:
                    raise ValueError("video differs from recorded historical SHA")
                if len(video_hashes) % 100 == 0:
                    print(json.dumps({"stage": "raw_video_SHA", "verified_videos": len(video_hashes)}), flush=True)
    if {video_hashes[r["video_path"]] for r in derived["train"]} & \
            {video_hashes[r["video_path"]] for r in derived["val"]}:
        raise ValueError("byte-identical video crosses hospital holdout")
    payloads = {f"{split}.jsonl": b"".join(encoded(r) for r in subset)
                for split, subset in derived.items()}
    protocol = {**detail["protocol"], "manifest_sha256":
                {k: hashlib.sha256(payloads[f"{k}.jsonl"]).hexdigest() for k in derived},
                "video_byte_overlap": 0, "video_sha256": video_hashes,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=repo, text=True).strip()}
    output.mkdir(parents=True, exist_ok=False)
    for filename, payload in {**payloads, "private_time_audit.json": encoded(detail),
                              "protocol.json": encoded(protocol)}.items():
        with (output / filename).open("xb") as stream:
            stream.write(payload)
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest-dir", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--unit-map", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write", action="store_true")
    p.add_argument("--clip-terminal-overrun",action="store_true",
                   help="owner-authorized four small terminal-only corrections; retain originals")
    args = p.parse_args()
    summary, detail, derived = prepare(args.source_manifest_dir, args.records, args.events,
                                        args.data_root, args.unit_map,args.clip_terminal_overrun)
    if args.write:
        write_prepared(args.output, args.source_manifest_dir, args.records, detail, derived)
        summary["files_created"] = True
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
