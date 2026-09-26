#!/usr/bin/env python3
"""Normalize FSN TXT annotations and build leakage-aware train/test manifests.

The script never changes source data.  It preserves provenance and questionable
annotations in meta_data, while only clean, canonical events enter clip manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable


SCHEMA_VERSION = "fsn-normalized-2.0"
SPLIT_POLICY_VERSION = "grouped-source-stratified-8-1-1-v2.0"
CANONICAL_LABELS = ["消毒", "进针", "运针", "扫散", "再灌注", "拔针", "固定"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
LABEL_MAP = {
    "消毒": "消毒",
    "进针": "进针",
    "j进针": "进针",
    "运针": "运针",
    "扫散": "扫散",
    "扫撒": "扫散",
    "扫散 扫散": "扫散",
    "灌注": "再灌注",
    "再灌注": "再灌注",
    "肌肉再灌注": "再灌注",
    "拔针": "拔针",
    "固定": "固定",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
TIME_TOKEN = r"(?:\d{1,3}:\d{2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)"
RANGE_RE = re.compile(rf"(?<![\d:.])({TIME_TOKEN})\s*-\s*({TIME_TOKEN})(?![\d:.])")
DATE_RE = re.compile(r"^\d{4} [A-Z][a-z]{2} \d{1,2}, [A-Z][a-z]{2} \d{2}:\d{2}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def normalized_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def media_key_from_txt(path: Path) -> str:
    name = path.name[:-4] if path.name.lower().endswith(".txt") else path.name
    suffix = Path(name).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        name = name[: -len(suffix)]
    return name


def parse_time(value: str) -> float:
    value = value.strip()
    if ":" not in value:
        return float(value)
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError(f"unsupported time token: {value}")
    return int(fields[0]) * 3600 + int(fields[1]) * 60 + float(fields[2])


def clean_label(raw_label: str | None) -> tuple[str | None, list[str]]:
    if raw_label is None:
        return None, ["unlabeled_interval"]
    label = unicodedata.normalize("NFKC", raw_label).strip()
    label = re.sub(r"\s+", " ", label)
    if not label:
        return None, ["unlabeled_interval"]
    normalized = LABEL_MAP.get(label)
    if normalized is None:
        return None, ["noncanonical_or_ambiguous_label"]
    return normalized, []


def extract_header_metadata(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    uri = next((line.strip() for line in lines[:8] if line.strip().startswith("file:")), None)
    raw_date = next((line.strip() for line in lines[:8] if DATE_RE.fullmatch(line.strip())), None)
    parsed_date = None
    if raw_date:
        try:
            parsed_date = datetime.strptime(raw_date, "%Y %b %d, %a %H:%M").isoformat()
        except ValueError:
            pass
    return {
        "original_media_uri": uri,
        "annotation_created_at_raw": raw_date,
        "annotation_created_at": parsed_date,
    }


def detect_format(text: str) -> str:
    if not text.strip():
        return "empty"
    if re.search(r"^default\t", text, re.MULTILINE):
        return "tsv_rows"
    if re.search(r"^default\s+", text, re.MULTILINE) and re.search(r"^TC\s+", text, re.MULTILINE):
        return "elan_wide"
    if any(RANGE_RE.fullmatch(line.strip()) and ":" in line for line in text.splitlines()):
        return "label_time_blocks"
    if extract_header_metadata(text)["original_media_uri"]:
        return "metadata_only"
    return "unknown"


def base_event(index: int, start: float, end: float, raw_label: str | None, line_number: int) -> dict[str, Any]:
    label, reasons = clean_label(raw_label)
    if start < 0:
        reasons.append("negative_start")
    if end <= start:
        reasons.append("nonpositive_duration")
    duration = end - start
    if 0 < duration < 0.1:
        reasons.append("duration_lt_0.1s")
    reasons = sorted(set(reasons))
    return {
        "event_index": index,
        "start_sec": round(start, 6),
        "end_sec": round(end, 6),
        "duration_sec": round(duration, 6),
        "raw_label": raw_label,
        "normalized_label": label,
        "label_id": LABEL_TO_ID.get(label) if label else None,
        "source_line_number": line_number,
        "supervision_status": "excluded" if reasons else "included",
        "exclusion_reasons": reasons,
        "qc_flags": [],
    }


def parse_tsv_rows(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < 9:
            event = base_event(len(events), 0.0, 0.0, None, line_number)
            event["exclusion_reasons"].append("malformed_tsv_row")
            events.append(event)
            continue
        try:
            start = float(columns[3].strip())
            end = float(columns[5].strip())
        except ValueError:
            start = parse_time(columns[2])
            end = parse_time(columns[4])
        events.append(base_event(len(events), start, end, columns[-1], line_number))
    return events


def parse_wide(text: str) -> list[dict[str, Any]]:
    """Parse ELAN wide text by column position, preserving blank labels.

    Each label begins at the same character offset as its matching time range.
    Whitespace splitting is unsafe because some files contain blank annotations.
    """
    lines = text.splitlines()
    events: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not re.match(r"^default\s+", line):
            continue
        tc_idx = idx + 1
        while tc_idx < len(lines) and not re.match(r"^TC\s+", lines[tc_idx]):
            tc_idx += 1
        if tc_idx >= len(lines):
            continue
        label_positions: dict[int, str] = {}
        # Multiple spaces delimit cells; a single space may be part of an entered label.
        for match in re.finditer(r"\S+(?:[ \u3000]\S+)*", line):
            if match.group() != "default":
                label_positions[match.start()] = match.group().strip()
        range_matches = list(RANGE_RE.finditer(lines[tc_idx]))
        integer_millisecond_row = bool(range_matches) and all(
            ":" not in match.group(1)
            and ":" not in match.group(2)
            and "." not in match.group(1)
            and "." not in match.group(2)
            for match in range_matches
        ) and max(float(match.group(2)) for match in range_matches) >= 1000
        for match in range_matches:
            raw_label = label_positions.get(match.start())
            start = parse_time(match.group(1))
            end = parse_time(match.group(2))
            if integer_millisecond_row:
                start /= 1000
                end /= 1000
            event = base_event(
                len(events), start, end, raw_label, tc_idx + 1
            )
            if integer_millisecond_row:
                event["qc_flags"].append("integer_millisecond_timebase_normalized_to_seconds")
            if raw_label is None:
                event["qc_flags"].append("blank_label_cell_preserved_by_column_alignment")
            events.append(event)
    return events


def parse_blocks(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    events: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        match = RANGE_RE.fullmatch(line.strip())
        if not match or ":" not in line:
            continue
        label_idx = idx - 1
        while label_idx >= 0 and not lines[label_idx].strip():
            label_idx -= 1
        raw_label = lines[label_idx].strip() if label_idx >= 0 else None
        events.append(
            base_event(len(events), parse_time(match.group(1)), parse_time(match.group(2)), raw_label, idx + 1)
        )
    return events


def parse_events(text: str, input_format: str) -> list[dict[str, Any]]:
    if input_format == "tsv_rows":
        return parse_tsv_rows(text)
    if input_format == "elan_wide":
        return parse_wide(text)
    if input_format == "label_time_blocks":
        return parse_blocks(text)
    return []


def iter_atoms(handle: BinaryIO, start: int, end: int) -> Iterable[tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= end:
        handle.seek(pos)
        header = handle.read(8)
        if len(header) != 8:
            break
        size32, atom_type = struct.unpack(">I4s", header)
        header_size = 8
        if size32 == 1:
            ext = handle.read(8)
            if len(ext) != 8:
                break
            size = struct.unpack(">Q", ext)[0]
            header_size = 16
        elif size32 == 0:
            size = end - pos
        else:
            size = size32
        if size < header_size or pos + size > end:
            break
        yield atom_type, pos + header_size, size - header_size
        pos += size


def atom_children(handle: BinaryIO, start: int, size: int) -> list[tuple[bytes, int, int]]:
    return list(iter_atoms(handle, start, start + size))


def atom_payload(handle: BinaryIO, start: int, size: int, limit: int = 2_000_000) -> bytes:
    handle.seek(start)
    return handle.read(min(size, limit))


def parse_mvhd(payload: bytes) -> float | None:
    if len(payload) < 24:
        return None
    version = payload[0]
    if version == 1 and len(payload) >= 32:
        timescale = struct.unpack(">I", payload[20:24])[0]
        duration = struct.unpack(">Q", payload[24:32])[0]
    else:
        timescale = struct.unpack(">I", payload[12:16])[0]
        duration = struct.unpack(">I", payload[16:20])[0]
    return duration / timescale if timescale else None


def parse_mdhd(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 24:
        return None, None
    if payload[0] == 1 and len(payload) >= 32:
        return struct.unpack(">I", payload[20:24])[0], struct.unpack(">Q", payload[24:32])[0]
    return struct.unpack(">I", payload[12:16])[0], struct.unpack(">I", payload[16:20])[0]


def parse_tkhd(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 84:
        return None, None
    width = struct.unpack(">I", payload[-8:-4])[0] / 65536
    height = struct.unpack(">I", payload[-4:])[0] / 65536
    matrix = payload[-44:-8]
    if len(matrix) == 36:
        a, b, c, d = (struct.unpack(">i", matrix[x : x + 4])[0] / 65536 for x in (0, 4, 12, 16))
        if abs(a) < 0.01 and abs(d) < 0.01 and abs(b) > 0.9 and abs(c) > 0.9:
            width, height = height, width
    return round(width) or None, round(height) or None


def parse_video_metadata(path: Path, compute_sha256: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "container_extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "sha256": None,
        "duration_sec": None,
        "width": None,
        "height": None,
        "codec": None,
        "frame_count": None,
        "average_fps": None,
        "metadata_reader": "iso_bmff_fallback",
    }
    if compute_sha256:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
        result["sha256"] = hasher.hexdigest()
    try:
        with path.open("rb") as handle:
            top = atom_children(handle, 0, path.stat().st_size)
            moov = next((atom for atom in top if atom[0] == b"moov"), None)
            if not moov:
                result["metadata_reader"] = "iso_bmff_fallback_no_moov"
                return result
            moov_children = atom_children(handle, moov[1], moov[2])
            mvhd = next((atom for atom in moov_children if atom[0] == b"mvhd"), None)
            if mvhd:
                result["duration_sec"] = parse_mvhd(atom_payload(handle, mvhd[1], mvhd[2]))
            for trak in (atom for atom in moov_children if atom[0] == b"trak"):
                children = atom_children(handle, trak[1], trak[2])
                tkhd = next((atom for atom in children if atom[0] == b"tkhd"), None)
                mdia = next((atom for atom in children if atom[0] == b"mdia"), None)
                if not mdia:
                    continue
                mdia_children = atom_children(handle, mdia[1], mdia[2])
                hdlr = next((atom for atom in mdia_children if atom[0] == b"hdlr"), None)
                hdlr_payload = atom_payload(handle, hdlr[1], hdlr[2]) if hdlr else b""
                if len(hdlr_payload) < 12 or hdlr_payload[8:12] != b"vide":
                    continue
                if tkhd:
                    result["width"], result["height"] = parse_tkhd(atom_payload(handle, tkhd[1], tkhd[2]))
                mdhd = next((atom for atom in mdia_children if atom[0] == b"mdhd"), None)
                timescale, track_duration = (None, None)
                if mdhd:
                    timescale, track_duration = parse_mdhd(atom_payload(handle, mdhd[1], mdhd[2]))
                minf = next((atom for atom in mdia_children if atom[0] == b"minf"), None)
                if not minf:
                    break
                stbl = next((atom for atom in atom_children(handle, minf[1], minf[2]) if atom[0] == b"stbl"), None)
                if not stbl:
                    break
                stbl_children = atom_children(handle, stbl[1], stbl[2])
                stsd = next((atom for atom in stbl_children if atom[0] == b"stsd"), None)
                if stsd:
                    payload = atom_payload(handle, stsd[1], stsd[2], 64)
                    if len(payload) >= 16:
                        result["codec"] = payload[12:16].decode("ascii", errors="replace")
                stts = next((atom for atom in stbl_children if atom[0] == b"stts"), None)
                if stts:
                    payload = atom_payload(handle, stts[1], stts[2])
                    if len(payload) >= 8:
                        entries = struct.unpack(">I", payload[4:8])[0]
                        total_samples = 0
                        total_ticks = 0
                        for offset in range(8, min(len(payload), 8 + entries * 8), 8):
                            count, delta = struct.unpack(">II", payload[offset : offset + 8])
                            total_samples += count
                            total_ticks += count * delta
                        result["frame_count"] = total_samples or None
                        seconds = total_ticks / timescale if timescale and total_ticks else None
                        if seconds:
                            result["average_fps"] = round(total_samples / seconds, 6)
                if result["duration_sec"] is None and timescale and track_duration:
                    result["duration_sec"] = track_duration / timescale
                break
    except (OSError, StopIteration, struct.error, ValueError) as exc:
        result["metadata_reader"] = f"iso_bmff_fallback_error:{type(exc).__name__}"
    if result["duration_sec"] is not None:
        result["duration_sec"] = round(float(result["duration_sec"]), 6)
    return result


def infer_group(source: str, media_key: str) -> dict[str, str]:
    acu = re.search(r"(?i)ACU\s*0*(\d+)", media_key)
    if acu:
        return {
            "group_id": f"{source}:subject:ACU{int(acu.group(1)):03d}",
            "basis": "filename_patient_code",
            "confidence": "high",
        }
    return {
        "group_id": f"{source}:record:{stable_hash(normalized_key(media_key), 20)}",
        "basis": "recording_id_only_no_subject_id",
        "confidence": "low",
    }


def infer_filename_metadata(media_key: str) -> dict[str, str | None]:
    date = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", media_key)
    return {"recording_date_from_filename": "-".join(date.groups()) if date else None}


def build_video_index(video_root: Path) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    if not video_root.exists():
        return index
    for path in sorted(
        p for p in video_root.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != ".DS_Store"
        and p.suffix.lower() in VIDEO_EXTENSIONS
    ):
        source = path.parent.name.removesuffix("_video")
        index[(source, normalized_key(path.stem))].append(path)
    return index


def subset_near_target(group_sizes: dict[str, int], target: int, seed_scope: str) -> set[str]:
    """Deterministically choose whole groups with total size nearest target."""
    groups = sorted(group_sizes, key=lambda group: stable_hash(f"{seed_scope}|{group}", 64))
    total = sum(group_sizes.values())
    if not groups or target <= 0:
        return set()
    if len(groups) == 1:
        return set()
    target = max(1, min(total - 1, target))
    reachable = [False] * (total + 1)
    previous_sum = [-1] * (total + 1)
    previous_group = [-1] * (total + 1)
    reachable[0] = True
    for group_idx, group in enumerate(groups):
        size = group_sizes[group]
        for subtotal in range(total - size, -1, -1):
            new_sum = subtotal + size
            if reachable[subtotal] and not reachable[new_sum]:
                reachable[new_sum] = True
                previous_sum[new_sum] = subtotal
                previous_group[new_sum] = group_idx
    candidates = [value for value in range(1, total) if reachable[value]]
    if not candidates:
        return set()
    chosen_sum = min(candidates, key=lambda value: (abs(value - target), value > target, value))
    selected: set[str] = set()
    while chosen_sum:
        group_idx = previous_group[chosen_sum]
        selected.add(groups[group_idx])
        chosen_sum = previous_sum[chosen_sum]
    return selected


def choose_eval_groups(
    records: list[dict[str, Any]], val_ratio: float, test_ratio: float, seed: str
) -> tuple[set[str], set[str]]:
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record["eligibility_status"] == "included":
            by_source[record["source_collection"]][record["grouping"]["group_id"]] += record[
                "event_count_supervised"
            ]
    selected_val: set[str] = set()
    selected_test: set[str] = set()
    for source, group_sizes in sorted(by_source.items()):
        total = sum(group_sizes.values())
        if len(group_sizes) < 3 or total < 3:
            continue
        test_groups = subset_near_target(
            dict(group_sizes), round(total * test_ratio), f"{seed}|{source}|test"
        )
        remaining = {group: size for group, size in group_sizes.items() if group not in test_groups}
        val_groups = subset_near_target(
            remaining, round(total * val_ratio), f"{seed}|{source}|val"
        )
        selected_test.update(test_groups)
        selected_val.update(val_groups)
    return selected_val, selected_test


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt-root", type=Path, default=Path("data/txt"))
    parser.add_argument("--video-root", type=Path, default=Path("data/video"))
    parser.add_argument("--output", type=Path, default=Path("processed_dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", default="fsn-reviewer-split-v2")
    parser.add_argument(
        "--skip-video-sha256", action="store_true",
        help="read container metadata without hashing every full video (faster for development)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing generated output directory")
    args = parser.parse_args()
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("--val-ratio and --test-ratio must be positive and sum to less than 1")
    workspace = Path.cwd()
    txt_root = args.txt_root.resolve()
    video_root = args.video_root.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {output}; use --force to replace it")
        shutil.rmtree(output)
    (output / "meta_data").mkdir(parents=True)
    (output / "annotations").mkdir()
    (output / "manifests").mkdir()

    video_index = build_video_index(video_root)
    video_cache: dict[Path, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []

    txt_paths = sorted((
        path for path in txt_root.rglob("*.txt")
        if not path.name.startswith("._") and path.name != ".DS_Store"
    ), key=lambda p: p.as_posix())
    raw_by_path = {path: path.read_bytes() for path in txt_paths}
    paths_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path, raw in raw_by_path.items():
        paths_by_hash[sha256_bytes(raw)].append(path)

    def duplicate_preference(path: Path) -> tuple[int, int, str]:
        source_dir = path.relative_to(txt_root).parts[0]
        source = source_dir.removesuffix("_txt")
        media_key = media_key_from_txt(path)
        candidates = video_index.get((source, normalized_key(media_key)), [])
        # Prefer the duplicate whose filename uniquely resolves to an existing
        # video. This prevents renamed upload copies from becoming canonical.
        return (0 if len(candidates) == 1 else 1, len(path.relative_to(txt_root).parts), path.as_posix())

    canonical_by_hash = {
        content_hash: min(paths, key=duplicate_preference)
        for content_hash, paths in paths_by_hash.items()
    }
    canonical_record_by_hash: dict[str, str] = {}
    for content_hash, canonical_path in canonical_by_hash.items():
        source = canonical_path.relative_to(txt_root).parts[0].removesuffix("_txt")
        canonical_record_by_hash[content_hash] = f"{source}-{stable_hash(relpath(canonical_path, workspace), 16)}"

    for txt_path in txt_paths:
        raw = raw_by_path[txt_path]
        try:
            text = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
            encoding = "gb18030"
        source_dir = txt_path.relative_to(txt_root).parts[0]
        source = source_dir.removesuffix("_txt")
        media_key = media_key_from_txt(txt_path)
        source_rel = relpath(txt_path, workspace)
        record_id = f"{source}-{stable_hash(source_rel, 16)}"
        input_format = detect_format(text)
        events = parse_events(text, input_format)
        metadata = extract_header_metadata(text)
        content_hash = sha256_bytes(raw)
        duplicate_of = (
            None if txt_path == canonical_by_hash[content_hash]
            else canonical_record_by_hash[content_hash]
        )
        candidates = video_index.get((source, normalized_key(media_key)), [])
        video_path = candidates[0] if len(candidates) == 1 else None
        video_metadata = None
        if video_path:
            if video_path not in video_cache:
                video_cache[video_path] = parse_video_metadata(
                    video_path, compute_sha256=not args.skip_video_sha256
                )
            video_metadata = video_cache[video_path]
        record_qc: list[str] = []
        if input_format in {"empty", "metadata_only", "unknown"}:
            record_qc.append(input_format)
        if duplicate_of:
            record_qc.append("exact_duplicate_txt")
        if len(candidates) > 1:
            record_qc.append("ambiguous_video_match")
        elif not candidates:
            record_qc.append("video_missing_from_current_snapshot")
        for event in events:
            event_id = f"{record_id}-e{event['event_index']:04d}"
            event.update({"schema_version": SCHEMA_VERSION, "event_id": event_id, "record_id": record_id})
            if video_metadata and video_metadata.get("duration_sec") is not None:
                if event["end_sec"] > video_metadata["duration_sec"] + 0.5:
                    event["qc_flags"].append("end_beyond_available_video_duration")
                    event["exclusion_reasons"].append("end_beyond_available_video_duration")
                    event["supervision_status"] = "excluded"
            all_events.append(event)
        included_events = [event for event in events if event["supervision_status"] == "included"]
        if not events:
            eligibility = "excluded"
            exclusion_reason = "no_parsed_events"
        elif duplicate_of:
            eligibility = "excluded"
            exclusion_reason = "exact_duplicate_txt"
        elif not included_events:
            eligibility = "excluded"
            exclusion_reason = "no_supervised_events_after_qc"
        else:
            eligibility = "included"
            exclusion_reason = None
        raw_labels = Counter(event["raw_label"] or "__BLANK__" for event in events)
        normalized_labels = Counter(
            event["normalized_label"] for event in events if event["normalized_label"] is not None
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "source_collection": source,
            "source_txt_path": source_rel,
            "source_txt_sha256": content_hash,
            "source_txt_size_bytes": len(raw),
            "text_encoding": encoding,
            "input_format": input_format,
            **metadata,
            "media_key": media_key,
            "expected_video_filenames": [media_key + ext for ext in sorted(VIDEO_EXTENSIONS)],
            "video_link_status": "matched" if video_path else ("ambiguous" if candidates else "missing"),
            "available_video_path": relpath(video_path, workspace) if video_path else None,
            "video_metadata": video_metadata,
            "duplicate_of_record_id": duplicate_of,
            "grouping": infer_group(source, media_key),
            "filename_metadata": infer_filename_metadata(media_key),
            "identity_metadata": {
                "patient_id": None,
                "practitioner_id": None,
                "treatment_session_id": None,
                "institution_id": None,
                "status": "not_provided",
            },
            "eligibility_status": eligibility,
            "eligibility_exclusion_reason": exclusion_reason,
            "split": None,
            "split_policy": None,
            "event_count_total": len(events),
            "event_count_supervised": len(included_events),
            "event_count_excluded": len(events) - len(included_events),
            "max_annotation_end_sec": max((event["end_sec"] for event in events), default=None),
            "raw_label_counts": dict(sorted(raw_labels.items())),
            "normalized_label_counts": dict(sorted(normalized_labels.items())),
            "qc_flags": sorted(set(record_qc + [flag for event in events for flag in event["qc_flags"]])),
        }
        records.append(record)

    selected_val_groups, selected_test_groups = choose_eval_groups(
        records, args.val_ratio, args.test_ratio, args.seed
    )
    split_by_record: dict[str, str] = {}
    for record in records:
        if record["eligibility_status"] != "included":
            split = "excluded"
        elif record["grouping"]["group_id"] in selected_test_groups:
            split = "test"
        elif record["grouping"]["group_id"] in selected_val_groups:
            split = "val"
        else:
            split = "train"
        record["split"] = split
        record["split_policy"] = {
            "version": SPLIT_POLICY_VERSION,
            "seed": args.seed,
            "requested_train_ratio": 1 - args.val_ratio - args.test_ratio,
            "requested_val_ratio": args.val_ratio,
            "requested_test_ratio": args.test_ratio,
            "ratio_basis": "supervised_clip_count",
            "unit": "group_id",
            "source_stratified": True,
            "clip_inherits_record_split": True,
            "identity_independence_status": "provisional_until_identity_map_is_completed",
        }
        split_by_record[record["record_id"]] = split
    for event in all_events:
        event["split"] = split_by_record[event["record_id"]]

    clips: list[dict[str, Any]] = []
    record_by_id = {record["record_id"]: record for record in records}
    for event in all_events:
        record = record_by_id[event["record_id"]]
        if record["split"] not in {"train", "val", "test"} or event["supervision_status"] != "included":
            continue
        label = event["normalized_label"]
        clip_id = f"clip-{stable_hash(event['event_id'] + '|' + str(event['start_sec']) + '|' + str(event['end_sec']), 20)}"
        clips.append({
            "schema_version": SCHEMA_VERSION,
            "clip_id": clip_id,
            "record_id": event["record_id"],
            "event_id": event["event_id"],
            "split": record["split"],
            "source_collection": record["source_collection"],
            "group_id": record["grouping"]["group_id"],
            "grouping_basis": record["grouping"]["basis"],
            "grouping_confidence": record["grouping"]["confidence"],
            "normalized_label": label,
            "label_id": event["label_id"],
            "raw_label": event["raw_label"],
            "clip_start_sec": event["start_sec"],
            "clip_end_sec": event["end_sec"],
            "clip_duration_sec": event["duration_sec"],
            "video_path": record["available_video_path"],
            "materialization_status": "ready_to_materialize" if record["available_video_path"] else "waiting_for_video",
            "planned_clip_path": f"clips/{record['split']}/{LABEL_TO_ID[label]:02d}_{label}/{clip_id}.mp4",
        })

    write_jsonl(output / "meta_data" / "records.jsonl", records)
    write_jsonl(output / "meta_data" / "events_all.jsonl", all_events)
    write_jsonl(output / "manifests" / "clips_all.jsonl", clips)
    write_jsonl(output / "manifests" / "train.jsonl", (clip for clip in clips if clip["split"] == "train"))
    write_jsonl(output / "manifests" / "val.jsonl", (clip for clip in clips if clip["split"] == "val"))
    write_jsonl(output / "manifests" / "test.jsonl", (clip for clip in clips if clip["split"] == "test"))
    write_jsonl(output / "manifests" / "excluded_records.jsonl", (r for r in records if r["split"] == "excluded"))

    identity_fields = [
        "record_id", "source_collection", "source_txt_path", "media_key", "patient_id",
        "practitioner_id", "treatment_session_id", "institution_id", "acquisition_domain", "notes",
    ]
    with (output / "meta_data" / "identity_map_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=identity_fields)
        writer.writeheader()
        for record in records:
            source = record["source_collection"]
            if source in {"lishui", "menzhen"}:
                domain = "clinical_recording"
            elif source in {"bilibili", "dy", "ks", "youtube"}:
                domain = "public_online_video"
            else:
                domain = "unknown_requires_confirmation"
            writer.writerow({
                "record_id": record["record_id"],
                "source_collection": source,
                "source_txt_path": record["source_txt_path"],
                "media_key": record["media_key"],
                "patient_id": "",
                "practitioner_id": "",
                "treatment_session_id": "",
                "institution_id": "",
                "acquisition_domain": domain,
                "notes": "",
            })

    source_holdout_root = output / "manifests" / "source_holdout"
    source_holdout_root.mkdir()
    source_holdout_summary: dict[str, dict[str, int]] = {}
    for held_source in sorted({clip["source_collection"] for clip in clips}):
        fold_root = source_holdout_root / held_source
        fold_root.mkdir()
        external_test = [clip for clip in clips if clip["source_collection"] == held_source]
        development_val = [
            clip for clip in clips
            if clip["source_collection"] != held_source and clip["split"] == "val"
        ]
        development_train = [
            clip for clip in clips
            if clip["source_collection"] != held_source and clip["split"] in {"train", "test"}
        ]
        write_jsonl(fold_root / "development_train.jsonl", development_train)
        write_jsonl(fold_root / "development_val.jsonl", development_val)
        write_jsonl(fold_root / "external_test.jsonl", external_test)
        source_holdout_summary[held_source] = {
            "development_train_clips": len(development_train),
            "development_val_clips": len(development_val),
            "external_test_clips": len(external_test),
        }
    json_dump(output / "meta_data" / "source_holdout_summary.json", {
        "purpose": "Cross-source domain-shift evaluation; this does not replace a truly external cohort.",
        "folds": source_holdout_summary,
    })

    for record in records:
        record_events = [event for event in all_events if event["record_id"] == record["record_id"]]
        json_dump(output / "annotations" / f"{record['record_id']}.json", {"record": record, "events": record_events})

    csv_fields = [
        "schema_version", "clip_id", "record_id", "event_id", "split", "source_collection", "group_id",
        "grouping_basis", "grouping_confidence", "normalized_label", "label_id", "raw_label",
        "clip_start_sec", "clip_end_sec", "clip_duration_sec", "video_path",
        "materialization_status", "planned_clip_path",
    ]
    with (output / "manifests" / "clips_all.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(clips)

    label_policy = {
        "schema_version": SCHEMA_VERSION,
        "canonical_labels_in_id_order": CANONICAL_LABELS,
        "raw_to_normalized": LABEL_MAP,
        "excluded_label_policy": "Unmapped, blank, or ambiguous labels remain in events_all.jsonl but not clip manifests.",
        "minimum_supervised_clip_duration_sec": 0.1,
    }
    json_dump(output / "meta_data" / "label_policy.json", label_policy)

    with (output / "meta_data" / "class_support_by_split.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "split", "label_id", "label", "clip_count", "total_duration_sec",
            "mean_duration_sec", "min_duration_sec", "max_duration_sec",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "val", "test"):
            for label in CANONICAL_LABELS:
                selected = [
                    clip for clip in clips
                    if clip["split"] == split and clip["normalized_label"] == label
                ]
                durations = [clip["clip_duration_sec"] for clip in selected]
                writer.writerow({
                    "split": split,
                    "label_id": LABEL_TO_ID[label],
                    "label": label,
                    "clip_count": len(selected),
                    "total_duration_sec": round(sum(durations), 6),
                    "mean_duration_sec": round(sum(durations) / len(durations), 6) if durations else "",
                    "min_duration_sec": min(durations) if durations else "",
                    "max_duration_sec": max(durations) if durations else "",
                })

    with (output / "meta_data" / "source_support_by_split.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = ["split", "source_collection", "record_count", "group_count", "clip_count", "duration_sec"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "val", "test"):
            for source in sorted({record["source_collection"] for record in records}):
                selected_records = [
                    record for record in records
                    if record["split"] == split and record["source_collection"] == source
                ]
                selected_clips = [
                    clip for clip in clips
                    if clip["split"] == split and clip["source_collection"] == source
                ]
                writer.writerow({
                    "split": split,
                    "source_collection": source,
                    "record_count": len(selected_records),
                    "group_count": len({record["grouping"]["group_id"] for record in selected_records}),
                    "clip_count": len(selected_clips),
                    "duration_sec": round(sum(clip["clip_duration_sec"] for clip in selected_clips), 6),
                })

    def nested_counts(rows: list[dict[str, Any]], first: str, second: str) -> dict[str, dict[str, int]]:
        values: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            values[str(row[first])][str(row[second])] += 1
        return {key: dict(sorted(value.items())) for key, value in sorted(values.items())}

    eligible_records = [record for record in records if record["split"] in {"train", "val", "test"}]
    evaluation_splits = ("train", "val", "test")
    group_memberships: dict[str, set[str]] = defaultdict(set)
    txt_hash_memberships: dict[str, set[str]] = defaultdict(set)
    video_hash_memberships: dict[str, set[str]] = defaultdict(set)
    for record in eligible_records:
        group_memberships[record["grouping"]["group_id"]].add(record["split"])
        txt_hash_memberships[record["source_txt_sha256"]].add(record["split"])
        if record["video_metadata"] and record["video_metadata"].get("sha256"):
            video_hash_memberships[record["video_metadata"]["sha256"]].add(record["split"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "input": {
            "txt_root": relpath(txt_root, workspace),
            "video_root": relpath(video_root, workspace),
            "raw_txt_files": len(records),
            "available_video_files": sum(1 for paths in video_index.values() for _ in paths),
        },
        "formats": dict(sorted(Counter(record["input_format"] for record in records).items())),
        "records_by_split": dict(sorted(Counter(record["split"] for record in records).items())),
        "records_by_source_and_split": nested_counts(records, "source_collection", "split"),
        "groups_by_split": {
            split: len({r["grouping"]["group_id"] for r in records if r["split"] == split})
            for split in (*evaluation_splits, "excluded")
        },
        "events": {
            "raw_total": len(all_events),
            "supervised_clip_total": len(clips),
            "excluded_from_supervision": len(all_events) - len(clips),
            "clips_by_split": dict(sorted(Counter(clip["split"] for clip in clips).items())),
            "clips_by_label_and_split": nested_counts(clips, "normalized_label", "split"),
            "annotated_duration_hours_by_split": {
                split: round(sum(c["clip_duration_sec"] for c in clips if c["split"] == split) / 3600, 6)
                for split in evaluation_splits
            },
        },
        "identity_metadata_completeness": {
            "eligible_records": len(eligible_records),
            "records_with_patient_id": sum(bool(r["identity_metadata"]["patient_id"]) for r in eligible_records),
            "records_with_practitioner_id": sum(bool(r["identity_metadata"]["practitioner_id"]) for r in eligible_records),
            "records_with_treatment_session_id": sum(
                bool(r["identity_metadata"]["treatment_session_id"]) for r in eligible_records
            ),
            "strict_patient_and_practitioner_independent_claim_ready": False,
            "template": "meta_data/identity_map_template.csv",
        },
        "source_holdout_protocols": source_holdout_summary,
        "video_link_status": dict(sorted(Counter(record["video_link_status"] for record in records).items())),
        "eligible_video_link_status": dict(sorted(Counter(
            record["video_link_status"] for record in eligible_records
        ).items())),
        "clip_materialization_status": dict(sorted(Counter(
            clip["materialization_status"] for clip in clips
        ).items())),
        "record_qc_flags": dict(sorted(Counter(flag for record in records for flag in record["qc_flags"]).items())),
        "event_exclusion_reasons": dict(sorted(Counter(
            reason for event in all_events for reason in event["exclusion_reasons"]
        ).items())),
        "split_policy": eligible_records[0]["split_policy"] if eligible_records else None,
        "leakage_check": {
            "groups_in_multiple_splits": sorted(
                key for key, memberships in group_memberships.items() if len(memberships) > 1
            ),
            "exact_txt_hashes_in_multiple_splits": sorted(
                key for key, memberships in txt_hash_memberships.items() if len(memberships) > 1
            ),
            "exact_video_hashes_in_multiple_splits": sorted(
                key for key, memberships in video_hash_memberships.items() if len(memberships) > 1
            ),
        },
    }
    json_dump(output / "meta_data" / "dataset_summary.json", summary)

    dictionary = f"""# FSN 规范化数据说明\n\n## 统一结构\n\n- `meta_data/records.jsonl`：每个原始 TXT 一条记录，保存路径、SHA-256、EAF URI、标注时间、解析格式、视频属性、分组、split 和 QC。\n- `meta_data/events_all.jsonl`：每个原始时间段一条记录；包括空标签、歧义标签和被排除事件。\n- `meta_data/identity_map_template.csv`：患者、术者、治疗 session 和机构映射模板。\n- `annotations/*.json`：按原始记录组织的统一 JSON。\n- `manifests/train.jsonl`、`val.jsonl`、`test.jsonl`：8:1:1 可监督 clip 清单。\n- `manifests/source_holdout/*/`：逐来源留一的域外测试清单。\n- `manifests/clips_all.csv`：与 clip JSONL 等价的 UTF-8-BOM 表格。\n\n## 清洗原则\n\n1. 三种原始格式统一为秒级 `start_sec/end_sec/duration_sec`。\n2. 原始标签永远保存在 `raw_label`；`normalized_label` 依据 `label_policy.json` 生成。\n3. 空标签、歧义/非规范标签、非正时长、短于 0.1 秒的事件不进入监督 clip，但仍保留。\n4. 完全相同的 TXT 只保留一个用于建模，副本标记 `duplicate_of_record_id`。\n5. 源 TXT 不修改；当前样例视频还额外保存文件哈希、帧数、帧率、编码、时长和分辨率。\n\n## 默认划分原则\n\n- 请求比例：train={1-args.val_ratio-args.test_ratio:.2%}、val={args.val_ratio:.2%}、test={args.test_ratio:.2%}；seed=`{args.seed}`。\n- 先按来源分层，再按 `group_id` 整组分配；clip 继承记录 split。\n- 能识别 `ACU###` 时按患者码分组；其余在身份表未填写前只能按原始记录分组，明确标为低可信度。\n- 目前的 8:1:1 是可运行的 provisional split，不能冒充完整的 patient/practitioner-independent split。\n- 填写 `identity_map_template.csv` 后应重建严格的患者、术者或联合身份隔离协议。\n- 来源留一 folds 用于测量 domain shift，但不能替代真正新增机构/新增患者的外部验证集。\n\n## 当前快照\n\n- 原始 TXT：{len(records)}\n- 当前可用视频：{summary['input']['available_video_files']}\n- train/val/test/excluded 记录：{summary['records_by_split']}\n- 可监督 clips：{len(clips)}\n- 泄漏检查：`groups_in_multiple_splits`、`exact_txt_hashes_in_multiple_splits`、`exact_video_hashes_in_multiple_splits` 均应为空。\n"""
    (output / "meta_data" / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")

    print(json.dumps({
        "output": relpath(output, workspace),
        "records": len(records),
        "events": len(all_events),
        "clips": len(clips),
        "records_by_split": summary["records_by_split"],
        "formats": summary["formats"],
        "leakage_check": summary["leakage_check"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
