#!/usr/bin/env python3
"""Quarantine exact duplicate videos without breaking annotation aliases.

Only files within the same source directory that have the same basename,
size, and SHA-256 are eligible. Different basenames are never moved because
they may each have a corresponding annotation record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_files(source: Path) -> list[Path]:
    return sorted(
        path for path in source.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not path.name.startswith("._")
        and path.name != ".DS_Store"
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def preferred_copy(paths: list[Path], source: Path) -> Path:
    return min(
        paths,
        key=lambda path: (
            len(path.relative_to(source).parts),
            path.relative_to(source).as_posix(),
        ),
    )


def plan_source(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = candidate_files(source)
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_name[path.name].append(path)

    planned: list[dict[str, Any]] = []
    conflicting_names = 0
    hashed_files = 0
    for same_name in by_name.values():
        if len(same_name) < 2:
            continue
        by_size: dict[int, list[Path]] = defaultdict(list)
        for path in same_name:
            by_size[path.stat().st_size].append(path)
        content_groups: list[list[Path]] = []
        distinct_content_keys: set[tuple[int, str]] = set()
        for size, same_size in by_size.items():
            if len(same_size) < 2:
                distinct_content_keys.add((size, "unique-size"))
                continue
            by_hash: dict[str, list[Path]] = defaultdict(list)
            for path in same_size:
                digest = sha256_file(path)
                hashed_files += 1
                by_hash[digest].append(path)
                distinct_content_keys.add((size, digest))
            content_groups.extend(group for group in by_hash.values() if len(group) > 1)
        if len(distinct_content_keys) > 1:
            conflicting_names += 1
        for group in content_groups:
            keep = preferred_copy(group, source)
            digest = sha256_file(keep)
            for duplicate in group:
                if duplicate == keep:
                    continue
                planned.append({
                    "source": source.name,
                    "keep": str(keep),
                    "duplicate": str(duplicate),
                    "relative_duplicate": duplicate.relative_to(source).as_posix(),
                    "size_bytes": duplicate.stat().st_size,
                    "sha256": digest,
                })

    summary = {
        "source": source.name,
        "video_files": len(files),
        "unique_basenames": len(by_name),
        "same_name_groups": sum(len(paths) > 1 for paths in by_name.values()),
        "same_name_different_content_groups": conflicting_names,
        "hashed_files": hashed_files,
        "planned_quarantine_files": len(planned),
        "planned_quarantine_bytes": sum(item["size_bytes"] for item in planned),
    }
    return planned, summary


def apply_plan(plan: list[dict[str, Any]], video_root: Path, quarantine: Path) -> Path:
    quarantine.mkdir(parents=True, exist_ok=True)
    log_path = quarantine / "dedupe_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        for item in plan:
            duplicate = Path(item["duplicate"])
            source_root = video_root / item["source"]
            relative = duplicate.relative_to(source_root)
            destination = quarantine / item["source"] / relative
            destination = destination.with_name(destination.name + ".duplicate")
            if destination.exists():
                raise FileExistsError(f"quarantine destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(duplicate), str(destination))
            log.write(json.dumps({**item, "quarantine": str(destination)}, ensure_ascii=False) + "\n")
            log.flush()
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["bilibili_video", "ks_video", "youtube_video"])
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--apply", action="store_true", help="move duplicates; default is dry-run")
    args = parser.parse_args()
    video_root = args.video_root.resolve()
    quarantine = (args.quarantine or (video_root / "_duplicate_quarantine")).resolve()
    plan: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for source_name in args.sources:
        source = video_root / source_name
        if not source.is_dir():
            raise SystemExit(f"source directory does not exist: {source}")
        source_plan, source_summary = plan_source(source)
        plan.extend(source_plan)
        summaries.append(source_summary)

    result: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "safety_rule": "same source + same basename + same size + same SHA-256",
        "sources": summaries,
        "total_planned_files": len(plan),
        "total_planned_bytes": sum(item["size_bytes"] for item in plan),
    }
    if args.apply:
        result["log"] = str(apply_plan(plan, video_root, quarantine))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
