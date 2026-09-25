#!/usr/bin/env python3
"""Materialize clip manifests after source videos have been restored."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed_dataset/manifests/clips_all.jsonl"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("processed_dataset"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    avconvert = shutil.which("avconvert")
    if not ffmpeg and not avconvert:
        raise SystemExit("Neither ffmpeg nor macOS avconvert is available.")

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    selected = [row for row in rows if args.split == "all" or row["split"] == args.split]
    created = skipped = missing = failed = 0
    for row in selected:
        if not row["video_path"]:
            missing += 1
            continue
        source = (args.workspace / row["video_path"]).resolve()
        destination = (args.output_root / row["planned_clip_path"]).resolve()
        if not source.exists():
            missing += 1
            continue
        if destination.exists() and not args.force:
            skipped += 1
            continue
        duration = row["clip_end_sec"] - row["clip_start_sec"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if ffmpeg:
            command = [
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y" if args.force else "-n",
                "-ss", f"{row['clip_start_sec']:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
                "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "-1",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", str(destination),
            ]
        else:
            command = [
                avconvert, "--source", str(source), "--preset", "PresetHighestQuality",
                "--output", str(destination), "--start", f"{row['clip_start_sec']:.6f}",
                "--duration", f"{duration:.6f}",
            ]
            if args.force:
                command.append("--replace")
        if args.dry_run:
            print(json.dumps({"clip_id": row["clip_id"], "command": command}, ensure_ascii=False))
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            created += 1
        else:
            failed += 1
    print(json.dumps({
        "selected": len(selected), "created": created, "skipped": skipped,
        "missing_video": missing, "failed": failed, "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
