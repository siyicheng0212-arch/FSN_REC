"""Efficient one-process-per-clip cache for full FSN training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pilot_data import ClipRecord, VideoDecodeError, _get_ffmpeg_exe, load_pilot_manifest


CACHE_VERSION = "fsn-full-uniform-ffmpeg-v1"


def cache_paths(cache_root: Path, record: ClipRecord) -> tuple[Path, Path]:
    directory = cache_root / record.split
    return directory / f"{record.clip_id}.npy", directory / f"{record.clip_id}.json"


def request_digest(record: ClipRecord, num_frames: int, crop_size: int) -> str:
    payload = {
        "version": CACHE_VERSION,
        "clip_id": record.clip_id,
        "video_path": record.video_path,
        "start": record.clip_start_sec,
        "end": record.clip_end_sec,
        "num_frames": num_frames,
        "crop_size": crop_size,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def extract_clip(record: ClipRecord, num_frames: int, crop_size: int, timeout: float) -> np.ndarray:
    if not Path(record.video_path).is_file():
        raise VideoDecodeError(f"source unavailable for {record.clip_id}")
    bin_width = record.clip_duration_sec / num_frames
    first_timestamp = record.clip_start_sec + 0.5 * bin_width
    sampling_seconds = max(record.clip_end_sec - first_timestamp, bin_width)
    fps = 1.0 / bin_width
    video_filter = (
        f"fps={fps:.12f},"
        f"scale={crop_size}:{crop_size}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={crop_size}:{crop_size},setsar=1"
    )
    command = [
        _get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{first_timestamp:.9f}", "-i", record.video_path,
        "-t", f"{sampling_seconds:.9f}", "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", video_filter, "-frames:v", str(num_frames), "-threads", "1",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError(f"decode timeout for {record.clip_id}") from exc
    expected = num_frames * crop_size * crop_size * 3
    if completed.returncode != 0 or len(completed.stdout) != expected:
        raise VideoDecodeError(
            f"decode failed for {record.clip_id} "
            f"(exit={completed.returncode}, bytes={len(completed.stdout)}/{expected})"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        num_frames, crop_size, crop_size, 3
    ).copy()


def valid_cache(record: ClipRecord, cache_root: Path, num_frames: int, crop_size: int) -> bool:
    array_path, metadata_path = cache_paths(cache_root, record)
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("request_digest") != request_digest(record, num_frames, crop_size):
            return False
        frames = np.load(array_path, mmap_mode="r", allow_pickle=False)
        return frames.dtype == np.uint8 and frames.shape == (num_frames, crop_size, crop_size, 3)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def cache_one(record: ClipRecord, cache_root: Path, num_frames: int, crop_size: int, timeout: float) -> str:
    array_path, metadata_path = cache_paths(cache_root, record)
    if valid_cache(record, cache_root, num_frames, crop_size):
        return "reused"
    frames = extract_clip(record, num_frames, crop_size, timeout)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=array_path.parent, suffix=".npy", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.save(temporary, frames, allow_pickle=False)
    os.replace(temporary_path, array_path)
    metadata = {
        "cache_version": CACHE_VERSION,
        "request_digest": request_digest(record, num_frames, crop_size),
        "clip_id": record.clip_id,
        "split": record.split,
        "label_id": record.label_id,
        "source_collection": record.source_collection,
        "group_id": record.group_id,
        "duration": record.clip_duration_sec,
        "num_frames": num_frames,
        "crop_size": crop_size,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    if not valid_cache(record, cache_root, num_frames, crop_size):
        raise RuntimeError(f"cache validation failed for {record.clip_id}")
    return "cached"


def cache_manifest(
    manifest: Path,
    cache_root: Path,
    num_frames: int,
    crop_size: int,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    records = load_pilot_manifest(manifest)
    counts = {"cached": 0, "reused": 0, "failed": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(cache_one, record, cache_root, num_frames, crop_size, timeout): record
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                counts[future.result()] += 1
            except Exception:
                counts["failed"] += 1
                failures.append(record.clip_id)
            if completed % 100 == 0 or completed == len(records):
                print(json.dumps({"completed": completed, "total": len(records), **counts}), flush=True)
    result = {"manifest": manifest.name, "total": len(records), **counts, "failed_clip_ids": failures}
    if failures:
        raise RuntimeError(json.dumps(result))
    return result


class FullClipDataset:
    def __init__(self, manifest: Path, cache_root: Path, num_frames: int = 36, crop_size: int = 224):
        import torch

        self.torch = torch
        self.records = load_pilot_manifest(manifest)
        self.cache_root = cache_root
        self.num_frames = num_frames
        self.crop_size = crop_size
        missing = [
            record.clip_id for record in self.records
            if not valid_cache(record, cache_root, num_frames, crop_size)
        ]
        if missing:
            raise RuntimeError(f"{len(missing)} invalid/missing caches; run full_data cache first")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        array_path, _ = cache_paths(self.cache_root, record)
        frames = np.load(array_path, allow_pickle=False)
        video = self.torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().div_(255)
        return {
            "video": video,
            "label": record.label_id,
            "clip_id": record.clip_id,
            "source": record.source_collection,
            "duration": record.clip_duration_sec,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--cache-dir", type=Path, required=True)
    cache.add_argument("--num-frames", type=int, default=36)
    cache.add_argument("--crop-size", type=int, default=224)
    cache.add_argument("--workers", type=int, default=8)
    cache.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.command == "cache":
        print(json.dumps(cache_manifest(
            args.manifest, args.cache_dir, args.num_frames,
            args.crop_size, args.workers, args.timeout,
        ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
