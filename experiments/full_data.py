"""Efficient one-process-per-clip cache for full FSN training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.pilot_data import ClipRecord, VideoDecodeError, _get_ffmpeg_exe, load_pilot_manifest


CACHE_VERSION = "fsn-full-uniform-ffmpeg-v2-short-clip-audit"
THREE_WINDOWS_CACHE_VERSION = "fsn-full-three-windows-1s-v1"
SAMPLING_MODES = ("uniform", "three_windows")
THREE_WINDOWS_DURATION_THRESHOLD_SEC = 3.0
THREE_WINDOWS_WINDOW_SEC = 1.0
SHORT_CLIP_POLICY = "fixed-length uniform temporal positions with deterministic ffmpeg frame repetition"
MIN_DECODED_FRAME_FRACTION = 0.80
DECODE_PADDING_POLICY = (
    "clone the final decoded frame only when ffmpeg returns at least 80% of "
    "the requested frames; retain decoded/padded counts in cache metadata"
)


def cache_paths(cache_root: Path, record: ClipRecord) -> tuple[Path, Path]:
    directory = cache_root / record.split
    return directory / f"{record.clip_id}.npy", directory / f"{record.clip_id}.json"


def sampling_windows(
    record: ClipRecord, num_frames: int, sampling: str = "uniform"
) -> list[tuple[float, float, int]]:
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"unknown sampling mode: {sampling}")
    if num_frames <= 0 or record.clip_duration_sec <= 0:
        raise ValueError("sampling requires positive frame count and duration")
    if sampling == "uniform" or record.clip_duration_sec <= THREE_WINDOWS_DURATION_THRESHOLD_SEC:
        return [(record.clip_start_sec, record.clip_end_sec, num_frames)]
    if num_frames % 3:
        raise ValueError("three_windows requires a frame count divisible by three")
    duration = record.clip_duration_sec
    start = record.clip_start_sec
    frames_per_window = num_frames // 3
    centers = (duration / 6.0, duration / 2.0, duration * 5.0 / 6.0)
    return [
        (
            start + center - THREE_WINDOWS_WINDOW_SEC / 2.0,
            start + center + THREE_WINDOWS_WINDOW_SEC / 2.0,
            frames_per_window,
        )
        for center in centers
    ]


def requested_timestamps(
    record: ClipRecord, num_frames: int, sampling: str = "uniform"
) -> list[float]:
    return [
        window_start + (index + 0.5) * (window_end - window_start) / frames
        for window_start, window_end, frames in sampling_windows(record, num_frames, sampling)
        for index in range(frames)
    ]


def request_digest(
    record: ClipRecord, num_frames: int, crop_size: int, sampling: str = "uniform"
) -> str:
    sampling_windows(record, num_frames, sampling)
    payload = {
        "version": CACHE_VERSION,
        "clip_id": record.clip_id,
        "video_path": record.video_path,
        "start": record.clip_start_sec,
        "end": record.clip_end_sec,
        "num_frames": num_frames,
        "crop_size": crop_size,
    }
    if sampling == "three_windows":
        source = Path(record.video_path).stat()
        payload.update({
            "version": THREE_WINDOWS_CACHE_VERSION,
            "sampling": sampling,
            "window_sec": THREE_WINDOWS_WINDOW_SEC,
            "duration_threshold_sec": THREE_WINDOWS_DURATION_THRESHOLD_SEC,
            "video_size_bytes": source.st_size,
            "video_mtime_ns": source.st_mtime_ns,
        })
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _extract_clip_with_audit(
    record: ClipRecord,
    num_frames: int,
    crop_size: int,
    timeout: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if record.clip_duration_sec <= 0:
        raise VideoDecodeError(f"clip {record.clip_id} has non-positive duration")
    if not Path(record.video_path).is_file():
        raise VideoDecodeError(f"source unavailable for {record.clip_id}")
    bin_width = record.clip_duration_sec / num_frames
    first_timestamp = record.clip_start_sec + 0.5 * bin_width
    # Give FFmpeg one complete clip-duration window from the first bin centre.
    # Output is still capped at ``num_frames``, so the requested timestamps end
    # at the final bin centre.  The former half-bin-short window could round to
    # 35 frames for a 36-frame request on otherwise valid videos.
    sampling_seconds = record.clip_duration_sec
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
    frame_bytes = crop_size * crop_size * 3
    expected = num_frames * frame_bytes
    actual = len(completed.stdout)
    if completed.returncode != 0 or actual == 0 or actual % frame_bytes:
        raise VideoDecodeError(
            f"decode failed for {record.clip_id} "
            f"(exit={completed.returncode}, bytes={actual}/{expected})"
        )
    decoded_frames = actual // frame_bytes
    minimum_safe_frames = math.ceil(num_frames * MIN_DECODED_FRAME_FRACTION)
    if not minimum_safe_frames <= decoded_frames <= num_frames:
        raise VideoDecodeError(
            f"decode produced an unsafe frame count for {record.clip_id} "
            f"(exit={completed.returncode}, frames={decoded_frames}/{num_frames}, "
            f"minimum_safe={minimum_safe_frames})"
        )
    frames = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        decoded_frames, crop_size, crop_size, 3
    )
    padding_frames = num_frames - decoded_frames
    if padding_frames:
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], padding_frames, axis=0)],
            axis=0,
        )
    return frames.copy(), {
        "ffmpeg_decoded_frames": decoded_frames,
        "decoder_padding_frames": padding_frames,
    }


def extract_clip(record: ClipRecord, num_frames: int, crop_size: int, timeout: float) -> np.ndarray:
    frames, _ = _extract_clip_with_audit(record, num_frames, crop_size, timeout)
    return frames


def extract_sampled_clip(
    record: ClipRecord,
    num_frames: int,
    crop_size: int,
    timeout: float,
    sampling: str = "uniform",
) -> tuple[np.ndarray, dict[str, int]]:
    windows = sampling_windows(record, num_frames, sampling)
    if len(windows) == 1:
        return _extract_clip_with_audit(record, num_frames, crop_size, timeout)
    arrays = []
    audits = []
    for start, end, frames in windows:
        window_record = replace(
            record,
            clip_start_sec=start,
            clip_end_sec=end,
            clip_duration_sec=end - start,
        )
        array, audit = _extract_clip_with_audit(
            window_record, frames, crop_size, timeout
        )
        arrays.append(array)
        audits.append(audit)
    return np.concatenate(arrays, axis=0), {
        key: sum(audit[key] for audit in audits)
        for key in ("ffmpeg_decoded_frames", "decoder_padding_frames")
    }


def valid_cache(
    record: ClipRecord, cache_root: Path, num_frames: int, crop_size: int,
    sampling: str = "uniform",
) -> bool:
    array_path, metadata_path = cache_paths(cache_root, record)
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("request_digest") != request_digest(record, num_frames, crop_size, sampling):
            return False
        frames = np.load(array_path, mmap_mode="r", allow_pickle=False)
        return frames.dtype == np.uint8 and frames.shape == (num_frames, crop_size, crop_size, 3)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def cache_one(
    record: ClipRecord, cache_root: Path, num_frames: int, crop_size: int,
    timeout: float, sampling: str = "uniform",
) -> str:
    array_path, metadata_path = cache_paths(cache_root, record)
    if valid_cache(record, cache_root, num_frames, crop_size, sampling):
        return "reused"
    frames, decode_audit = extract_sampled_clip(
        record, num_frames, crop_size, timeout, sampling
    )
    array_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=array_path.parent, suffix=".npy", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.save(temporary, frames, allow_pickle=False)
    os.replace(temporary_path, array_path)
    metadata = {
        "cache_version": (
            CACHE_VERSION if sampling == "uniform" else THREE_WINDOWS_CACHE_VERSION
        ),
        "request_digest": request_digest(record, num_frames, crop_size, sampling),
        "clip_id": record.clip_id,
        "split": record.split,
        "label_id": record.label_id,
        "source_collection": record.source_collection,
        "group_id": record.group_id,
        "duration": record.clip_duration_sec,
        "num_frames": num_frames,
        "crop_size": crop_size,
        "short_clip_sampling_policy": SHORT_CLIP_POLICY,
        "decode_padding_policy": DECODE_PADDING_POLICY,
        **decode_audit,
        "pixel_unique_frames": len({
            hashlib.sha256(frame.tobytes()).digest() for frame in frames
        }),
    }
    if sampling == "three_windows":
        metadata.update({
            "sampling": sampling,
            "window_sec": THREE_WINDOWS_WINDOW_SEC,
            "duration_threshold_sec": THREE_WINDOWS_DURATION_THRESHOLD_SEC,
            "requested_timestamps_sec": requested_timestamps(record, num_frames, sampling),
        })
    metadata["pixel_repeat_fraction"] = 1.0 - metadata["pixel_unique_frames"] / num_frames
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    if not valid_cache(record, cache_root, num_frames, crop_size, sampling):
        raise RuntimeError(f"cache validation failed for {record.clip_id}")
    return "cached"


def cache_manifest(
    manifest: Path,
    cache_root: Path,
    num_frames: int,
    crop_size: int,
    workers: int,
    timeout: float,
    sampling: str = "uniform",
) -> dict[str, Any]:
    records = load_pilot_manifest(manifest)
    counts = {"cached": 0, "reused": 0, "failed": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(cache_one, record, cache_root, num_frames, crop_size, timeout, sampling): record
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
    result = {"manifest": manifest.name, "sampling": sampling, "total": len(records), **counts, "failed_clip_ids": failures}
    if failures:
        raise RuntimeError(json.dumps(result))
    return result


class FullClipDataset:
    def __init__(
        self, manifest: Path, cache_root: Path, num_frames: int = 36,
        crop_size: int = 224, sampling: str = "uniform",
    ):
        import torch

        self.torch = torch
        self.records = load_pilot_manifest(manifest)
        self.cache_root = cache_root
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.sampling = sampling
        missing = [
            record.clip_id for record in self.records
            if not valid_cache(record, cache_root, num_frames, crop_size, sampling)
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
            "record_id": record.record_id or record.clip_id,
            "source": record.source_collection,
            "duration": record.clip_duration_sec,
            "clip_start_sec": record.clip_start_sec,
            "clip_end_sec": record.clip_end_sec,
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
    cache.add_argument("--sampling", choices=SAMPLING_MODES, default="uniform")
    args = parser.parse_args()
    if args.command == "cache":
        print(json.dumps(cache_manifest(
            args.manifest, args.cache_dir, args.num_frames,
            args.crop_size, args.workers, args.timeout, args.sampling,
        ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
