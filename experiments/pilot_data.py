"""Strict, model-neutral data layer for small FSN action-recognition pilots.

Video decoding is deliberately kept out of ``Dataset.__getitem__``. Clips are
decoded once into opaque ``<clip_id>.npz`` files so every compared model sees
the exact same RGB tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_LABELS: dict[int, str] = {
    0: "消毒",
    1: "进针",
    2: "运针",
    3: "扫散",
    4: "再灌注",
    5: "拔针",
    6: "固定",
}
SPLITS = ("train", "val", "test")
CACHE_SCHEMA = "fsn-pilot-cache-1.0"
PILOT_SCHEMA = "fsn-pilot-manifest-1.0"
EXTRACTION_VERSION = "uniform-bin-centres-ffmpeg-rgb24-v1"
_CLIP_ID_RE = re.compile(r"^clip-[A-Za-z0-9_-]{1,120}$")


class PilotDataError(RuntimeError):
    """Base class for errors safe to display in the CLI."""


class ManifestValidationError(PilotDataError):
    pass


class DependencyUnavailableError(PilotDataError):
    pass


class VideoDecodeError(PilotDataError):
    pass


class CacheValidationError(PilotDataError):
    pass


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    split: str
    label_id: int
    normalized_label: str
    source_collection: str
    group_id: str
    video_path: str
    clip_start_sec: float
    clip_end_sec: float
    clip_duration_sec: float

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PILOT_SCHEMA,
            "clip_id": self.clip_id,
            "split": self.split,
            "label_id": self.label_id,
            "normalized_label": self.normalized_label,
            "source_collection": self.source_collection,
            "group_id": self.group_id,
            "video_path": self.video_path,
            "clip_start_sec": self.clip_start_sec,
            "clip_end_sec": self.clip_end_sec,
            "clip_duration_sec": self.clip_duration_sec,
        }


def _finite_number(value: Any, field: str, clip_id: str) -> float:
    if isinstance(value, bool):
        raise ManifestValidationError(f"{field} is not numeric for {clip_id}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} is not numeric for {clip_id}") from exc
    if not math.isfinite(number):
        raise ManifestValidationError(f"{field} is not finite for {clip_id}")
    return number


def _parse_record(
    row: Mapping[str, Any], *, expected_split: str | None, require_video: bool
) -> ClipRecord:
    clip_id = row.get("clip_id")
    if not isinstance(clip_id, str) or not _CLIP_ID_RE.fullmatch(clip_id):
        raise ManifestValidationError("invalid or unsafe clip_id in manifest")
    split = row.get("split")
    if split not in SPLITS:
        raise ManifestValidationError(f"invalid split for {clip_id}")
    if expected_split is not None and split != expected_split:
        raise ManifestValidationError(f"split mismatch for {clip_id}")

    label_id = row.get("label_id")
    if isinstance(label_id, bool) or not isinstance(label_id, int):
        raise ManifestValidationError(f"label_id is not an integer for {clip_id}")
    label_name = row.get("normalized_label")
    if label_id not in EXPECTED_LABELS or label_name != EXPECTED_LABELS[label_id]:
        raise ManifestValidationError(f"unexpected seven-class label mapping for {clip_id}")
    source = row.get("source_collection")
    if not isinstance(source, str) or not source.strip():
        raise ManifestValidationError(f"missing source_collection for {clip_id}")
    group_id = row.get("group_id")
    if not isinstance(group_id, str) or not group_id.strip():
        # Clip-specific fallback is only for pilot diversity; it is not subject ID evidence.
        group_id = clip_id

    start = _finite_number(row.get("clip_start_sec"), "clip_start_sec", clip_id)
    end = _finite_number(row.get("clip_end_sec"), "clip_end_sec", clip_id)
    duration = _finite_number(row.get("clip_duration_sec"), "clip_duration_sec", clip_id)
    if start < 0 or end <= start or duration <= 0:
        raise ManifestValidationError(f"invalid clip interval for {clip_id}")
    if not math.isclose(end - start, duration, rel_tol=0.0, abs_tol=1e-6):
        raise ManifestValidationError(f"duration does not match interval for {clip_id}")

    video_path = row.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        raise ManifestValidationError(f"missing source video for {clip_id}")
    path = Path(video_path)
    if not path.is_absolute():
        raise ManifestValidationError(f"video_path is not absolute for {clip_id}")
    if require_video and not path.is_file():
        raise ManifestValidationError(
            f"source video is unavailable for {clip_id}; mount the dataset before continuing"
        )
    return ClipRecord(
        clip_id=clip_id,
        split=split,
        label_id=label_id,
        normalized_label=label_name,
        source_collection=source,
        group_id=group_id,
        video_path=video_path,
        clip_start_sec=start,
        clip_end_sec=end,
        clip_duration_sec=duration,
    )


def _read_jsonl_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ManifestValidationError(f"cannot open manifest {path.name}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"invalid JSON in {path.name} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ManifestValidationError(
                    f"non-object row in {path.name} at line {line_number}"
                )
            yield line_number, row


def load_pilot_manifest(path: str | os.PathLike[str]) -> list[ClipRecord]:
    """Load and strictly validate one generated pilot manifest."""
    manifest_path = Path(path)
    records: list[ClipRecord] = []
    seen: set[str] = set()
    split: str | None = None
    for _, row in _read_jsonl_rows(manifest_path):
        record = _parse_record(row, expected_split=split, require_video=False)
        split = record.split
        if record.clip_id in seen:
            raise ManifestValidationError(f"duplicate clip_id {record.clip_id}")
        seen.add(record.clip_id)
        records.append(record)
    if not records:
        raise ManifestValidationError(f"manifest {manifest_path.name} is empty")
    return records


def _stable_score(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _diverse_pick(
    candidates: Sequence[ClipRecord], count: int, *, seed: int, split: str, label_id: int
) -> list[ClipRecord]:
    """Deterministically round-robin sources, preferring distinct groups."""
    pools: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in candidates:
        pools[record.source_collection].append(record)
    for source, pool in pools.items():
        pool.sort(key=lambda item: _stable_score(seed, split, label_id, source, item.clip_id))
    sources = sorted(
        pools, key=lambda source: _stable_score(seed, split, label_id, "source", source)
    )
    selected: list[ClipRecord] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()
    for require_new_group in (True, False):
        while len(selected) < count:
            made_progress = False
            for source in sources:
                choice = next(
                    (
                        item
                        for item in pools[source]
                        if item.clip_id not in selected_ids
                        and (not require_new_group or item.group_id not in selected_groups)
                    ),
                    None,
                )
                if choice is None:
                    continue
                selected.append(choice)
                selected_ids.add(choice.clip_id)
                selected_groups.add(choice.group_id)
                made_progress = True
                if len(selected) == count:
                    return selected
            if not made_progress:
                break
    return selected


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_balanced_pilot_manifests(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    per_class: int = 4,
    seed: int = 20260925,
    require_videos: bool = True,
    verify_decodable: bool = False,
    preflight_samples: int = 2,
    preflight_crop_size: int = 32,
    preflight_timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Create deterministic, seven-class-balanced train/val/test manifests.

    Null paths explicitly marked ``waiting_for_video`` are excluded and counted.
    A non-null unavailable path fails immediately when ``require_videos`` is true,
    which catches an unmounted external disk.
    """
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class <= 0:
        raise ValueError("per_class must be a positive integer")
    source_dir, destination = Path(input_dir), Path(output_dir)
    summary: dict[str, Any] = {
        "schema_version": PILOT_SCHEMA,
        "seed": seed,
        "per_class_per_split": per_class,
        "splits": {},
    }
    all_clip_ids: set[str] = set()
    selected_by_split: dict[str, list[ClipRecord]] = {}
    for split in SPLITS:
        candidates_by_label: dict[int, list[ClipRecord]] = defaultdict(list)
        unavailable_declared = 0
        for _, row in _read_jsonl_rows(source_dir / f"{split}.jsonl"):
            if row.get("video_path") in (None, "") and row.get("materialization_status") == "waiting_for_video":
                unavailable_declared += 1
                continue
            record = _parse_record(row, expected_split=split, require_video=require_videos)
            if record.clip_id in all_clip_ids:
                raise ManifestValidationError(f"duplicate clip_id {record.clip_id}")
            all_clip_ids.add(record.clip_id)
            candidates_by_label[record.label_id].append(record)
        if set(candidates_by_label) != set(EXPECTED_LABELS):
            raise ManifestValidationError(f"{split} does not contain all seven labels")

        chosen: list[ClipRecord] = []
        decode_rejected: Counter[int] = Counter()
        for label_id in EXPECTED_LABELS:
            available = list(candidates_by_label[label_id])
            if len(available) < per_class:
                raise ManifestValidationError(
                    f"{split} label {label_id} has {len(available)} eligible clips; {per_class} required"
                )
            verified_ids: set[str] = set()
            while True:
                picked = _diverse_pick(
                    available, per_class, seed=seed, split=split, label_id=label_id
                )
                if len(picked) != per_class:
                    raise ManifestValidationError(
                        f"could not select enough decodable clips for {split} label {label_id}"
                    )
                if not verify_decodable:
                    break
                rejected: ClipRecord | None = None
                for record in picked:
                    if record.clip_id in verified_ids:
                        continue
                    try:
                        _preflight_record(
                            record,
                            sample_count=preflight_samples,
                            crop_size=preflight_crop_size,
                            timeout_sec=preflight_timeout_sec,
                        )
                    except VideoDecodeError:
                        rejected = record
                        break
                    verified_ids.add(record.clip_id)
                if rejected is None:
                    break
                # The complete clip is excluded and counted. No frame is ever
                # substituted, and the deterministic selector chooses again.
                available = [item for item in available if item.clip_id != rejected.clip_id]
                decode_rejected[label_id] += 1
            chosen.extend(picked)
        selected_by_split[split] = chosen
        class_counts = Counter(item.label_id for item in chosen)
        source_counts = Counter(item.source_collection for item in chosen)
        summary["splits"][split] = {
            "selected": len(chosen),
            "declared_unavailable_skipped": unavailable_declared,
            "class_counts": {str(key): class_counts[key] for key in EXPECTED_LABELS},
            "source_counts": dict(sorted(source_counts.items())),
            "unique_groups": len({item.group_id for item in chosen}),
            "decode_rejected": {
                str(key): decode_rejected[key] for key in EXPECTED_LABELS
            },
        }

    destination.mkdir(parents=True, exist_ok=True)
    for split, chosen in selected_by_split.items():
        _atomic_write_jsonl(
            destination / f"{split}.jsonl",
            (record.to_manifest_dict() for record in chosen),
        )
    _atomic_write_json(destination / "summary.json", summary)
    return summary


def _get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise DependencyUnavailableError(
            "imageio-ffmpeg is required; install experiments/requirements-pilot.txt"
        ) from exc
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise DependencyUnavailableError("imageio-ffmpeg could not provide its ffmpeg binary") from exc
    if not executable or not Path(executable).is_file():
        raise DependencyUnavailableError("imageio-ffmpeg returned an unavailable ffmpeg binary")
    return executable


def _decode_one_frame(
    ffmpeg_exe: str,
    record: ClipRecord,
    timestamp: float,
    sample_index: int,
    total_samples: int,
    crop_size: int,
    timeout_sec: float,
) -> np.ndarray:
    video_filter = (
        f"scale={crop_size}:{crop_size}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={crop_size}:{crop_size},setsar=1"
    )
    command = [
        ffmpeg_exe,
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{timestamp:.9f}", "-i", record.video_path,
        "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1",
        "-threads", "1", "-vf", video_filter, "-pix_fmt", "rgb24",
        "-f", "rawvideo", "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError(
            f"decode timed out for {record.clip_id} at sample {sample_index + 1}/{total_samples}"
        ) from exc
    except OSError as exc:
        raise VideoDecodeError(
            f"ffmpeg could not start for {record.clip_id} at sample {sample_index + 1}/{total_samples}"
        ) from exc
    expected_bytes = crop_size * crop_size * 3
    if result.returncode != 0 or len(result.stdout) != expected_bytes:
        # stderr can contain an absolute source path, so it is never echoed.
        raise VideoDecodeError(
            f"decode failed for {record.clip_id} at sample {sample_index + 1}/{total_samples} "
            f"(exit={result.returncode}, bytes={len(result.stdout)}/{expected_bytes})"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(crop_size, crop_size, 3).copy()


def extract_uniform_frames(
    record: ClipRecord,
    *,
    num_frames: int = 8,
    crop_size: int = 224,
    timeout_sec: float = 30.0,
) -> np.ndarray:
    """Extract RGB frames at equal-bin centres with no fallback substitution."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError("crop_size must be a positive integer")
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if not Path(record.video_path).is_file():
        raise VideoDecodeError(f"source video is unavailable for {record.clip_id}")
    ffmpeg_exe = _get_ffmpeg_exe()
    bin_width = record.clip_duration_sec / num_frames
    timestamps = [
        record.clip_start_sec + (index + 0.5) * bin_width for index in range(num_frames)
    ]
    frames = [
        _decode_one_frame(
            ffmpeg_exe, record, timestamp, index, num_frames, crop_size, timeout_sec
        )
        for index, timestamp in enumerate(timestamps)
    ]
    result = np.stack(frames, axis=0)
    if result.dtype != np.uint8 or result.shape != (num_frames, crop_size, crop_size, 3):
        raise VideoDecodeError(f"unexpected decoded tensor for {record.clip_id}")
    return result


def _preflight_record(
    record: ClipRecord,
    *,
    sample_count: int,
    crop_size: int,
    timeout_sec: float,
) -> None:
    """Decode a few bin-centre frames before admitting a clip to a pilot."""
    if sample_count <= 0 or crop_size <= 0 or timeout_sec <= 0:
        raise ValueError("preflight settings must be positive")
    if not Path(record.video_path).is_file():
        raise VideoDecodeError(f"source video is unavailable for {record.clip_id}")
    ffmpeg_exe = _get_ffmpeg_exe()
    width = record.clip_duration_sec / sample_count
    for index in range(sample_count):
        timestamp = record.clip_start_sec + (index + 0.5) * width
        _decode_one_frame(
            ffmpeg_exe,
            record,
            timestamp,
            index,
            sample_count,
            crop_size,
            timeout_sec,
        )


def cache_request_digest(record: ClipRecord, *, num_frames: int, crop_size: int) -> str:
    request = {
        "extraction_version": EXTRACTION_VERSION,
        "clip_id": record.clip_id,
        "video_path": record.video_path,
        "clip_start_sec": record.clip_start_sec,
        "clip_end_sec": record.clip_end_sec,
        "num_frames": num_frames,
        "crop_size": crop_size,
    }
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path_for(cache_dir: str | os.PathLike[str], record: ClipRecord) -> Path:
    """Return ``cache_dir/split/<opaque clip_id>.npz``."""
    return Path(cache_dir) / record.split / f"{record.clip_id}.npz"


def _validate_cache(
    path: Path, record: ClipRecord, *, num_frames: int, crop_size: int
) -> np.ndarray:
    if not path.is_file():
        raise CacheValidationError(f"cache is missing for {record.clip_id}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"frames", "request_digest", "schema_version", "num_frames", "crop_size"}
            if not required.issubset(archive.files):
                raise CacheValidationError(f"cache fields are incomplete for {record.clip_id}")
            schema = str(archive["schema_version"].item())
            digest = str(archive["request_digest"].item())
            stored_num_frames = int(archive["num_frames"].item())
            stored_crop_size = int(archive["crop_size"].item())
            frames = archive["frames"].copy()
    except CacheValidationError:
        raise
    except Exception as exc:
        raise CacheValidationError(f"cache cannot be read for {record.clip_id}") from exc
    expected_digest = cache_request_digest(record, num_frames=num_frames, crop_size=crop_size)
    if schema != CACHE_SCHEMA or digest != expected_digest:
        raise CacheValidationError(f"cache is stale for {record.clip_id}")
    if stored_num_frames != num_frames or stored_crop_size != crop_size:
        raise CacheValidationError(f"cache settings mismatch for {record.clip_id}")
    if frames.dtype != np.uint8 or frames.shape != (num_frames, crop_size, crop_size, 3):
        raise CacheValidationError(f"cache tensor is invalid for {record.clip_id}")
    return frames


def _write_cache(
    path: Path,
    record: ClipRecord,
    frames: np.ndarray,
    *,
    num_frames: int,
    crop_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                frames=frames,
                request_digest=np.asarray(
                    cache_request_digest(record, num_frames=num_frames, crop_size=crop_size)
                ),
                schema_version=np.asarray(CACHE_SCHEMA),
                num_frames=np.asarray(num_frames, dtype=np.int64),
                crop_size=np.asarray(crop_size, dtype=np.int64),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_manifest(
    manifest_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    *,
    num_frames: int = 8,
    crop_size: int = 224,
    timeout_sec: float = 30.0,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Decode all rows atomically; never repair or replace failed frames."""
    records = load_pilot_manifest(manifest_path)
    cached = reused = 0
    for index, record in enumerate(records, 1):
        output_path = cache_path_for(cache_dir, record)
        if output_path.exists() and not overwrite:
            _validate_cache(
                output_path, record, num_frames=num_frames, crop_size=crop_size
            )
            reused += 1
        else:
            frames = extract_uniform_frames(
                record,
                num_frames=num_frames,
                crop_size=crop_size,
                timeout_sec=timeout_sec,
            )
            _write_cache(
                output_path,
                record,
                frames,
                num_frames=num_frames,
                crop_size=crop_size,
            )
            _validate_cache(
                output_path, record, num_frames=num_frames, crop_size=crop_size
            )
            cached += 1
        if progress is not None:
            progress(index, len(records))
    return {"total": len(records), "cached": cached, "reused": reused}


class PilotClipDataset:
    """PyTorch Dataset yielding video, label, opaque ID, source, and duration."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str],
        *,
        num_frames: int = 8,
        crop_size: int = 224,
        transform: Callable[[Any], Any] | None = None,
        verify_all: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("torch is required for PilotClipDataset") from exc
        self._torch = torch
        self.records = load_pilot_manifest(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.transform = transform
        if verify_all:
            for record in self.records:
                _validate_cache(
                    cache_path_for(self.cache_dir, record),
                    record,
                    num_frames=self.num_frames,
                    crop_size=self.crop_size,
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = _validate_cache(
            cache_path_for(self.cache_dir, record),
            record,
            num_frames=self.num_frames,
            crop_size=self.crop_size,
        )
        video = self._torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        if self.transform is not None:
            video = self.transform(video)
        return {
            "video": video,
            "label": record.label_id,
            "clip_id": record.clip_id,
            "source": record.source_collection,
            "duration": record.clip_duration_sec,
        }


def _make_synthetic_video(ffmpeg_exe: str, output_path: Path) -> None:
    command = [
        ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=12:duration=2",
        "-an", "-c:v", "mpeg4", "-y", str(output_path),
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30
    )
    if result.returncode != 0 or not output_path.is_file():
        raise VideoDecodeError("could not create the non-sensitive synthetic smoke video")


def run_smoke_test(work_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """End-to-end extraction/cache/Dataset smoke test using generated pixels."""
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    video_path = (root / "synthetic.mp4").resolve()
    _make_synthetic_video(_get_ffmpeg_exe(), video_path)
    row = ClipRecord(
        clip_id="clip-synthetic-smoke",
        split="test",
        label_id=0,
        normalized_label=EXPECTED_LABELS[0],
        source_collection="synthetic",
        group_id="synthetic-group",
        video_path=str(video_path),
        clip_start_sec=0.25,
        clip_end_sec=1.75,
        clip_duration_sec=1.5,
    )
    manifest = root / "test.jsonl"
    _atomic_write_jsonl(manifest, [row.to_manifest_dict()])
    result = cache_manifest(manifest, root / "cache", num_frames=4, crop_size=32)
    dataset = PilotClipDataset(manifest, root / "cache", num_frames=4, crop_size=32)
    item = dataset[0]
    if tuple(item["video"].shape) != (4, 3, 32, 32) or item["label"] != 0:
        raise PilotDataError("smoke Dataset output did not match its contract")
    return {**result, "shape": list(item["video"].shape), "status": "pass"}


def _progress(completed: int, total: int) -> None:
    print(f"cache progress {completed}/{total}", file=sys.stderr, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="create balanced pilot manifests")
    build.add_argument("--input-dir", type=Path, default=Path("processed_external/manifests"))
    build.add_argument("--output-dir", type=Path, default=Path("experiments/pilot_manifests"))
    build.add_argument("--per-class", type=int, default=4)
    build.add_argument("--seed", type=int, default=20260925)
    build.add_argument("--allow-unmounted-videos", action="store_true")
    build.add_argument(
        "--verify-decodable",
        action="store_true",
        help="preflight selected clips and count/reselect whole clips that cannot decode",
    )
    build.add_argument("--preflight-samples", type=int, default=2)
    build.add_argument("--preflight-crop-size", type=int, default=32)
    build.add_argument("--preflight-timeout-sec", type=float, default=30.0)
    cache = commands.add_parser("cache", help="decode one pilot manifest into NPZ cache")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--cache-dir", type=Path, default=Path("experiments/pilot_cache"))
    cache.add_argument("--num-frames", type=int, default=8)
    cache.add_argument("--crop-size", type=int, default=224)
    cache.add_argument("--timeout-sec", type=float, default=30.0)
    cache.add_argument("--overwrite", action="store_true")
    smoke = commands.add_parser("smoke", help="run a synthetic end-to-end smoke test")
    smoke.add_argument("--work-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_balanced_pilot_manifests(
                args.input_dir,
                args.output_dir,
                per_class=args.per_class,
                seed=args.seed,
                require_videos=not args.allow_unmounted_videos,
                verify_decodable=args.verify_decodable,
                preflight_samples=args.preflight_samples,
                preflight_crop_size=args.preflight_crop_size,
                preflight_timeout_sec=args.preflight_timeout_sec,
            )
        elif args.command == "cache":
            result = cache_manifest(
                args.manifest,
                args.cache_dir,
                num_frames=args.num_frames,
                crop_size=args.crop_size,
                timeout_sec=args.timeout_sec,
                overwrite=args.overwrite,
                progress=_progress,
            )
        elif args.command == "smoke":
            if args.work_dir is not None:
                result = run_smoke_test(args.work_dir)
            else:
                with tempfile.TemporaryDirectory(prefix="fsn-pilot-smoke-") as temporary:
                    result = run_smoke_test(temporary)
        else:
            raise AssertionError(args.command)
    except PilotDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
