"""Read existing caches without confusing storage location with split role."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.full_data import cache_paths, valid_cache
from experiments.pilot_data import ClipRecord, load_pilot_manifest


def _checked(record, root, frames, size):
    if not valid_cache(record, root, frames, size):
        return False
    _, metadata_path = cache_paths(root, record)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, expected in (("clip_id", record.clip_id), ("label_id", record.label_id),
                          ("group_id", record.group_id)):
        if key not in metadata or metadata[key] != expected:
            raise RuntimeError(f"cache metadata {key} mismatch for {record.clip_id}")
    return True


def resolve_cache_record(record: ClipRecord, cache_root: Path, num_frames=36,
                         crop_size=224, declared_cache_split=None) -> ClipRecord:
    """Return a storage-only copy; never change the logical record or manifest.

    A development validation clip may physically remain in cache/train after
    recording-group splitting. Fallback is only permitted for an absent role
    path, never to hide a corrupt/mismatched role cache.
    """
    root = Path(cache_root)
    if record.split not in {"train", "val"}:
        raise ValueError("this development experiment does not read test")
    if declared_cache_split is not None:
        if declared_cache_split not in {"train", "val"}:
            raise ValueError("cache_split must be train or val")
        storage = replace(record, split=declared_cache_split)
        if not _checked(storage, root, num_frames, crop_size):
            raise RuntimeError(f"declared cache missing/invalid for {record.clip_id}")
        return storage
    role_paths = cache_paths(root, record)
    if any(path.exists() for path in role_paths):
        if not _checked(record, root, num_frames, crop_size):
            raise RuntimeError(f"role cache corrupt/mismatched for {record.clip_id}")
        other = replace(record, split="val" if record.split == "train" else "train")
        if _checked(other, root, num_frames, crop_size):
            raise RuntimeError(f"ambiguous duplicate cache for {record.clip_id}; declare cache_split")
        return record
    candidates = [replace(record, split=split) for split in ("train", "val")]
    valid = [row for row in candidates if _checked(row, root, num_frames, crop_size)]
    if len(valid) != 1:
        raise RuntimeError(f"expected one valid cache for {record.clip_id}, found {len(valid)}")
    return valid[0]


def mapping_sha256(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class ResolvedFullClipDataset:
    """Uint8 CPU tensors keep accumulation-window buffering inexpensive."""
    def __init__(self, manifest, cache_root, num_frames=36, crop_size=224):
        self.records = load_pilot_manifest(Path(manifest))
        self.cache_root = Path(cache_root)
        raw = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        explicit = {row["clip_id"]: row.get("cache_split") for row in raw}
        self.storage_records = [resolve_cache_record(
            row, self.cache_root, num_frames, crop_size, explicit[row.clip_id]
        ) for row in self.records]
        self.mapping = [{"split": role.split, "clip_id": role.clip_id, "cache_split": storage.split}
                        for role, storage in zip(self.records, self.storage_records)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, storage = self.records[index], self.storage_records[index]
        array = np.load(cache_paths(self.cache_root, storage)[0], allow_pickle=False)
        return {
            "video": torch.from_numpy(array.copy()).permute(0, 3, 1, 2),
            "label": row.label_id, "clip_id": row.clip_id,
            "group_id": row.group_id, "source": row.source_collection,
            "duration": row.clip_duration_sec,
        }
