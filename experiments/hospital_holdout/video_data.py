"""Raw-video, timestamp-aware single-view data; no upscaled old square cache.

Spatial transform matches the pinned VideoMAEv2 validation transform on RGB
numpy frames. Temporal sampling remains an explicitly declared FSN uniform view,
not the upstream K710 16x5x3 benchmark. Menzhen888 is development validation.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.hospital_holdout.prepare import SCHEMA, EXPECTED_COUNTS, rows, sha, encoded

PREPROCESS = {
    "schema": "fsn-raw-rgb16-short256-centre224-imagenet-v1",
    "frames": 16, "RGB": True, "short_side": 256, "crop": 224,
    "resize": "opencv_INTER_LINEAR_aspect_preserving",
    "crop_position": "int(round((dimension-crop)/2))",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "temporal": "nearest_PTS_centres_to_16_uniform_bin_centres_inside_annotation",
    "views": 1, "official_spatial_validation_transform": True,
    "official_full_training_or_multiview_recipe": False,
}


def spatial_rgb(frames, crop=224, short_side=256):
    import cv2
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 \
            or not len(frames) or min(frames.shape[1:3]) <= 0 or crop > short_side:
        raise ValueError("need nonempty original uint8 RGB [T,H,W,3] frames")
    h, w = frames.shape[1:3]
    nh, nw = (int(short_side * h / w), short_side) if w < h \
        else (short_side, int(short_side * w / h))
    y, x = int(round((nh - crop) / 2)), int(round((nw - crop) / 2))
    result = []
    for frame in frames:
        resized = frame if (h, w) == (nh, nw) else cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        result.append(resized[y:y + crop, x:x + crop])
    return np.ascontiguousarray(np.stack(result))


def normalise_rgb(frames, mean=PREPROCESS["mean"], std=PREPROCESS["std"]):
    value = torch.from_numpy(np.array(frames, copy=True)).permute(3, 0, 1, 2).float() / 255
    return (value - value.new_tensor(mean)[:, None, None, None]) / \
        value.new_tensor(std)[:, None, None, None]


def select_indices(pts, start, end, count=16):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all() \
            or not np.isfinite([start, end]).all() or start < 0 or end <= start \
            or count < 1 or np.any(pts[:, 1] <= pts[:, 0]):
        raise ValueError("invalid PTS or annotation interval")
    centres = pts.mean(axis=1)
    if np.any(np.diff(centres) < 0):
        raise ValueError("video PTS must be monotonic")
    eligible = np.flatnonzero((centres >= start) & (centres < end))
    if not len(eligible):
        raise ValueError("annotation interval has no actual frame centre; do not invent a frame")
    targets = start + (np.arange(count) + .5) * (end - start) / count
    local = centres[eligible]
    right = np.searchsorted(local, targets).clip(0, len(local) - 1)
    left = (right - 1).clip(0, len(local) - 1)
    chosen = np.where(abs(local[left] - targets) <= abs(local[right] - targets), left, right)
    return eligible[chosen]


def read_protocol(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol.get("schema") != SCHEMA or protocol.get("counts") != EXPECTED_COUNTS \
            or protocol.get("timestamp_audit_passed") is not True \
            or protocol.get("selection") != "best_finetune_Menzhen888_macro_f1" \
            or protocol.get("R_D_enabled") is not False:
        raise ValueError("incorrect hospital holdout protocol")
    return protocol


def manifest_rows(directory, split):
    if split not in {"train", "val"}:
        raise ValueError("this protocol reads train and Menzhen development validation only")
    protocol = read_protocol(directory)
    path = Path(directory) / f"{split}.jsonl"
    if sha(path) != protocol["manifest_sha256"][split]:
        raise ValueError("sealed manifest SHA mismatch")
    subset = rows(path)
    if len(subset) != EXPECTED_COUNTS[split] or len({r["clip_id"] for r in subset}) != len(subset):
        raise ValueError("sealed split count/unique-ID mismatch")
    if any(r["split"] != split or r.get("time_unit_verified") is not True for r in subset):
        raise ValueError("split or timestamp provenance mismatch")
    if any((r["source_collection"].casefold() == "menzhen") != (split == "val") for r in subset):
        raise ValueError("Menzhen must appear only in development validation")
    return subset, protocol


def make_cache(protocol_dir, split, output):
    # Import only in explicit CPU preparation, not unit tests or read-only plans.
    from decord import VideoReader, cpu
    subset, protocol = manifest_rows(protocol_dir, split)
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    for protected in (repo, Path(protocol_dir).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("cache must be private, outside checkout and manifests")
    if output.exists():
        raise ValueError("new cache output must not exist; partial failures are not auto-resumed")
    for p in {r["video_path"] for r in subset}:
        if sha(p) != protocol["video_sha256"][p]:
            raise ValueError("original video changed after timestamp/duplicate audit")
    output.mkdir(parents=True, exist_ok=False)
    readers, inventory = OrderedDict(), []
    for row in subset:
        path = row["video_path"]
        if path not in readers:
            reader = VideoReader(path, num_threads=1, ctx=cpu(0))
            timestamps = reader.get_frame_timestamp(np.arange(len(reader)))
            readers[path] = (reader, timestamps)
            if len(readers) > 4:
                readers.popitem(last=False)
        reader, pts = readers[path]
        readers.move_to_end(path)
        indices = select_indices(pts, row["clip_start_sec"], row["clip_end_sec"])
        rgb = reader.get_batch(indices.tolist()).asnumpy()
        array = spatial_rgb(rgb)
        name = hashlib.sha256(row["clip_id"].encode()).hexdigest() + ".npy"
        with (output / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        inventory.append({"clip_id": row["clip_id"], "file": name, "sha256": sha(output / name),
                          "label_id": row["label_id"], "frame_indices": indices.tolist(),
                          "actual_frame_pts_sec": np.asarray(pts)[indices].tolist(),
                          "repeated_frames": 16 - len(set(indices.tolist())),
                          "original_shape": list(rgb.shape), "cached_shape": list(array.shape)})
        if len(inventory) % 100 == 0:
            print(json.dumps({"split": split, "cached": len(inventory), "total": len(subset)}), flush=True)
    with (output / "private_index.json").open("xb") as stream:
        stream.write(encoded(inventory))
    seal = {"status": "complete", "split": split, "count": len(subset),
            "manifest_sha256": protocol["manifest_sha256"][split], "preprocessing": PREPROCESS,
            "private_index_sha256": sha(output / "private_index.json"),
            "protocol_sha256": sha(Path(protocol_dir) / "protocol.json"),
            "repeated_frame_clips": sum(r["repeated_frames"] > 0 for r in inventory)}
    with (output / "cache_seal.json").open("xb") as stream:
        stream.write(encoded(seal))
    return seal


class HospitalDataset(Dataset):
    def __init__(self, protocol_dir, cache_dir, split):
        self.raw_rows, protocol = manifest_rows(protocol_dir, split)
        self.split, self.cache_dir = split, Path(cache_dir)
        self.seal = json.loads((self.cache_dir / "cache_seal.json").read_text())
        index_path = self.cache_dir / "private_index.json"
        if self.seal.get("status") != "complete" or self.seal.get("split") != split \
                or self.seal.get("count") != EXPECTED_COUNTS[split] \
                or self.seal.get("manifest_sha256") != protocol["manifest_sha256"][split] \
                or self.seal.get("preprocessing") != PREPROCESS \
                or self.seal.get("protocol_sha256") != sha(Path(protocol_dir) / "protocol.json") \
                or self.seal.get("private_index_sha256") != sha(index_path):
            raise ValueError("cache seal does not match corrected hospital protocol")
        inventory = json.loads(index_path.read_text())
        indexed = {r["clip_id"]: r for r in inventory}
        if len(indexed) != len(self.raw_rows) or len(inventory) != len(indexed) \
                or set(indexed) != {r["clip_id"] for r in self.raw_rows}:
            raise ValueError("cache does not cover every split clip exactly once")
        self.mapping = []
        for row in self.raw_rows:
            item = indexed[row["clip_id"]]
            filename = hashlib.sha256(row["clip_id"].encode()).hexdigest() + ".npy"
            if item["file"] != filename or item["label_id"] != row["label_id"] \
                    or sha(self.cache_dir / filename) != item["sha256"]:
                raise ValueError("cache content/label/path mismatch")
            self.mapping.append(item)

    def __len__(self):
        return len(self.raw_rows)

    def __getitem__(self, index):
        row, item = self.raw_rows[index], self.mapping[index]
        array = np.load(self.cache_dir / item["file"], allow_pickle=False)
        if array.dtype != np.uint8 or array.shape != (16, 224, 224, 3):
            raise ValueError("raw-video cache has incorrect RGB geometry")
        return {"video": normalise_rgb(array), "label": row["label_id"],
                "clip_id": row["clip_id"], "group_id": row["group_id"],
                "source": row["source_collection"], "duration": row["clip_duration_sec"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-dir", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    subset, protocol = manifest_rows(args.protocol_dir, args.split)
    print(json.dumps({"split": args.split, "count": len(subset), "preprocessing": PREPROCESS,
                      "read_only": not args.execute}, ensure_ascii=False))
    if args.execute:
        print(json.dumps(make_cache(args.protocol_dir, args.split, args.output)))


if __name__ == "__main__":
    main()
