#!/usr/bin/env python3
"""Derive a two-way train/validation protocol without duplicating frame arrays.

The original train and validation manifests become the new training manifest.
The original test manifest becomes validation.  This protocol deliberately has
no independent test set and must not be reported as a held-out test result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL = "train-plus-original-val__original-test-as-validation-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap(rows: list[dict[str, Any]], new_split: str) -> list[dict[str, Any]]:
    remapped = []
    for row in rows:
        original_split = row["split"]
        remapped.append({
            **row,
            "split": new_split,
            "original_split": original_split,
            "derived_protocol": PROTOCOL,
        })
    return remapped


def link_cache(
    rows: list[dict[str, Any]], source_cache: Path, output_cache: Path
) -> dict[str, int]:
    counts = {"linked_arrays": 0, "reused_arrays": 0, "metadata_written": 0}
    for row in rows:
        clip_id = row["clip_id"]
        original_split = row["original_split"]
        new_split = row["split"]
        source_array = source_cache / original_split / f"{clip_id}.npy"
        source_metadata = source_cache / original_split / f"{clip_id}.json"
        if not source_array.is_file() or not source_metadata.is_file():
            raise FileNotFoundError(f"missing source cache for {clip_id}")
        destination_dir = output_cache / new_split
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_array = destination_dir / source_array.name
        destination_metadata = destination_dir / source_metadata.name
        if destination_array.exists():
            if destination_array.stat().st_size != source_array.stat().st_size:
                raise RuntimeError(f"conflicting destination cache for {clip_id}")
            counts["reused_arrays"] += 1
        else:
            os.link(source_array, destination_array)
            counts["linked_arrays"] += 1
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        metadata.update({
            "split": new_split,
            "original_split": original_split,
            "derived_protocol": PROTOCOL,
        })
        destination_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts["metadata_written"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifests", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-manifests", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_manifests.resolve()
    output = args.output_manifests.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = {
        split: read_jsonl(source / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    train_rows = remap(original["train"] + original["val"], "train")
    val_rows = remap(original["test"], "val")
    train_groups = {row["group_id"] for row in train_rows}
    val_groups = {row["group_id"] for row in val_rows}
    overlap = sorted(train_groups & val_groups)
    if overlap:
        raise RuntimeError(f"group leakage in derived protocol: {overlap[:10]}")

    train_path, val_path = output / "train.jsonl", output / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    cache_counts = link_cache(
        train_rows + val_rows, args.source_cache.resolve(), args.output_cache.resolve()
    )
    summary = {
        "schema_version": "fsn-derived-protocol-1.0",
        "protocol": PROTOCOL,
        "warning": "validation is the original test split; no independent test exists",
        "source_manifest_sha256": {
            split: sha256_file(source / f"{split}.jsonl") for split in original
        },
        "counts": {
            "original_train": len(original["train"]),
            "original_val": len(original["val"]),
            "original_test": len(original["test"]),
            "derived_train": len(train_rows),
            "derived_val": len(val_rows),
        },
        "class_counts": {
            split: dict(sorted(Counter(row["label_id"] for row in rows).items()))
            for split, rows in (("train", train_rows), ("val", val_rows))
        },
        "group_overlap": overlap,
        "manifest_sha256": {
            "train": sha256_file(train_path),
            "val": sha256_file(val_path),
        },
        "cache": cache_counts,
    }
    (output / "protocol_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
