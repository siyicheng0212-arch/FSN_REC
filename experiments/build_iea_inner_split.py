#!/usr/bin/env python3
"""Derive a source-stratified, video-group-disjoint inner split from train only.

The original ``split`` field stays ``train`` so the frozen train-cache paths
remain valid. The output filenames and ``inner_split_role`` define the new
logical train/validation roles. No original validation/test row is read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiments.pilot_data import load_pilot_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_groups(
    rows: list[dict], *, fraction: float, seed: int, min_per_class: int
) -> tuple[set[str], int]:
    if not 0 < fraction < 0.5:
        raise ValueError("inner validation fraction must be between 0 and 0.5")
    groups: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("inner split input must contain original train rows only")
        group, source = row["group_id"], row["source_collection"]
        groups[group].append(row)
        by_source[source].add(group)
    if sum(len(values) for values in by_source.values()) != len(groups):
        raise ValueError("one group spans multiple source collections")
    for attempt in range(1000):
        validation: set[str] = set()
        for source, group_ids in sorted(by_source.items()):
            ordered = sorted(
                group_ids,
                key=lambda group: hashlib.sha256(
                    f"{seed + attempt}|{source}|{group}".encode()
                ).hexdigest(),
            )
            count = min(len(ordered) - 1, max(1, round(len(ordered) * fraction)))
            if count <= 0:
                raise ValueError(f"source {source} has too few groups")
            validation.update(ordered[:count])
        val_counts = Counter(
            row["label_id"] for row in rows if row["group_id"] in validation
        )
        train_counts = Counter(
            row["label_id"] for row in rows if row["group_id"] not in validation
        )
        if all(
            val_counts[label] >= min_per_class
            and train_counts[label] >= min_per_class
            for label in range(7)
        ):
            return validation, seed + attempt
    raise ValueError("could not make a source/group split with every class represented")


def run(args: argparse.Namespace) -> dict:
    source = args.source_manifest.resolve()
    if sha256(source) != args.expected_sha256:
        raise ValueError("frozen source train-manifest hash mismatch")
    records = load_pilot_manifest(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len(records):
        raise ValueError("record count changed while reading manifest")
    validation, effective_seed = choose_groups(
        rows, fraction=args.val_fraction, seed=args.seed,
        min_per_class=args.min_per_class,
    )
    selected = {
        "train": [row for row in rows if row["group_id"] not in validation],
        "val": [row for row in rows if row["group_id"] in validation],
    }
    group_sets = {
        split: {row["group_id"] for row in items}
        for split, items in selected.items()
    }
    video_sets = {
        split: {row["video_path"] for row in items}
        for split, items in selected.items()
    }
    if group_sets["train"] & group_sets["val"] or video_sets["train"] & video_sets["val"]:
        raise RuntimeError("video-level leakage in inner split")
    if sum(map(len, selected.values())) != len(rows):
        raise RuntimeError("inner split dropped source clips")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"inner split output already exists: {output}")
    output.mkdir(parents=True)
    for split, items in selected.items():
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps({**row, "inner_split_role": split}, ensure_ascii=False)
                    + "\n" for row in items),
            encoding="utf-8",
        )
    summary = {
        "schema_version": "fsn-iea-inner-split-1.0",
        "source_train_sha256": args.expected_sha256,
        "requested_seed": args.seed,
        "effective_seed": effective_seed,
        "val_fraction_by_source_groups": args.val_fraction,
        "original_split_field_preserved_for_cache": "train",
        "counts": {split: len(items) for split, items in selected.items()},
        "group_counts": {split: len(groups) for split, groups in group_sets.items()},
        "class_counts": {
            split: dict(sorted(Counter(row["label_id"] for row in items).items()))
            for split, items in selected.items()
        },
        "source_counts": {
            split: dict(sorted(Counter(row["source_collection"] for row in items).items()))
            for split, items in selected.items()
        },
        "manifest_sha256": {
            split: sha256(output / f"{split}.jsonl") for split in selected
        },
        "group_overlap": [],
        "video_path_overlap": [],
        "independent_test_used": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
