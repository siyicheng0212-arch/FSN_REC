#!/usr/bin/env python3
"""Build strict patient/practitioner-independent manifests from a completed identity map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def subset_near_target(group_sizes: dict[str, int], target: int, seed_scope: str) -> set[str]:
    groups = sorted(group_sizes, key=lambda group: stable_hash(f"{seed_scope}|{group}"))
    total = sum(group_sizes.values())
    if len(groups) < 2 or target <= 0:
        return set()
    target = max(1, min(total - 1, target))
    reachable = [False] * (total + 1)
    prev_sum = [-1] * (total + 1)
    prev_group = [-1] * (total + 1)
    reachable[0] = True
    for idx, group in enumerate(groups):
        size = group_sizes[group]
        for subtotal in range(total - size, -1, -1):
            new_sum = subtotal + size
            if reachable[subtotal] and not reachable[new_sum]:
                reachable[new_sum] = True
                prev_sum[new_sum] = subtotal
                prev_group[new_sum] = idx
    candidates = [value for value in range(1, total) if reachable[value]]
    if not candidates:
        return set()
    value = min(candidates, key=lambda item: (abs(item - target), item > target, item))
    selected: set[str] = set()
    while value:
        idx = prev_group[value]
        selected.add(groups[idx])
        value = prev_sum[value]
    return selected


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_groups(
    record_ids: list[str], identity: dict[str, dict[str, str]], protocol: str
) -> dict[str, str]:
    if protocol == "patient":
        return {record_id: f"patient:{identity[record_id]['patient_id']}" for record_id in record_ids}
    if protocol == "practitioner":
        return {
            record_id: f"practitioner:{identity[record_id]['practitioner_id']}" for record_id in record_ids
        }
    dsu = DisjointSet(record_ids)
    first_by_token: dict[str, str] = {}
    for record_id in record_ids:
        row = identity[record_id]
        tokens = [
            f"patient:{row['patient_id']}",
            f"practitioner:{row['practitioner_id']}",
        ]
        if row.get("treatment_session_id"):
            tokens.append(f"session:{row['treatment_session_id']}")
        for token in tokens:
            if token in first_by_token:
                dsu.union(record_id, first_by_token[token])
            else:
                first_by_token[token] = record_id
    components: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        components[dsu.find(record_id)].append(record_id)
    component_id = {
        record_id: "joint:" + stable_hash("|".join(sorted(members)))[:20]
        for members in components.values()
        for record_id in members
    }
    return component_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("processed_dataset/meta_data/records.jsonl"))
    parser.add_argument("--clips", type=Path, default=Path("processed_dataset/manifests/clips_all.jsonl"))
    parser.add_argument(
        "--identity-map", type=Path, default=Path("processed_dataset/meta_data/identity_map_template.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("processed_dataset/manifests/identity_holdout"))
    parser.add_argument("--sources", default="lishui,menzhen")
    parser.add_argument("--protocol", choices=("patient", "practitioner", "joint", "all"), default="all")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", default="fsn-identity-holdout-v1")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sources = {value.strip() for value in args.sources.split(",") if value.strip()}
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("invalid val/test ratios")

    records = [
        row for row in read_jsonl(args.records)
        if row["eligibility_status"] == "included" and row["source_collection"] in sources
    ]
    clips = [row for row in read_jsonl(args.clips) if row["source_collection"] in sources]
    with args.identity_map.open(encoding="utf-8-sig", newline="") as handle:
        identity = {row["record_id"]: row for row in csv.DictReader(handle)}
    missing_rows = [row["record_id"] for row in records if row["record_id"] not in identity]
    missing_patient = [
        row["record_id"] for row in records
        if row["record_id"] not in identity or not identity[row["record_id"]].get("patient_id", "").strip()
    ]
    missing_practitioner = [
        row["record_id"] for row in records
        if row["record_id"] not in identity or not identity[row["record_id"]].get("practitioner_id", "").strip()
    ]
    audit = {
        "sources": sorted(sources),
        "eligible_records": len(records),
        "clips": len(clips),
        "missing_identity_rows": len(missing_rows),
        "missing_patient_id": len(missing_patient),
        "missing_practitioner_id": len(missing_practitioner),
        "patient_protocol_ready": not missing_patient,
        "practitioner_protocol_ready": not missing_practitioner,
        "joint_protocol_ready": not missing_patient and not missing_practitioner,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        return

    protocols = ["patient", "practitioner", "joint"] if args.protocol == "all" else [args.protocol]
    blockers = {
        "patient": missing_patient,
        "practitioner": missing_practitioner,
        "joint": sorted(set(missing_patient + missing_practitioner)),
    }
    blocked = [protocol for protocol in protocols if blockers[protocol]]
    if blocked:
        raise SystemExit(
            "Identity map is incomplete for protocol(s): " + ", ".join(blocked)
            + ". Fill the CSV or run --audit-only."
        )
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"output exists: {args.output}; use --force")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    record_ids = [row["record_id"] for row in records]
    clips_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        clips_by_record[clip["record_id"]].append(clip)

    for protocol in protocols:
        group_by_record = build_groups(record_ids, identity, protocol)
        group_sizes: dict[str, int] = defaultdict(int)
        for record_id, group in group_by_record.items():
            group_sizes[group] += len(clips_by_record[record_id])
        if len(group_sizes) < 3:
            raise SystemExit(
                f"Protocol {protocol} has only {len(group_sizes)} independent group(s); "
                "at least three are required for train/val/test."
            )
        total = sum(group_sizes.values())
        test_groups = subset_near_target(dict(group_sizes), round(total * args.test_ratio), f"{args.seed}|{protocol}|test")
        remaining = {group: size for group, size in group_sizes.items() if group not in test_groups}
        val_groups = subset_near_target(remaining, round(total * args.val_ratio), f"{args.seed}|{protocol}|val")
        if not test_groups or not val_groups:
            raise SystemExit(f"Protocol {protocol} cannot form non-empty validation and test group sets.")
        output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for clip in clips:
            group = group_by_record[clip["record_id"]]
            split = "test" if group in test_groups else ("val" if group in val_groups else "train")
            output_rows[split].append({**clip, "split": split, "identity_group_id": group, "protocol": protocol})
        protocol_root = args.output / protocol
        protocol_root.mkdir()
        for split, rows in output_rows.items():
            write_jsonl(protocol_root / f"{split}.jsonl", rows)
        summary = {
            "protocol": protocol,
            "sources": sorted(sources),
            "seed": args.seed,
            "ratio_basis": "supervised_clip_count",
            "clips_by_split": {split: len(rows) for split, rows in output_rows.items()},
            "groups_by_split": {
                split: len({row["identity_group_id"] for row in rows}) for split, rows in output_rows.items()
            },
            "identity_groups_in_multiple_splits": sorted(
                {row["identity_group_id"] for row in output_rows["train"]}
                & ({row["identity_group_id"] for row in output_rows["val"]}
                   | {row["identity_group_id"] for row in output_rows["test"]})
                | ({row["identity_group_id"] for row in output_rows["val"]}
                   & {row["identity_group_id"] for row in output_rows["test"]})
            ),
        }
        (protocol_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
