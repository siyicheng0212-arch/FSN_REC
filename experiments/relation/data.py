"""Audited offline evidence and explicit relation-label policies.

C = trusted forward process connection; D = no trusted connection; U = unknown.
U is excluded from supervised BCE rather than being silently relabeled D.
The source builder explicitly generates source_rule_v1 labels under the user's
declared policy. This loader never infers labels from action equality.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EVIDENCE_KEYS = ("global_tokens", "local_tokens", "global_positions", "local_positions")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _sha(row, key):
    value = _text(row, key).lower()
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"{key} must be a 64-digit SHA256")
    return value


def _verify_feature(row):
    if "feature_sha256" in row and sha256_file(row["resolved_feature_path"]) != row["feature_sha256"]:
        raise ValueError(f"{row['clip_id']}: feature_sha256 does not match file bytes")


def load_feature(path):
    """Read numeric arrays only; logits are retained for D, never fed to R."""
    with np.load(path, allow_pickle=False) as stored:
        missing = [key for key in (*EVIDENCE_KEYS, "logits") if key not in stored]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        arrays = {key: np.array(stored[key], copy=True) for key in (*EVIDENCE_KEYS, "logits")}
    for key, value in arrays.items():
        if value.dtype.kind not in "fi" or not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} must contain finite numeric values")
        with np.errstate(over="ignore"):
            arrays[key] = value.astype(np.float32)
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path}: {key} exceeds float32 range")
    if arrays["logits"].shape != (7,):
        raise ValueError(f"{path}: logits must have shape [7]")
    for branch in ("global", "local"):
        tokens, positions = arrays[f"{branch}_tokens"], arrays[f"{branch}_positions"]
        if tokens.ndim != 2 or min(tokens.shape) < 1 or positions.shape != (tokens.shape[0],):
            raise ValueError(f"{path}: {branch} tokens/positions shapes are invalid")
        if np.any(positions < 0) or np.any(positions > 1) or np.any(np.diff(positions) < 0):
            raise ValueError(f"{path}: {branch} positions must be ordered in [0,1]")
    return arrays


@dataclass
class FeatureIndex:
    path: Path
    clips: dict
    checkpoint_sha256: str
    shapes: dict

    @property
    def global_dim(self):
        return self.shapes["global_tokens"][1]

    @property
    def local_dim(self):
        return self.shapes["local_tokens"][1]

    def read(self, clip_id):
        _verify_feature(self.clips[clip_id])
        arrays = load_feature(self.clips[clip_id]["resolved_feature_path"])
        if any(arrays[key].shape != self.shapes[key] for key in EVIDENCE_KEYS):
            raise ValueError(f"{clip_id}: feature shapes changed after preflight")
        return arrays


def load_feature_index(path):
    """Validate every file and all groups before any training/output creation."""
    path = Path(path).resolve()
    export_protocol = path.parent / "protocol.json"
    if path.name == "index.jsonl" and export_protocol.exists():
        # Generic manually authored feature indexes remain supported. When the
        # canonical exporter wrote a seal, an interrupted export is not ready.
        seal = json.loads(export_protocol.read_text())
        if seal.get("A_frozen") is True:
            if seal.get("status") != "complete" or seal.get("index_sha256") != sha256_file(path):
                raise ValueError("canonical feature export is incomplete or its index changed")
    clips, hashes, groups, shapes = {}, set(), {"train": set(), "val": set()}, None
    feature_paths = set()
    for raw in read_jsonl(path):
        row = dict(raw)
        clip_id, group_id = _text(row, "clip_id"), _text(row, "group_id")
        if clip_id in clips:
            raise ValueError(f"duplicate clip_id: {clip_id}")
        split = row.get("split")
        if not isinstance(split, str) or split not in groups:
            raise ValueError("index split must be train or val")
        checkpoint = _sha(row, "checkpoint_sha256")
        relative = Path(_text(row, "feature_path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("feature_path must be relative to the index directory without '..'")
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(path.parent):
            raise ValueError("feature_path escapes index directory")
        if resolved in feature_paths:
            raise ValueError("different clips must not reference the same resolved feature file")
        feature_paths.add(resolved)
        row["resolved_feature_path"] = resolved
        if "feature_sha256" in row:
            row["feature_sha256"] = _sha(row, "feature_sha256")
        _verify_feature(row)
        arrays = load_feature(resolved)
        current = {key: arrays[key].shape for key in EVIDENCE_KEYS}
        if shapes is not None and current != shapes:
            raise ValueError("all feature shapes must match for this fixed-shape prototype")
        shapes = current
        clips[clip_id] = row
        hashes.add(checkpoint)
        groups[split].add(group_id)
    if len(hashes) != 1:
        raise ValueError("all evidence must use the same frozen A checkpoint")
    if groups["train"] & groups["val"]:
        raise ValueError("train/val group leakage in feature index")
    return FeatureIndex(path, clips, hashes.pop(), shapes)


def load_edges(path, index):
    edges, identifiers, pairs = [], set(), set()
    for raw in read_jsonl(path):
        row = dict(raw)
        edge_id = _text(row, "edge_id")
        if edge_id in identifiers:
            raise ValueError(f"duplicate edge_id: {edge_id}")
        if not isinstance(row.get("status"), str) or row["status"] not in {"C", "D", "U"}:
            raise ValueError("edge status must be C, D or U")
        if not isinstance(row.get("split"), str) or row["split"] not in {"train", "val"}:
            raise ValueError("edge split must be train or val")
        if row.get("label_origin") == "source_rule_v1":
            kind = row.get("source_kind")
            if kind not in {"clinical", "network"} or row["status"] != {"clinical": "C", "network": "D"}[kind]:
                raise ValueError("source-policy label disagrees with its declared source kind")
        endpoints = [_text(row, key) for key in ("left_clip_id", "right_clip_id")]
        if tuple(endpoints) in pairs:
            raise ValueError("duplicate directed pair; consolidate annotation before training")
        if endpoints[0] == endpoints[1]:
            raise ValueError("a relation edge cannot join a clip to itself")
        for clip_id in endpoints:
            if clip_id not in index.clips:
                raise ValueError(f"edge {edge_id}: unknown endpoint {clip_id}")
            if index.clips[clip_id]["split"] != row["split"]:
                raise ValueError(f"edge {edge_id}: endpoint split does not match edge split")
        if row["status"] == "C" and index.clips[endpoints[0]]["group_id"] != index.clips[endpoints[1]]["group_id"]:
            raise ValueError("C edges must stay within one declared recording/episode group")
        identifiers.add(edge_id)
        pairs.add(tuple(endpoints))
        edges.append(row)
    return edges


def audit_edges(edges):
    return {split: {status: sum(row["split"] == split and row["status"] == status
                                for row in edges) for status in ("C", "D", "U")}
            for split in ("train", "val")}


class RelationDataset(Dataset):
    """C/D targets are 1/0. U has sentinel -1 and is audit-only."""

    def __init__(self, index, edges, split, statuses=("C", "D")):
        if split not in {"train", "val"} or not set(statuses) <= {"C", "D", "U"}:
            raise ValueError("invalid split/status selection")
        self.index = index
        self.rows = [row for row in edges if row["split"] == split and row["status"] in statuses]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        evidence = []
        for name in ("left_clip_id", "right_clip_id"):
            arrays = self.index.read(row[name])
            evidence.append({key: torch.from_numpy(arrays[key]) for key in EVIDENCE_KEYS})
        return {"left": evidence[0], "right": evidence[1],
                "target": torch.tensor({"C": 1., "D": 0., "U": -1.}[row["status"]]),
                "edge_id": row["edge_id"], "status": row["status"]}
