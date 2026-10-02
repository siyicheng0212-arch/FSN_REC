"""Reuse sealed full7372/823 frozen-A artifacts; never re-split or read test."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from experiments.aligned_protocol import EXPECTED_COUNTS
from experiments.relation.data import load_edges, load_feature_index, read_jsonl, sha256_file
from experiments.relation.protocol import SOURCE_FILES, validate_source_artifacts, relation_source_hashes


@dataclass
class Bundle:
    index: object
    metadata: dict
    chains: dict
    edges: list
    fingerprint: dict


def code_hashes():
    root = Path(__file__).resolve().parents[2]
    return relation_source_hashes() | {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def validate_chains(chains, metadata, split):
    """Only original source-builder chains enter supervised TCN training."""
    if split not in {"train", "val"}:
        raise ValueError("only train/val are supported")
    expected = {key for key, row in metadata.items() if row["split"] == split}
    seen = set()
    for chain in chains:
        ids = chain.get("ordered_clip_ids")
        eligible = chain.get("eligible")
        if (chain.get("split") != split or chain.get("synthetic_chain")
                or not isinstance(ids, list) or not ids or not isinstance(eligible, list)
                or len(eligible) != len(ids) - 1 or any(type(x) is not bool for x in eligible)):
            raise ValueError("invalid original candidate chain")
        for clip in ids:
            if clip not in expected or clip in seen:
                raise ValueError("cross-split, unknown or repeated chain clip")
            seen.add(clip)
            row = metadata[clip]
            for key in ("group_id", "source_kind", "source_video_key", "timebase_key", "recording_key"):
                if chain.get(key) != row.get(key):
                    raise ValueError(f"original chain crosses or mismatches {key}")
        for i, is_open in enumerate(eligible):
            left, right = (metadata[ids[j]] for j in (i, i + 1))
            if right["clip_start_sec"] < left["clip_start_sec"]:
                raise ValueError("original chain is not ordered")
            if is_open and (right["clip_start_sec"] < left["clip_end_sec"]
                            or right["clip_start_sec"] - left["clip_end_sec"] > 5.0):
                raise ValueError("eligible edge violates frozen nonoverlap/gap rules")
    if seen != expected:
        raise ValueError("each split clip must occur exactly once")


def load_bundle(feature_index, source_protocol_dir):
    index = load_feature_index(feature_index)
    directory = Path(source_protocol_dir).resolve()
    audit = validate_source_artifacts(directory, index)
    if audit.get("max_gap_seconds") != 5.0:
        raise ValueError("this minimal protocol retains the sealed 5-second cutoff")
    if any("feature_sha256" not in row for row in index.clips.values()):
        raise ValueError("every frozen feature must carry its SHA256")
    metadata = {row["clip_id"]: row for row in read_jsonl(directory / "manifest_metadata.jsonl")}
    from .hard_negatives import validate_metadata
    validate_metadata(metadata)
    counts = {s: sum(row["split"] == s for row in metadata.values()) for s in ("train", "val")}
    if counts != EXPECTED_COUNTS:
        raise ValueError("FSN-TCN requires all train7372/val823 clips")
    chains = {s: read_jsonl(directory / f"chains_{s}.jsonl") for s in ("train", "val")}
    for split in chains:
        validate_chains(chains[split], metadata, split)
    edges = load_edges(directory / "edges.jsonl", index)
    candidate_pairs = {(chain["split"], left, right) for split in chains for chain in chains[split]
                       for left, right, valid in zip(chain["ordered_clip_ids"], chain["ordered_clip_ids"][1:], chain["eligible"])
                       if valid}
    if candidate_pairs != {(edge["split"], edge["left_clip_id"], edge["right_clip_id"]) for edge in edges}:
        raise ValueError("sealed source edges and candidate chains disagree")
    fingerprint = {
        "feature_index_sha256": sha256_file(feature_index),
        "A_checkpoint_sha256": index.checkpoint_sha256,
        "source_audit_sha256": sha256_file(directory / "audit.json"),
        "source_files_sha256": {name: sha256_file(directory / name) for name in SOURCE_FILES},
        "manifest_sha256": audit["manifest_sha256"], "counts": counts,
        "data_protocol": audit["data_protocol"],
        "evaluation_role": "val823_development_not_independent_test",
    }
    return Bundle(index, metadata, chains, edges, fingerprint)


def load_chain(bundle, chain, device="cpu"):
    ids = chain["ordered_clip_ids"]
    features, logits, labels = [], [], []
    for clip in ids:
        arrays = bundle.index.read(clip)
        features.append(np.concatenate((arrays["global_tokens"].mean(0), arrays["local_tokens"].mean(0))))
        logits.append(arrays["logits"])
        labels.append(bundle.metadata[clip]["label_id"])
    return (torch.from_numpy(np.stack(features)).to(device),
            torch.from_numpy(np.stack(logits)).to(device),
            torch.tensor(labels, dtype=torch.long, device=device))


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fresh_output(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    if path.is_relative_to(root):
        raise ValueError("private outputs must be outside the code worktree")
    path.mkdir(parents=True, exist_ok=False)
    return path
