"""Verify source-policy artifacts and their correspondence to frozen A evidence."""
import json
import math
from pathlib import Path

from experiments.aligned_protocol import DATA_PROTOCOL, validate_manifest_protocol
from .data import read_jsonl, sha256_file

SOURCE_POLICY = "source_rule_v1"
SOURCE_FILES = ("edges.jsonl", "chains_train.jsonl", "chains_val.jsonl", "manifest_metadata.jsonl")


def relation_source_hashes():
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("source_map.json")]
    files += [root / "experiments" / name for name in (
        "model_wrappers.py", "aligned_data.py", "aligned_protocol.py", "full_data.py", "pilot_data.py")]
    files += sorted((root / "models" / "Uni-AdaFocus-TSM-FSN").rglob("*.py"))
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def edge_label_policy(edges):
    origins = {row.get("label_origin", "manual_supplied") for row in edges}
    if len(origins) != 1 or next(iter(origins)) not in {"manual_supplied", SOURCE_POLICY}:
        raise ValueError("one run must use one explicit supported edge-label policy")
    return next(iter(origins))


def validate_source_artifacts(directory, index=None, edges_path=None, *, enforce_full=True):
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    if audit.get("label_policy") != SOURCE_POLICY:
        raise ValueError("expected explicit source_rule_v1 artifact policy")
    if enforce_full:
        if audit.get("data_protocol") != DATA_PROTOCOL:
            raise ValueError("source artifacts are not from fixed train7372/val823")
        validate_manifest_protocol(audit.get("counts"), audit.get("manifest_sha256"))
    seals = audit.get("files_sha256", {})
    if any(seals.get(name) != sha256_file(directory / name) for name in SOURCE_FILES):
        raise ValueError("source-policy artifact bytes changed or were not sealed")
    if edges_path is not None and sha256_file(edges_path) != seals["edges.jsonl"]:
        raise ValueError("edge file differs from the fixed source-policy artifacts")
    if index is not None:
        rows = read_jsonl(directory / "manifest_metadata.jsonl")
        metadata = {}
        for row in rows:
            clip_id = row.get("clip_id")
            if clip_id not in index.clips or clip_id in metadata:
                raise ValueError("metadata must match feature-index clips exactly once")
            feature = index.clips[clip_id]
            for key in ("split", "group_id", "label_id"):
                if row.get(key) != feature.get(key):
                    raise ValueError(f"source metadata and A index differ: {key}")
            for key in ("source_collection", "source_video_key", "clip_start_sec", "clip_end_sec", "duration"):
                if key not in feature or row.get(key) != feature[key]:
                    raise ValueError(f"missing or mismatched A metadata: {key}; re-export with this code")
            if row.get("source_kind") not in {"clinical", "network"}:
                raise ValueError("unknown source kind")
            duration = row.get("duration")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid clip duration")
            metadata[clip_id] = row
        if metadata.keys() != index.clips.keys():
            raise ValueError("source metadata does not cover the full A evidence index")
        export_seal_path = index.path.parent / "protocol.json"
        if not export_seal_path.exists():
            raise ValueError("formal source-policy run requires a sealed canonical feature export")
        export_seal = json.loads(export_seal_path.read_text())
        if not export_seal.get("A_frozen") or export_seal.get("status") != "complete":
            raise ValueError("A export is not complete and frozen")
        exported_manifest_sha = {s: export_seal.get(s + "_manifest_sha256") for s in ("train", "val")}
        if exported_manifest_sha != audit.get("manifest_sha256"):
            raise ValueError("feature export and source labels use different original manifests")
    return audit
