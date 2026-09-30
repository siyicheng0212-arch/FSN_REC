"""Freeze private JSONL manifests after a strict, model-neutral leakage audit.

No split is generated or remapped. Relative video paths require an explicit
video_root. This module verifies manifest metadata, not video pixels or caches;
the trainer must separately validate its existing cache for every record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from experiments.pilot_data import ClipRecord, EXPECTED_LABELS

PROTOCOL_VERSION = "fsn-cvm-frozen-protocol-1.0"
_SPLITS = ("train", "val", "test")
_CLIP_ID = re.compile(r"clip-[A-Za-z0-9_-]{1,120}\Z")
_SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")


class ProtocolError(RuntimeError):
    """A safe, identifier-free preflight error."""


@dataclass(frozen=True)
class FrozenProtocol:
    records: Mapping[str, tuple[ClipRecord, ...]]
    manifest_paths: Mapping[str, Path]
    cache_splits: Mapping[str, str]
    summary: Mapping[str, Any]
    protocol_sha256: str

    def cache_record(self, record: ClipRecord) -> ClipRecord:
        """Use an explicitly declared cache route, without changing split roles."""
        if record.clip_id not in self.cache_splits:
            raise ProtocolError("record does not belong to the frozen protocol")
        return replace(record, split=self.cache_splits[record.clip_id])


@dataclass(frozen=True)
class _Entry:
    row: dict[str, Any]
    record: ClipRecord | None
    clip_id: str | None
    group_id: str | None
    source_keys: frozenset[tuple[str, str]]
    identities: Mapping[str, str]
    verified: frozenset[str]
    cache_split: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError("cannot read a required protocol input") from exc
    return digest.hexdigest()


def _read_rows(path: Path, *, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ProtocolError("manifest SHA-256 changed while reading")
        lines = data.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProtocolError("cannot read a UTF-8 manifest") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid JSON at manifest line {line_number}") from exc
        if not isinstance(row, dict):
            raise ProtocolError(f"non-object manifest row at line {line_number}")
        rows.append(row)
    if not rows:
        raise ProtocolError("manifest is empty")
    return rows


def _string(value: Any, field: str, *, required: bool = False) -> str | None:
    if value is None or value == "":
        if required:
            raise ProtocolError(f"missing {field}")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"invalid {field}")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{field} must be a finite number")
    return number


def _identities(row: Mapping[str, Any]) -> tuple[dict[str, str], frozenset[str]]:
    metadata = row.get("identity_metadata", {})
    if not isinstance(metadata, dict):
        raise ProtocolError("invalid identity_metadata")
    status = _string(metadata.get("status"), "identity status")
    common_verified = (
        row.get("identity_verified") is True
        or metadata.get("verified") is True
        or status in {"verified", "confirmed", "manually_verified"}
    )
    identities, verified = {}, set()
    for kind in ("patient", "practitioner"):
        field = f"{kind}_id"
        top, nested = row.get(field), metadata.get(field)
        if top and nested and top != nested:
            raise ProtocolError(f"conflicting {field} metadata")
        identity = _string(top or nested, field)
        local_verified = any(
            container.get(flag) is True
            for container in (row, metadata)
            for flag in (f"{kind}_verified", f"{field}_verified")
        )
        is_verified = common_verified or local_verified
        if identity:
            identities[kind] = identity
            if is_verified:
                verified.add(kind)
        elif local_verified:
            raise ProtocolError(f"verified {field} has no identity value")
    return identities, frozenset(verified)


def _parse_entry(row: Mapping[str, Any], *, role: str | None, video_root: Path | None) -> _Entry:
    normalized = dict(row)
    full = role is not None
    clip_id = _string(row.get("clip_id"), "clip_id", required=full)
    if clip_id and not _CLIP_ID.fullmatch(clip_id):
        raise ProtocolError("invalid clip_id")
    grouping = row.get("grouping", {})
    if not isinstance(grouping, dict):
        raise ProtocolError("invalid grouping metadata")
    group_id = _string(row.get("group_id") or grouping.get("group_id"), "group_id", required=full)
    if group_id:
        normalized["group_id"] = group_id
    source_keys = set()
    video_path = _string(row.get("video_path"), "video_path", required=full)
    if video_path:
        path = Path(video_path).expanduser()
        if not path.is_absolute():
            if video_root is None:
                raise ProtocolError("relative video_path requires explicit --video-root")
            path = video_root / path
        resolved = str(path.resolve())
        normalized["video_path"] = resolved
        source_keys.add(("path", resolved))
    for field in ("source_video_id", "record_id"):
        value = _string(row.get(field), field)
        if value:
            source_keys.add((field, value))
    video_metadata = row.get("video_metadata", {})
    if not isinstance(video_metadata, dict):
        raise ProtocolError("invalid video_metadata")
    hashes = [row.get("source_video_sha256"), row.get("video_sha256"), video_metadata.get("sha256")]
    for value in hashes:
        if value is not None:
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ProtocolError("invalid source video SHA-256")
            source_keys.add(("video_sha256", value.lower()))
    if len({value.lower() for value in hashes if value is not None}) > 1:
        raise ProtocolError("conflicting source video SHA-256 metadata")
    identities, verified = _identities(row)
    record, cache_split = None, None
    if full:
        if row.get("split") != role:
            raise ProtocolError(f"manifest split does not match its {role} role")
        label = row.get("label_id")
        if isinstance(label, bool) or not isinstance(label, int) or label not in EXPECTED_LABELS:
            raise ProtocolError("invalid seven-class label_id")
        if row.get("normalized_label") != EXPECTED_LABELS[label]:
            raise ProtocolError("conflicting seven-class label mapping")
        source = _string(row.get("source_collection"), "source_collection", required=True)
        start = _number(row.get("clip_start_sec"), "clip_start_sec")
        end = _number(row.get("clip_end_sec"), "clip_end_sec")
        duration = _number(row.get("clip_duration_sec"), "clip_duration_sec")
        if start < 0 or end <= start or duration <= 0:
            raise ProtocolError("invalid clip duration or interval")
        if not math.isclose(end - start, duration, rel_tol=0, abs_tol=1e-6):
            raise ProtocolError("clip duration does not match its interval")
        video_duration = row.get("source_video_duration_sec", video_metadata.get("duration_sec"))
        if video_duration is not None:
            if end > _number(video_duration, "source_video_duration_sec") + 1e-6:
                raise ProtocolError("clip interval exceeds declared source video duration")
        cache_split = row.get("cache_split", role)
        if cache_split not in _SPLITS:
            raise ProtocolError("invalid explicit cache_split")
        record = ClipRecord(clip_id, role, label, EXPECTED_LABELS[label], source,
                            group_id, normalized["video_path"], start, end, duration)
    elif not (clip_id or group_id or source_keys or identities):
        raise ProtocolError("prior evaluation row has no usable exposure identifier")
    return _Entry(normalized, record, clip_id, group_id, frozenset(source_keys), identities, verified, cache_split)


def _declared_exposed(row: Mapping[str, Any]) -> bool:
    original_split = _string(row.get("original_split"), "original_split")
    return (
        original_split in {"val", "validation", "dev", "development"}
        or any(row.get(field) is True for field in
               ("prior_eval_exposed", "prior_evaluation_exposure", "evaluation_exposed", "tainted"))
        or "original-test-as-validation" in str(row.get("derived_protocol", ""))
    )


def _audit(entries: Mapping[str, Sequence[_Entry]], prior: Sequence[_Entry]) -> dict[str, Any]:
    clips, groups, sources, identity_members = {}, {}, {}, defaultdict(list)
    intervals = defaultdict(list)
    identity_counts = {kind: {split: 0 for split in entries} for kind in ("patient", "practitioner")}
    for split, members in entries.items():
        counts = Counter(entry.record.label_id for entry in members)
        if set(counts) != set(EXPECTED_LABELS):
            raise ProtocolError(f"all seven classes are required in {split}")
        for entry in members:
            record = entry.record
            if record.clip_id in clips:
                raise ProtocolError("duplicate clip_id or conflicting clip labels")
            clips[record.clip_id] = split
            if entry.group_id in groups and groups[entry.group_id] != split:
                raise ProtocolError("recording group overlap between splits")
            groups[entry.group_id] = split
            for key in entry.source_keys:
                if key in sources and sources[key] != split:
                    raise ProtocolError("source video or recording overlap between splits")
                sources[key] = split
                intervals[key].append(record)
            for kind, identity in entry.identities.items():
                identity_members[(kind, identity)].append((split, kind in entry.verified))
            for kind in entry.verified:
                identity_counts[kind][split] += 1
    for (kind, _), members in identity_members.items():
        if len({split for split, _ in members}) > 1 and any(verified for _, verified in members):
            raise ProtocolError(f"verified {kind} identity overlap between splits")
    for members in intervals.values():
        ordered = sorted(members, key=lambda r: (r.clip_start_sec, r.clip_end_sec))
        active = []
        for record in ordered:
            active = [earlier for earlier in active if earlier.clip_end_sec > record.clip_start_sec + 1e-6]
            for earlier in active:
                if earlier.label_id != record.label_id:
                    raise ProtocolError("overlapping source intervals have conflicting labels")
                if (math.isclose(earlier.clip_start_sec, record.clip_start_sec, abs_tol=1e-6, rel_tol=0)
                        and math.isclose(earlier.clip_end_sec, record.clip_end_sec, abs_tol=1e-6, rel_tol=0)):
                    raise ProtocolError("duplicate source interval under different clip IDs")
            active.append(record)

    prior_clips = {entry.clip_id for entry in prior if entry.clip_id}
    prior_groups = {entry.group_id for entry in prior if entry.group_id}
    prior_sources = {key for entry in prior for key in entry.source_keys}
    prior_identities = {(kind, value) for entry in prior for kind, value in entry.identities.items()}
    exposed = {}
    for split, members in entries.items():
        count = 0
        for entry in members:
            seen = (
                _declared_exposed(entry.row)
                or entry.clip_id in prior_clips or entry.group_id in prior_groups
                or bool(entry.source_keys & prior_sources)
                or any((kind, value) in prior_identities for kind, value in entry.identities.items())
            )
            count += bool(seen)
        exposed[split] = count
    if exposed.get("test", 0):
        raise ProtocolError("test has prior evaluation exposure; it cannot be an independent test")
    history_full_rows = sum(bool(entry.clip_id and entry.group_id
                                 and any(kind == "path" for kind, _ in entry.source_keys)) for entry in prior)
    history_complete = bool(prior) and history_full_rows == len(prior)
    independence = {}
    for kind in identity_counts:
        complete = all(identity_counts[kind][split] == len(members) for split, members in entries.items())
        independence[kind] = {
            "verified_record_counts": identity_counts[kind],
            "complete_verified_coverage": complete,
            "independent_claim_ready": complete,
        }
    return {
        "split_counts": {split: len(members) for split, members in entries.items()},
        "class_counts": {split: {str(label): sum(e.record.label_id == label for e in members)
                                  for label in EXPECTED_LABELS} for split, members in entries.items()},
        "group_counts": {split: len({e.group_id for e in members}) for split, members in entries.items()},
        "duration_sec": {split: {"total": sum(e.record.clip_duration_sec for e in members),
                                  "min": min(e.record.clip_duration_sec for e in members),
                                  "max": max(e.record.clip_duration_sec for e in members)}
                         for split, members in entries.items()},
        "recording_group_disjoint": True,
        "source_video_disjoint": True,
        "source_video_check": "canonical path plus available explicit recording IDs and video hashes; no pixel fingerprint",
        "identity_independence": independence,
        "recording_group_is_patient_identity": False,
        "prior_exposure_record_counts": exposed,
        "prior_exposure_identifier_coverage": {"rows": len(prior), "rows_with_clip_group_and_source_path": history_full_rows},
        "test_history_status": ("no_test_provided" if "test" not in entries else
                                "verified_clean" if history_complete else "history_not_verified"),
        "test_history_scope": "only the supplied prior evaluation manifests and explicit exposure provenance",
        "independent_test_claim": "not certified: evaluation history and identity completeness must be reported",
    }


def _write_json(path: Path, payload: Any, *, private: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if private:
        path.chmod(0o600)


def _write_jsonl(path: Path, entries: Sequence[_Entry]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)


def _repository_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def audit_protocol(train: str | Path, val: str | Path, output: str | Path, *,
                   test: str | Path | None = None, prior_eval_manifests: Sequence[str | Path] = (),
                   video_root: str | Path | None = None) -> Path:
    """Validate, then freeze supplied splits into a newly created directory.

    Partial prior exposure rows can prove contamination, but a clean terminal
    test requires clip/group/source-path coverage in every supplied history row.
    """
    destination = Path(output)
    if destination.exists():
        raise ProtocolError("output must be a new directory; overwrites are forbidden")
    root = Path(video_root).expanduser().resolve() if video_root is not None else None
    paths = {"train": Path(train), "val": Path(val)}
    if test is not None:
        paths["test"] = Path(test)
    source_hashes = {split: _sha256(path) for split, path in paths.items()}
    entries = {split: [_parse_entry(row, role=split, video_root=root)
                       for row in _read_rows(path, expected_sha256=source_hashes[split])]
               for split, path in paths.items()}
    prior_paths = [Path(path) for path in prior_eval_manifests]
    prior_source_hashes = [_sha256(path) for path in prior_paths]
    prior_sets = [[_parse_entry(row, role=None, video_root=root)
                   for row in _read_rows(path, expected_sha256=prior_source_hashes[index])]
                  for index, path in enumerate(prior_paths)]
    audit = _audit(entries, [entry for members in prior_sets for entry in members])
    # Hash before publishing; snapshots, rather than mutable source inputs, are loaded later.
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ProtocolError("cannot create a new output directory") from exc
    private = destination / "private"
    private.mkdir(mode=0o700)
    manifests = {}
    for split, members in entries.items():
        path = private / f"{split}.jsonl"
        _write_jsonl(path, members)
        manifests[split] = {"file": f"private/{split}.jsonl", "sha256": _sha256(path),
                            "source_sha256": source_hashes[split]}
    prior_descriptors = []
    for index, members in enumerate(prior_sets):
        filename = f"prior-eval-{index:03d}.jsonl"
        path = private / filename
        _write_jsonl(path, members)
        prior_descriptors.append({"file": f"private/{filename}", "sha256": _sha256(path),
                                  "source_sha256": prior_source_hashes[index]})
    _write_json(private / "audit_details.json", {
        "input_manifests": {split: str(path.resolve()) for split, path in paths.items()},
        "prior_eval_manifests": [str(path.resolve()) for path in prior_paths],
        "video_root": str(root) if root else None,
        "cache_splits": {entry.clip_id: entry.cache_split for members in entries.values() for entry in members},
    }, private=True)
    summary = {
        "schema_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": _sha256(Path(__file__)),
        "repository_revision": _repository_revision(),
        "label_mapping": {str(key): value for key, value in EXPECTED_LABELS.items()},
        "manifests": manifests,
        "prior_eval_manifests": prior_descriptors,
        "audit": audit,
        "test_history_status": audit["test_history_status"],
        "test_history_scope": audit["test_history_scope"],
        "cache_validation": "required in trainer; this protocol does not inspect or rebuild caches",
        "split_generation": "none; supplied roles retained; no automatic repartition or test reuse",
    }
    protocol_path = destination / "protocol.json"
    _write_json(protocol_path, summary)
    return protocol_path


def _frozen_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen_value(item) for item in value)
    return value


def load_protocol(path: str | Path) -> FrozenProtocol:
    """Recheck hashes and leakage before a trainer can use frozen manifests."""
    protocol_path = Path(path)
    try:
        summary = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot read the frozen protocol JSON") from exc
    if not isinstance(summary, dict) or summary.get("schema_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported frozen protocol version")
    if summary.get("implementation_sha256") != _sha256(Path(__file__)):
        raise ProtocolError("protocol implementation changed; generate a new audit")
    if summary.get("label_mapping") != {str(key): value for key, value in EXPECTED_LABELS.items()}:
        raise ProtocolError("frozen label mapping changed")
    descriptors = summary.get("manifests")
    if not isinstance(descriptors, dict) or set(descriptors) not in ({"train", "val"}, {"train", "val", "test"}):
        raise ProtocolError("invalid frozen split descriptors")
    entries, manifest_paths = {}, {}
    for split, descriptor in descriptors.items():
        expected = f"private/{split}.jsonl"
        if not isinstance(descriptor, dict) or descriptor.get("file") != expected:
            raise ProtocolError("unsafe frozen manifest location")
        manifest_path = protocol_path.parent / expected
        if _sha256(manifest_path) != descriptor.get("sha256"):
            raise ProtocolError("frozen manifest SHA-256 changed")
        manifest_paths[split] = manifest_path.resolve()
        entries[split] = [_parse_entry(row, role=split, video_root=None)
                          for row in _read_rows(manifest_path, expected_sha256=descriptor.get("sha256"))]
    prior_descriptors = summary.get("prior_eval_manifests")
    if not isinstance(prior_descriptors, list):
        raise ProtocolError("invalid frozen exposure descriptors")
    prior = []
    for index, descriptor in enumerate(prior_descriptors):
        expected = f"private/prior-eval-{index:03d}.jsonl"
        if not isinstance(descriptor, dict) or descriptor.get("file") != expected:
            raise ProtocolError("unsafe frozen exposure manifest location")
        prior_path = protocol_path.parent / expected
        if _sha256(prior_path) != descriptor.get("sha256"):
            raise ProtocolError("frozen exposure SHA-256 changed")
        prior.extend(_parse_entry(row, role=None, video_root=None)
                     for row in _read_rows(prior_path, expected_sha256=descriptor.get("sha256")))
    actual_audit = _audit(entries, prior)
    if actual_audit != summary.get("audit"):
        raise ProtocolError("frozen audit summary does not match its manifests")
    if any(summary.get(field) != actual_audit[field] for field in ("test_history_status", "test_history_scope")):
        raise ProtocolError("frozen test history metadata changed")
    return FrozenProtocol(
        records=MappingProxyType({split: tuple(entry.record for entry in members) for split, members in entries.items()}),
        manifest_paths=MappingProxyType(manifest_paths),
        cache_splits=MappingProxyType({entry.clip_id: entry.cache_split for members in entries.values() for entry in members}),
        summary=_frozen_value(summary), protocol_sha256=_sha256(protocol_path),
    )


load_and_audit = load_protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="audit supplied JSONL splits and freeze them in a new directory")
    audit.add_argument("--train", type=Path, required=True)
    audit.add_argument("--val", type=Path, required=True)
    audit.add_argument("--test", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--video-root", type=Path)
    audit.add_argument("--prior-eval-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = audit_protocol(args.train, args.val, args.output, test=args.test,
                                prior_eval_manifests=args.prior_eval_manifest, video_root=args.video_root)
        # Never print input paths, clip IDs or private group/identity metadata.
        print(json.dumps(json.loads(result.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True))
    except (ProtocolError, OSError) as exc:
        message = str(exc) if isinstance(exc, ProtocolError) else "cannot write the frozen protocol"
        print(f"protocol audit failed: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
