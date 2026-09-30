"""Derive a source-domain holdout using only an initially clean frozen test.

Source collection denotes a dataset/acquisition domain, not a patient,
practitioner or hospital identity. No train/validation clip is promoted to test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence
from collections.abc import Mapping

from cvm.protocol import FrozenProtocol, ProtocolError, audit_protocol, load_protocol
from experiments.pilot_data import EXPECTED_LABELS

SOURCE_HOLDOUT_VERSION = "fsn-cvm-source-holdout-1.0"
PREDECLARED_SELECTION = "predeclared_before_source_scores"


class SourceHoldoutDataInsufficient(ProtocolError):
    """The requested domain cannot form the declared seven-class experiment."""

    def __init__(self, aggregate: dict[str, Any]):
        super().__init__("data_insufficient: requested source holdout lacks required test sources or seven-class coverage; no automatic source change")
        self.aggregate = aggregate


def source_alias(source: str) -> str:
    return "source-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError("required source-holdout artifact is unavailable") from exc


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot read source-holdout metadata") from exc
    if not isinstance(value, dict):
        raise ProtocolError("source-holdout metadata must be an object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot read a source-holdout manifest") from exc
    if any(not isinstance(row, dict) for row in values):
        raise ProtocolError("source-holdout manifest rows must be objects")
    return values


def _sources(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ProtocolError("explicit nonempty heldout sources are required")
    normalized = tuple(value.strip() for value in values)
    if len(set(normalized)) != len(normalized):
        raise ProtocolError("heldout sources must not be repeated")
    if len({source_alias(value) for value in normalized}) != len(normalized):
        raise ProtocolError("source alias collision")
    return tuple(sorted(normalized))


def _select(parent: FrozenProtocol, selected: tuple[str, ...]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    original = {split: _rows(path) for split, path in parent.manifest_paths.items()}
    selected_set = set(selected)
    kept = {
        split: [row for row in original[split]
                if (row["source_collection"].strip() in selected_set) == (split == "test")]
        for split in ("train", "val", "test")
    }
    present_test_sources = {row["source_collection"].strip() for row in original["test"]}
    missing_sources = selected_set - present_test_sources
    missing_classes = {split: sorted(set(EXPECTED_LABELS) - {row["label_id"] for row in rows})
                       for split, rows in kept.items()}
    aggregate = {
        "status": "data_insufficient" if missing_sources or any(missing_classes.values()) else "ready",
        "heldout_source_aliases": [source_alias(value) for value in selected],
        "missing_initial_test_source_aliases": sorted(source_alias(value) for value in missing_sources),
        "split_counts": {split: len(rows) for split, rows in kept.items()},
        "missing_label_ids": missing_classes,
        "removed_record_counts": {split: len(original[split]) - len(rows) for split, rows in kept.items()},
        "class_counts": {split: {str(label): sum(row["label_id"] == label for row in rows) for label in EXPECTED_LABELS}
                         for split, rows in kept.items()},
    }
    return kept, aggregate


def _require_clean_parent(path: Path, parent: FrozenProtocol) -> None:
    if "test" not in parent.records or parent.summary.get("test_history_status") != "verified_clean":
        raise ProtocolError("source holdout requires an initial verified_clean test; used validation cannot be promoted")
    if parent.summary.get("source_holdout") is not None:
        raise ProtocolError("derive source holdout from the initial frozen protocol, not another derivative")
    if _sha(path) != parent.protocol_sha256:
        raise ProtocolError("initial protocol changed during source holdout planning")


def derive_source_holdout(protocol: str | Path, heldout_sources: Sequence[str], output: str | Path, *,
                          source_selection_exposure: str) -> Path:
    """Freeze an exact subset of a clean parent; never resplit or add clips."""
    if source_selection_exposure != PREDECLARED_SELECTION:
        raise ProtocolError("source selection must be explicitly predeclared before source scores")
    selected = _sources(heldout_sources)
    destination = Path(output)
    if destination.exists():
        raise ProtocolError("output must be a new directory; overwrites are forbidden")
    parent_path = Path(protocol).resolve()
    parent = load_protocol(parent_path)
    _require_clean_parent(parent_path, parent)
    kept, aggregate = _select(parent, selected)
    if aggregate["status"] == "data_insufficient":
        raise SourceHoldoutDataInsufficient(aggregate)
    prior_paths = [parent_path.parent / descriptor["file"] for descriptor in parent.summary["prior_eval_manifests"]]
    if not prior_paths:
        raise ProtocolError("initial clean status has no frozen evaluation history")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Temporary private inputs are copied by audit_protocol into its own frozen
    # snapshots. Neither original manifests nor existing caches are modified.
    with tempfile.TemporaryDirectory(prefix=".source-holdout-input-", dir=destination.parent) as temporary:
        inputs = Path(temporary)
        for split, rows in kept.items():
            path = inputs / f"{split}.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            path.chmod(0o600)
        result = audit_protocol(inputs / "train.jsonl", inputs / "val.jsonl", destination,
                                test=inputs / "test.jsonl", prior_eval_manifests=prior_paths)
    plan = {
        "schema_version": SOURCE_HOLDOUT_VERSION,
        "parent_protocol_path": str(parent_path),
        "parent_protocol_sha256": parent.protocol_sha256,
        "heldout_sources": list(selected),
        "source_selection_exposure": source_selection_exposure,
        "selection_declaration": "operator attestation; software cannot prove prior score exposure history",
    }
    plan_path = destination / "private" / "source_holdout_plan.json"
    with plan_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    plan_path.chmod(0o600)
    summary = _json(result)
    if summary.get("test_history_status") != "verified_clean":
        raise ProtocolError("derived source holdout did not preserve clean parent history")
    summary["source_holdout"] = {
        "schema_version": SOURCE_HOLDOUT_VERSION,
        "implementation_sha256": _sha(Path(__file__)),
        "parent_protocol_sha256": parent.protocol_sha256,
        "plan_file": "private/source_holdout_plan.json",
        "plan_sha256": _sha(plan_path),
        "source_selection_exposure": source_selection_exposure,
        "selection_declaration": plan["selection_declaration"],
        "source_scope": "source_collection dataset/acquisition domain; not patient or practitioner independence",
        "test_selection": "only selected sources already in the initial verified_clean test; no train/val promotion",
        "history_provenance": "all frozen parent prior-evaluation manifests carried unchanged; clean is limited to declared history",
        **aggregate,
    }
    temporary_summary = result.with_suffix(".source-holdout.tmp")
    with temporary_summary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary_summary, result)
    load_source_holdout_protocol(result)
    return result


def validate_source_holdout_protocol(path: str | Path, protocol: FrozenProtocol) -> None:
    """Hook for the generic loader: verify exact parent membership/provenance."""
    protocol_path = Path(path)
    declaration = protocol.summary.get("source_holdout")
    if not isinstance(declaration, Mapping) or declaration.get("schema_version") != SOURCE_HOLDOUT_VERSION:
        raise ProtocolError("missing or unsupported source-holdout declaration")
    if declaration.get("implementation_sha256") != _sha(Path(__file__)):
        raise ProtocolError("source-holdout implementation changed; generate a new audit")
    if declaration.get("plan_file") != "private/source_holdout_plan.json":
        raise ProtocolError("unsafe source-holdout plan location")
    plan_path = protocol_path.parent / "private" / "source_holdout_plan.json"
    if _sha(plan_path) != declaration.get("plan_sha256"):
        raise ProtocolError("source-holdout private plan SHA changed")
    plan = _json(plan_path)
    if plan.get("schema_version") != SOURCE_HOLDOUT_VERSION or plan.get("source_selection_exposure") != PREDECLARED_SELECTION:
        raise ProtocolError("source-holdout selection was not frozen before source scores")
    parent_value = plan.get("parent_protocol_path")
    if not isinstance(parent_value, str) or not Path(parent_value).is_absolute():
        raise ProtocolError("invalid private parent protocol location")
    parent_path = Path(parent_value)
    parent_sha = _sha(parent_path)
    if parent_sha != plan.get("parent_protocol_sha256") or parent_sha != declaration.get("parent_protocol_sha256"):
        raise ProtocolError("source-holdout parent protocol SHA changed")
    # Disallow nested/self-referential derivations before invoking the generic loader.
    if _json(parent_path).get("source_holdout") is not None:
        raise ProtocolError("source-holdout parent must be the initial frozen protocol")
    parent = load_protocol(parent_path)
    _require_clean_parent(parent_path, parent)
    selected = _sources(plan.get("heldout_sources", []))
    expected, aggregate = _select(parent, selected)
    if aggregate["status"] != "ready":
        raise ProtocolError("source-holdout parent data no longer supports the declared seven-class experiment")
    for key, value in aggregate.items():
        frozen = declaration.get(key)
        if isinstance(value, list):
            frozen = list(frozen) if isinstance(frozen, (list, tuple)) else frozen
        elif isinstance(value, dict):
            frozen = json.loads(json.dumps(_plain(frozen)))
        if frozen != value:
            raise ProtocolError("source-holdout aggregate differs from frozen selection")
    if declaration.get("source_selection_exposure") != PREDECLARED_SELECTION:
        raise ProtocolError("source-holdout public selection exposure changed")
    for split in ("train", "val", "test"):
        if split not in protocol.manifest_paths or _rows(protocol.manifest_paths[split]) != expected[split]:
            raise ProtocolError("source-holdout manifests are not the exact declared parent subsets; test promotion is forbidden")
    before = [descriptor["sha256"] for descriptor in parent.summary["prior_eval_manifests"]]
    after = [descriptor["sha256"] for descriptor in protocol.summary["prior_eval_manifests"]]
    if before != after or protocol.summary.get("test_history_status") != "verified_clean":
        raise ProtocolError("source-holdout history provenance was lost or reset")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def load_source_holdout_protocol(path: str | Path) -> FrozenProtocol:
    protocol = load_protocol(path)
    validate_source_holdout_protocol(path, protocol)
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser("derive")
    derive.add_argument("--protocol", type=Path, required=True)
    derive.add_argument("--heldout-source", action="append", required=True)
    derive.add_argument("--source-selection-exposure", choices=(PREDECLARED_SELECTION,), required=True)
    derive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = derive_source_holdout(args.protocol, args.heldout_source, args.output,
                                       source_selection_exposure=args.source_selection_exposure)
        summary = _json(result)
        print(json.dumps({"status": "ready", "protocol_sha256": _sha(result),
                          "source_holdout": summary["source_holdout"]}, ensure_ascii=False, sort_keys=True))
    except SourceHoldoutDataInsufficient as exc:
        print(json.dumps(exc.aggregate, ensure_ascii=False, sort_keys=True))
        return 3
    except (ProtocolError, OSError) as exc:
        message = str(exc) if isinstance(exc, ProtocolError) else "cannot write source-holdout artifacts"
        print(f"source holdout failed: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
