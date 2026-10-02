"""Consume existing evidence seals and verify the user's known source types."""
import json
from pathlib import Path

from experiments.fsn_tcn.io import load_bundle as _load_bundle


def validate_source_kinds(bundle):
    mapping = json.loads((Path(__file__).resolve().parents[1] / "relation" / "source_map.json").read_text())
    for row in bundle.metadata.values():
        declared = mapping.get(row["source_collection"])
        if declared is not None and declared != row["source_kind"]:
            raise ValueError("sealed source type disagrees with the known source map; preserve artifacts and audit, do not relabel silently")
    return bundle


def load_bundle(feature_index, source_protocol_dir):
    return validate_source_kinds(_load_bundle(feature_index, source_protocol_dir))
