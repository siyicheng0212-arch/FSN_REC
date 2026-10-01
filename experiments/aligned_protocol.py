"""Frozen full-data protocol shared by training, CUDA smoke and launcher."""
from __future__ import annotations

DATA_PROTOCOL = "fsn-full-7372-823-development-v1"
EXPECTED_COUNTS = {"train": 7372, "val": 823}
EXPECTED_MANIFEST_SHA = {
    "train": "093d0adf08dc1d382c0d2e1863a2f4724cb24ebd5b3d0af070e0c76ca989f1d3",
    "val": "ad785ec18b14b63616582a143384fac649e69e27d142dfcf87c6852f9d3c6f02",
}
EVALUATION_STATUS = (
    "full 7372/823 development protocol; all 823 validation clips select best; "
    "previously used development validation, not an independent test"
)


def validate_manifest_protocol(counts, manifest_sha256):
    """Reject subsets or regenerated manifests before reading feature caches."""
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"{DATA_PROTOCOL} requires counts {EXPECTED_COUNTS}, got {counts}; "
            "reuse the frozen full manifests rather than internal validation"
        )
    if manifest_sha256 != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(
            f"{DATA_PROTOCOL} manifest SHA mismatch; "
            "reuse the frozen full manifests without rewriting them"
        )
