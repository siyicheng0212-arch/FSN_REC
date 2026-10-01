#!/usr/bin/env bash
# Three independent queues; default is read-only planning, never DDP.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${FSN_PYTHON:?Set FSN_PYTHON to the verified CUDA Python executable}"
exec "$FSN_PYTHON" -u -m experiments.run_aligned_context --python "$FSN_PYTHON" "$@"
