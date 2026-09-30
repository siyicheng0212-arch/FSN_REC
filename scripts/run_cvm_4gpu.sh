#!/usr/bin/env bash
# Usage: bash scripts/run_cvm_4gpu.sh /new/run_plan.json [--execute]
# Dry-run by default. Four independent jobs, not DDP; never resumes.
set -euo pipefail
set -o noclobber
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /new/run_plan.json [--execute]" >&2
  exit 2
fi
PLAN_PATH=$1
EXECUTION_FLAG=${2:-}
if [[ -n "$EXECUTION_FLAG" && "$EXECUTION_FLAG" != "--execute" ]]; then
  echo "The only optional argument is --execute." >&2
  exit 2
fi
SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_DIRECTORY=$(cd -- "$SCRIPT_DIRECTORY/.." && pwd)
CVM_PYTHON=${CVM_PYTHON:-python}
cd -- "$REPOSITORY_DIRECTORY"
if [[ "$EXECUTION_FLAG" == "--execute" ]]; then
  exec "$CVM_PYTHON" -m cvm.run_suite launch --plan "$PLAN_PATH" --python "$CVM_PYTHON" --execute
fi
exec "$CVM_PYTHON" -m cvm.run_suite launch --plan "$PLAN_PATH" --python "$CVM_PYTHON"
