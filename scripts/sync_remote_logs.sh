#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repo_root}/remote_logs"
remote="${FSN_REMOTE_HOST:-fsn-autodl}"

for directory in transfer_diagnostic formal_results formal_trainval_results; do
  if ! ssh -o BatchMode=yes "${remote}" \
    "test -d /root/autodl-tmp/${directory}"; then
    continue
  fi
  mkdir -p "${destination}/${directory}"
  rsync -az --prune-empty-dirs \
    --include='*/' \
    --include='*.log' \
    --include='result.json' \
    --include='formal_manifest_summary.json' \
    --exclude='*' \
    "${remote}:/root/autodl-tmp/${directory}/" \
    "${destination}/${directory}/"
done

ssh -o BatchMode=yes "${remote}" \
  'date; pgrep -af "experiments.train_adafocus" || true; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader' \
  > "${destination}/server_status.txt"

printf 'Synced FSN logs from %s to %s\n' "${remote}" "${destination}"
