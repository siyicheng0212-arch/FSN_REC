#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repo_root}/remote_logs"
remote="${FSN_REMOTE_HOST:-fsn-autodl}"

mkdir -p "${destination}/transfer_diagnostic" "${destination}/formal_results"

rsync -az --prune-empty-dirs \
  --include='*/' \
  --include='*.log' \
  --include='result.json' \
  --include='formal_manifest_summary.json' \
  --exclude='*' \
  "${remote}:/root/autodl-tmp/transfer_diagnostic/" \
  "${destination}/transfer_diagnostic/"

rsync -az --prune-empty-dirs \
  --include='*/' \
  --include='*.log' \
  --include='result.json' \
  --exclude='*' \
  "${remote}:/root/autodl-tmp/formal_results/" \
  "${destination}/formal_results/"

ssh -o BatchMode=yes "${remote}" \
  'date; pgrep -af "experiments.train_adafocus" || true; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader' \
  > "${destination}/server_status.txt"

printf 'Synced FSN logs from %s to %s\n' "${remote}" "${destination}"
