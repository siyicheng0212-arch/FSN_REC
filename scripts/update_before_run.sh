#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to update a dirty worktree. Preserve or commit local changes first." >&2
  exit 1
fi

git fetch origin main
git merge --ff-only origin/main
git rev-parse HEAD | tee .last_run_commit
