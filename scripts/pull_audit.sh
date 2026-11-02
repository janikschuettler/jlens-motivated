#!/usr/bin/env bash
# Copy the human-readable audit file(s) of a run set from arena9-mayor to this
# machine (data/ is gitignored), so they can be opened in the editor.
# usage: scripts/pull_audit.sh m0
set -euo pipefail
cd "$(dirname "$0")/.."
run_set="${1:?usage: pull_audit.sh <run_set>}"
mkdir -p "data/runs/$run_set"
rsync -az "arena9-mayor:/root/jlens-motivated/data/runs/$run_set/audit.md" "arena9-mayor:/root/jlens-motivated/data/runs/$run_set/pilot_summary.md" "data/runs/$run_set/"
echo "data/runs/$run_set/audit.md"
