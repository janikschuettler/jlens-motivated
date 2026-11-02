#!/usr/bin/env bash
# Copy the human-readable reports (audit.md, pilot_summary.md, ...) of a run set from the GPU machine to this
# machine (data/ is gitignored), so they can be opened in the editor.
# usage: scripts/pull_audit.sh m0
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_HOST="${GPU_HOST:-arena9-octavia}"
run_set="${1:?usage: pull_audit.sh <run_set>}"
mkdir -p "data/runs/$run_set"
rsync -az --include="*.md" --exclude="*" "$GPU_HOST:/root/jlens-motivated/data/runs/$run_set/" "data/runs/$run_set/"
ls "data/runs/$run_set"/*.md
