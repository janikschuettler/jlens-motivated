#!/usr/bin/env bash
# Copy the repo (code, configs, docs, .git) to arena9-mayor. data/ and virtual
# envs stay where they are on each machine. Lockfiles are written by uv on mayor,
# so newer ones are pulled back first (otherwise --delete would remove them).
set -euo pipefail
cd "$(dirname "$0")/.."
for env in gen lens; do
    rsync -az --update "arena9-mayor:/root/jlens-motivated/envs/$env/uv.lock" "envs/$env/" 2>/dev/null || true
done
rsync -az --delete \
    --exclude 'data/' --exclude '.venv/' --exclude '__pycache__/' \
    --exclude '.pytest_cache/' --exclude '.ipynb_checkpoints/' \
    ./ arena9-mayor:/root/jlens-motivated/
