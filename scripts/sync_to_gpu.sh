#!/usr/bin/env bash
# Copy the repo (code, configs, docs, .git) to the GPU machine ($GPU_HOST, default arena9-octavia; override with GPU_HOST=<ssh host>). data/ and virtual
# envs stay where they are on each machine. Lockfiles are written by uv on the GPU machine,
# so newer ones are pulled back first (otherwise --delete would remove them).
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_HOST="${GPU_HOST:-arena9-octavia}"
for env in gen lens; do
    rsync -az --update "$GPU_HOST:/root/jlens-motivated/envs/$env/uv.lock" "envs/$env/" 2>/dev/null || true
done
rsync -az --delete \
    --exclude 'data/' --exclude 'third_party/' --exclude '.venv/' --exclude '__pycache__/' \
    --exclude '.pytest_cache/' --exclude '.ipynb_checkpoints/' \
    ./ "$GPU_HOST":/root/jlens-motivated/
