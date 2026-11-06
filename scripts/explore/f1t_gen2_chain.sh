#!/usr/bin/env bash
# F1-T on derpy: C3 generation (chunks of 4 samples/item), then T2 re-asking forks on the wording runs.
set -euo pipefail
unset VIRTUAL_ENV; export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
L=data/logs
echo "$(date +%T) c3"; uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/f1t_c3.yaml --chunk-size 496 --deadline 03:40 > $L/f1t_c3.log 2>&1
echo "$(date +%T) forks"; uv run --project envs/gen python scripts/explore/run_forks.py configs/explore/f1t_forks_t2.yaml --runs data/explore/f1t/wording/runs.jsonl --name f1t_s1/forks_reelicit > $L/f1t_s1_forks.log 2>&1
echo "$(date +%T) gen2 done"
