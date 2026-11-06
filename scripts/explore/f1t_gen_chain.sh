#!/usr/bin/env bash
# F1-T stage 1 on derpy: W (hint wordings) + C3 generation, then forks for T2. Stops at the first failure (rule 1).
set -euo pipefail
unset VIRTUAL_ENV
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
L=data/logs; mkdir -p $L
R="uv run --project envs/gen python"
echo "$(date +%T) wording"; $R scripts/explore/run_arms.py configs/explore/f1t_wording.yaml --chunk-size 912 --deadline 04:15 > $L/f1t_wording.log 2>&1
echo "$(date +%T) c3"; $R scripts/explore/run_w2.py configs/explore/f1t_c3.yaml --chunk-size 992 --deadline 04:15 > $L/f1t_c3.log 2>&1
echo "$(date +%T) gen done"
