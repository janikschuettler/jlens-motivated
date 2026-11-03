#!/usr/bin/env bash
# Wave 2 part 2 on octavia, one step after another; stops at the first failure (rule 1).
# Logs: data/explore/logs/w2p2_<step>.log. Started from the repo root.
set -euo pipefail
unset VIRTUAL_ENV
cd "$(dirname "$0")/../.."
L=data/explore/logs
R="uv run --project envs/gen python"
step() { echo "$(date +%H:%M:%S) START $1"; }
step wording;  $R scripts/explore/run_arms.py configs/explore/w2_wording.yaml --chunk-size 400 --deadline 01:45 > $L/w2p2_wording.log 2>&1
step w2b;      $R scripts/explore/run_w2.py configs/explore/w2_prefill.yaml configs/explore/w2_pushback.yaml configs/explore/w2_opinion.yaml --chunk-size 840 --deadline 02:15 > $L/w2p2_w2b.log 2>&1
RUNS="data/explore/w2/wording/runs.jsonl data/explore/w2/prefill/runs.jsonl data/explore/w2/pushback/runs.jsonl data/explore/w2/opinion/runs.jsonl"
step reelicit; $R scripts/explore/run_forks.py configs/explore/w2_forks_reelicit.yaml --runs $RUNS --name w2/forks_reelicit_p2 > $L/w2p2_reelicit.log 2>&1
step early;    $R scripts/explore/run_forks.py configs/explore/w2_forks_early.yaml --runs $RUNS --name w2/forks_early_p2 > $L/w2p2_early.log 2>&1
step moral_screen; $R scripts/explore/run_w2.py configs/explore/w2_moral_screen.yaml > $L/w2p2_moral_screen.log 2>&1
step moral_known;  $R scripts/explore/w2_known.py data/explore/w2/moral_screen --arm screen --setting A --out data/explore/w2/moral_known.jsonl > $L/w2p2_moral_known.log 2>&1
step moral_nudge;  $R scripts/explore/run_w2.py configs/explore/w2_moral_nudge.yaml --chunk-size 360 --deadline 02:40 > $L/w2p2_moral_nudge.log 2>&1
step moral_reelicit; $R scripts/explore/run_forks.py configs/explore/w2_forks_reelicit.yaml --runs data/explore/w2/moral_nudge/runs.jsonl --name w2/forks_reelicit_moral > $L/w2p2_moral_reelicit.log 2>&1
echo "$(date +%H:%M:%S) ALL DONE"
