#!/usr/bin/env bash
# Stage A: M&B's own HF-generate path (patched only for local I/O), Qwen3-8B as configured
# in their models.json (enable_thinking false, DEFAULT_SAMPLING), on the first N (MB_N) items of our seed-0 draw
# (their get_dataset takes the first N rows). Unbiased ran with N=60; expert with N=32, then 48.
# Usage: run_stageA_hf.sh <bias:unbiased|expert|self|metadata> [hint_idx...]
set -euo pipefail
BIAS=$1; shift
cd /root/jlens-motivated/third_party/motivated-reasoning
export MOTIVATION_HOME=/root/jlens-motivated/data/runs/mb_replication/stageA_hf
export MB_LOCAL_DATASET=/root/jlens-motivated/data/runs/mb_replication/items_ds
export MB_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export HF_HUB_OFFLINE=1
mkdir -p "$MOTIVATION_HOME/logs"
N=${MB_N:-60}
EXTRA=${MB_EXTRA:-}   # e.g. --keep_existing to extend an existing file to N rows
COMMON="--model qwen-3-8b --dataset mmlu --generate --max_samples $N --bs_gen 16 $EXTRA"
if [[ "$BIAS" == "unbiased" ]]; then
    .venv/bin/python -I /root/jlens-motivated/scripts/mb/seeded_main.py --seed 100 -- $COMMON --reason_first \
        > "$MOTIVATION_HOME/logs/unbiased_rf.txt" 2>&1
    exit 0
fi
for h in "$@"; do
    seed=$(( 1000 + h ))
    [[ "$BIAS" == "self" ]] && seed=$(( 2000 + h ))
    [[ "$BIAS" == "metadata" ]] && seed=$(( 3000 + h ))
    .venv/bin/python -I /root/jlens-motivated/scripts/mb/seeded_main.py --seed $seed -- $COMMON --bias "$BIAS" --hint_idx "$h" \
        > "$MOTIVATION_HOME/logs/${BIAS}_${h}.txt" 2>&1
done
