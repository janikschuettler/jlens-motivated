#!/usr/bin/env bash
# Slot 2: exploratory other-letter targets (part 1), then generation (part 2) in priority order.
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="uv run --project envs/lens python"
MAXP="${MAXP:-3}"
echo "$(date +%H:%M:%S) slot 2 start (git $(git rev-parse --short HEAD))"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> data/logs/steer_gpu_mem2.log; sleep 5; done ) &
MEM=$!
trap 'kill $MEM 2>/dev/null || true' EXIT
$R scripts/lens/steer_answer.py --dirs data/steer/directions.pt --out data/steer/answer --mode other
echo "$(date +%H:%M:%S) other-letter part done; peak GPU MiB so far $(sort -n data/logs/steer_gpu_mem2.log | tail -1)"
for p in $(seq 0 "$MAXP"); do
  $R scripts/lens/steer_generate.py --run data/steer/gen/jobs_main.jsonl --dirs data/steer/directions.pt --out data/steer/gen --max-priority "$p"
  echo "$(date +%H:%M:%S) part 2 priority $p done; peak GPU MiB so far $(sort -n data/logs/steer_gpu_mem2.log | tail -1)"
done
echo CHAIN2_DONE
