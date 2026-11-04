#!/usr/bin/env bash
# Steering test GPU chain (octavia): pilot -> alpha rule -> part 1 -> part 2 (priorities in order).
# Stops on the first failure (rule 1). Log: data/logs/steer_chain.log
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
R="uv run --project envs/lens python"
MAXP="${MAXP:-3}"
echo "$(date +%H:%M) start (git $(git rev-parse --short HEAD 2>/dev/null || echo ?))"
$R scripts/lens/steer_generate.py --run data/steer/gen/jobs_pilot.jsonl --dirs data/steer/directions.pt --out data/steer/gen
echo "$(date +%H:%M) pilot done"
CUDA_VISIBLE_DEVICES="" $R scripts/lens/analyze_steer.py --answer data/steer/none --gen data/steer/gen --out data/steer/analysis_pilot.json
ALPHAS=$(python3 -c "import json; print(*json.load(open('data/steer/analysis_pilot.json'))['pilot_rule']['alpha_lo_hi'])")
echo "$(date +%H:%M) pilot rule alpha_lo alpha_hi = $ALPHAS"
CUDA_VISIBLE_DEVICES="" $R scripts/lens/steer_generate.py --build main --alphas $ALPHAS --out data/steer/gen
$R scripts/lens/steer_answer.py --dirs data/steer/directions.pt --out data/steer/answer
echo "$(date +%H:%M) part 1 done"
for p in $(seq 0 "$MAXP"); do
  $R scripts/lens/steer_generate.py --run data/steer/gen/jobs_main.jsonl --dirs data/steer/directions.pt --out data/steer/gen --max-priority "$p"
  echo "$(date +%H:%M) part 2 priority $p done"
done
echo "$(date +%H:%M) CHAIN_DONE"
