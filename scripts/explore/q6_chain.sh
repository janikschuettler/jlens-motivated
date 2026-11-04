#!/bin/bash
# Q6 point of no return on octavia: first pass for each case in order, then the optional
# second pass (cuts inside the commitment segment) while time remains. Stops at the first
# failure (rule 1). Usage: bash scripts/explore/q6_chain.sh [GEN_DEADLINE HH:MM] [REFINE_DEADLINE HH:MM]
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
GEN_DL=${1:-06:35}
REF_DL=${2:-06:50}
S=configs/explore/q6_point_of_no_return.yaml
L=data/logs
run() { uv run --project envs/gen python scripts/explore/run_q6.py $S "$@"; }
before() { [ "$(date +%H%M)" \< "${1/:/}" ]; }
echo "q6-point-of-no-return $(date +%H:%M) (overnight agent, Q6 resampling)" > $L/GPU_OWNER
echo "$(date +%T) c3"; run --case c3 > $L/q6_c3.log 2>&1
echo "$(date +%T) hint8b"; run --case hint8b --chunks 3 --deadline $GEN_DL > $L/q6_hint8b.log 2>&1
if before $GEN_DL; then
  echo "$(date +%T) grader14b"
  JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub run --case grader14b --chunks 2 --deadline $GEN_DL > $L/q6_grader14b.log 2>&1
fi
if before $GEN_DL; then echo "$(date +%T) hack_ws"; run --case hack_ws > $L/q6_hack_ws.log 2>&1; fi
for c in c3 hack_ws hint8b; do
  if before $REF_DL && [ -f data/explore/q6_pnr/$c/resamples.jsonl ]; then
    echo "$(date +%T) refine $c"; run --case $c --refine --deadline $REF_DL > $L/q6_${c}_refine.log 2>&1
  fi
done
if before $REF_DL && [ -f data/explore/q6_pnr/grader14b/resamples.jsonl ]; then
  echo "$(date +%T) refine grader14b"
  JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub run --case grader14b --refine --deadline $REF_DL > $L/q6_grader14b_refine.log 2>&1
fi
rm -f $L/GPU_OWNER
echo "$(date +%T) done"
