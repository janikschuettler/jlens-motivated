#!/bin/bash
# Eval-awareness replication readouts on octavia (after scripts/gen/evalaware_corpus.py).
# Usage: scripts/explore/evalaware_chain.sh [corpus|grader|all]
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
D=data/explore/evalaware
R=data/readouts/evalaware
LENS="uv run --project envs/lens python scripts/lens/readout_evalaware.py"
what=${1:-all}
log() { echo "$(date +%T) $*"; }
if [[ $what == corpus || $what == all ]]; then
  head -1 $D/blackmail/runs.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['run_id'])" > $D/select_blackmail0.txt
  log blackmail_prompt
  $LENS --runs $D/blackmail/runs.jsonl --select $D/select_blackmail0.txt --out $R/blackmail_prompt --all-prompt --validate 1
  for s in oasst wildchat; do
    log $s; $LENS --runs $D/$s/runs.jsonl --out $R/$s --all-prompt --validate 1
  done
  for s in redux_home redux_bench redux_casual grader_agent grader_plain syco moral hack_task blackmail; do
    log $s; $LENS --runs $D/$s/runs.jsonl --out $R/$s --validate 1
  done
fi
if [[ $what == grader || $what == all ]]; then
  log nt8; $LENS --runs data/explore/models_grader/nt8_arms/runs.jsonl --select $D/select_nt8.txt --out $R/grader_nt8 --validate 1
  ADIR=$(ls -d ~/.cache/huggingface/hub/models--cds-jb--qwen3-8b-reward-hack-kl-cot/snapshots/7ec92bca778d3228fb056061bd94f96717e0a6d3/disposition)
  log hack; $LENS --runs data/hack/gen/runs.jsonl --select $D/select_hack.txt --out $R/hack_disposition --lora $ADIR \
    --reading-model disposition --validate 1
  log q14; JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub $LENS \
    --runs data/explore/models_grader/q14_arms/runs.jsonl --select $D/select_q14.txt --out $R/grader_q14 --validate 1
fi
log "done; disk: $(df -h /root | tail -1)"
