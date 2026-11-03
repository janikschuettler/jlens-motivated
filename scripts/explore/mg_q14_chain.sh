#!/bin/bash
# Models x grader-bug Phase 2 on octavia: Qwen3-14B checks, screen, known items, arms.
# Stops at the first failure (rule 1). Weights and lens in the RAM-disk HF cache.
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
export JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub
L=data/logs
while pgrep -f "run_w2.py configs/explore/mg_qwen8b" >/dev/null; do sleep 20; done
echo "$(date +%T) M0.1 model check"; uv run --project envs/lens python scripts/lens/m0_check_model.py > $L/mg_q14_m0_1.log 2>&1
echo "$(date +%T) M0.2 lens check"; uv run --project envs/lens python scripts/lens/m0_check_lens.py > $L/mg_q14_m0_2.log 2>&1
echo "$(date +%T) screen"; uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/mg_q14_screen.yaml --chunk-size 2000 > $L/mg_q14_screen.log 2>&1
uv run --project envs/gen python scripts/explore/mg_known.py --screen data/explore/models_grader/q14_screen \
  --pool data/explore/known_items.jsonl --setting Q14 --out data/explore/models_grader/q14_known_items.jsonl
echo "$(date +%T) arms"; uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/mg_q14_arms.yaml --chunk-size 1500 > $L/mg_q14_arms.log 2>&1
echo "$(date +%T) done"
