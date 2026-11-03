#!/bin/bash
# Models x grader-bug follow-up on octavia: non-thinking screens + agent-framing arms for Qwen3-8B, then Qwen3-14B.
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
L=data/logs
for tag in nt8 nt14; do
  if [ $tag = nt14 ]; then export JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub; fi
  echo "$(date +%T) $tag screen"
  uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/mg_${tag}_screen.yaml --chunk-size 2000 > $L/mg_${tag}_screen.log 2>&1
  uv run --project envs/gen python scripts/explore/mg_known.py --screen data/explore/models_grader/${tag}_screen \
    --pool data/explore/known_items.jsonl --setting ${tag^^} --out data/explore/models_grader/${tag}_known_items.jsonl
  echo "$(date +%T) $tag arms"
  uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/mg_${tag}_arms.yaml --chunk-size 3000 > $L/mg_${tag}_arms.log 2>&1
  python3 scripts/explore/mg_local_rates.py --labels data/explore/models_grader/${tag}_arms/labels.jsonl --out data/explore/models_grader/${tag}_arms/rates.json
done
echo "$(date +%T) done"
