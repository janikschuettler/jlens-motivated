#!/bin/bash
# Models x grader-bug Phase 2 on octavia: Qwen3-14B readout of the selected runs, then hand the GPU on.
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
export JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub
A=data/explore/models_grader/q14_arms
python3 scripts/explore/mg_select_readout.py --runs $A/runs.jsonl --labels $A/labels.jsonl --out $A/readout_runs.jsonl
python3 scripts/explore/mg_local_rates.py --labels $A/labels.jsonl --out $A/rates.json
echo "$(date +%T) readout"
uv run --project envs/lens python scripts/lens/readout_runs.py --runs $A/readout_runs.jsonl --out data/readouts/mg_q14 \
  --concept-tokens configs/concept_tokens_grader.json --validate 3 > data/logs/mg_q14_readout.log 2>&1
echo "$(date +%T) readout done; disk: $(df -h /root | tail -1)"
echo "gptoss $(date +%T) handed over by the models_grader parent agent (Qwen3-14B readout done)" > data/logs/GPU_OWNER
