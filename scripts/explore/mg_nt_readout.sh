#!/bin/bash
# Models x grader-bug follow-up: lens readouts of the non-thinking arms (Qwen3-8B, then Qwen3-14B).
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
for tag in nt8 nt14; do
  if [ $tag = nt14 ]; then export JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub; fi
  A=data/explore/models_grader/${tag}_arms
  python3 scripts/explore/mg_select_readout.py --runs $A/runs.jsonl --labels $A/labels.jsonl --out $A/readout_runs.jsonl
  echo "$(date +%T) $tag readout"
  uv run --project envs/lens python scripts/lens/readout_runs.py --runs $A/readout_runs.jsonl --out data/readouts/mg_$tag \
    --concept-tokens configs/concept_tokens_grader.json --validate 3 > data/logs/mg_${tag}_readout.log 2>&1
  uv run --project envs/lens python scripts/lens/analyze_models_grader.py --readouts data/readouts/mg_$tag \
    --runs $A/readout_runs.jsonl --out data/readouts/mg_${tag}_analysis.json > data/logs/mg_${tag}_analysis.log 2>&1
  echo "$(date +%T) $tag done; $(df -h /root | tail -1)"
done
