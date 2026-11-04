#!/usr/bin/env bash
# Slot 2, restarted 03:10: cap 2,048 tokens, order unsteered + strength 0.2 first, then 0.8 (see report).
set -euo pipefail
cd /root/jlens-motivated
unset VIRTUAL_ENV
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date +%H:%M:%S) generation restart, cap 2048, priorities 0-1 (git $(git rev-parse --short HEAD))"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> data/logs/steer_gpu_mem2.log; sleep 5; done ) &
MEM=$!
trap 'kill $MEM 2>/dev/null || true' EXIT
uv run --project envs/lens python -c "import transformers, torch; print('attn check', torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())"
uv run --project envs/lens python scripts/lens/steer_generate.py --run data/steer/gen/jobs_main.jsonl --dirs data/steer/directions.pt --out data/steer/gen --max-priority 0 --max-new 2048
echo "$(date +%H:%M:%S) part 2 priority 0 (unsteered + strength 0.2) done"
uv run --project envs/lens python scripts/lens/steer_generate.py --run data/steer/gen/jobs_main.jsonl --dirs data/steer/directions.pt --out data/steer/gen --max-priority 1 --max-new 2048
echo "$(date +%H:%M:%S) part 2 priority 1 done; peak GPU MiB $(sort -n data/logs/steer_gpu_mem2.log | tail -1)"
echo CHAIN3_DONE
