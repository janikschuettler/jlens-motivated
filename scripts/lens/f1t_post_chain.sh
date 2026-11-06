#!/usr/bin/env bash
# F1-T on derpy after generation: re-asking forks, selection, readouts (validated), features. Stops at first failure.
# usage: f1t_post_chain.sh TAG KNOWN_ITEMS_W KNOWN_ITEMS_C3 [WORDING_RUNS_DIR C3_RUNS_DIR]
set -euo pipefail
unset VIRTUAL_ENV; export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
TAG=$1; IW=$2; IC=$3; G=data/explore/${4:-f1t}; L=data/logs; RD=data/readouts; FE=data/f1t/feat_$TAG
mkdir -p $L $FE
if [ ! -s data/explore/f1t_$TAG/forks_reelicit/forks.jsonl ]; then echo "$(date +%T) forks"; uv run --project envs/gen python scripts/explore/run_forks.py configs/explore/f1t_forks_t2.yaml --runs $G/wording/runs.jsonl --name f1t_$TAG/forks_reelicit > $L/f1t_${TAG}_forks.log 2>&1; fi
echo "$(date +%T) select"; python3 scripts/lens/f1t_select.py --data $G --known $IW $IC --out data/f1t/sel_$TAG
echo "$(date +%T) readout wording"; uv run --project envs/lens python scripts/lens/readout_runs.py --runs data/f1t/sel_$TAG/wording.jsonl --out $RD/f1t_${TAG}_wording --validate 3 > $L/f1t_${TAG}_ro_wording.log 2>&1
echo "$(date +%T) readout c3"; uv run --project envs/lens python scripts/lens/readout_runs.py --runs data/f1t/sel_$TAG/c3.jsonl --out $RD/f1t_${TAG}_c3 --validate 3 > $L/f1t_${TAG}_ro_c3.log 2>&1
echo "$(date +%T) features"
E=scripts/lens/competitors_extract.py
uv run --project envs/lens python $E --kind wording --readouts $RD/f1t_${TAG}_wording --labels $G/wording/labels.jsonl --known $IW --out $FE/wording.npz --cot-layers "" --workers 16
uv run --project envs/lens python $E --kind c3 --readouts $RD/f1t_${TAG}_c3 --labels $G/c3/labels.jsonl --known $IC --out $FE/c3.npz --cot-layers "" --workers 16
uv run --project envs/lens python scripts/lens/f1t_t2_new.py --readouts $RD/f1t_${TAG}_wording --forks data/explore/f1t_$TAG/forks_reelicit/forks.jsonl --known $IW --out $FE/t2_new.jsonl
echo "$(date +%T) post done"
