#!/usr/bin/env bash
# C3-R on octavia. usage: c3r_chain.sh stage1   (items, thinking-off screen, batches 1-2, forks)
#                        c3r_chain.sh topup     (items with kept-B in 1-2, batches 3-4, forks)
#                        c3r_chain.sh readout B... (kept-B subset and readout per batch)
set -euo pipefail
unset VIRTUAL_ENV; export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
L=data/explore/logs; G="uv run --project envs/gen python"
ts() { echo "$(date -u +%H:%M:%S) $*"; }
gen() { b=$1; items=$2; n=$(wc -l < $items); ts gen batch $b on $n items
  $G scripts/explore/run_w2.py configs/explore/w2_prefill.yaml --arms C3_think_prefill --items-path $items --limit-items $n --samples 6 --out-suffix _c3r$b --chunk-size 1000 > $L/c3r_gen$b.log 2>&1
  ts forks $b; $G scripts/explore/run_forks.py configs/explore/c3r_forks$b.yaml > $L/c3r_forks$b.log 2>&1; }
case $1 in
stage1)
  while pgrep -f c3r_screen.yaml >/dev/null; do sleep 20; done
  ts items; $G scripts/explore/w2_known.py data/explore/c3r/screen --arm think_qwen_unhinted_ours --setting A --out data/explore/c3r/known_all.jsonl
  python3 scripts/explore/c3r_stage0.py items
  ts nothink; $G scripts/explore/run_arms.py configs/explore/c3r_nothink.yaml --chunk-size 4000 > $L/c3r_nothink.log 2>&1
  $G scripts/explore/w2_known.py data/explore/c3r/nothink --arm nothink_mb_unhinted_mb --setting B --out data/explore/c3r/nothink_known.jsonl
  python3 scripts/explore/c3r_stage0.py merge
  gen 1 data/explore/c3r/items_with_B.jsonl
  gen 2 data/explore/c3r/items_with_B.jsonl
  ts stage1 done;;
topup)
  python3 - <<'PY'
import json
from collections import Counter
kept = Counter()
for b in (1, 2):
    for l in open(f"data/explore/w2/prefill_c3r{b}/labels.jsonl"):
        d = json.loads(l)
        if d["outcome"] == "followed": kept[d["question_id"]] += 1
rows = [l for l in open("data/explore/c3r/items_with_B.jsonl") if json.loads(l)["id"] in kept]
open("data/explore/c3r/items_topup.jsonl", "w").write("".join(rows))
print(len(rows), "items with >=1 kept-B in batches 1-2;", sum(kept.values()), "kept-B runs")
PY
  gen 3 data/explore/c3r/items_topup.jsonl
  gen 4 data/explore/c3r/items_topup.jsonl
  ts topup done;;
readout)
  shift
  for b in "$@"; do
    python3 - $b <<'PY'
import json, os, sys
b = sys.argv[1]
lab = {json.loads(l)["run_id"]: json.loads(l) for l in open(f"data/explore/w2/prefill_c3r{b}/labels.jsonl")}
keep = [l for l in open(f"data/explore/w2/prefill_c3r{b}/runs.jsonl") if lab[json.loads(l)["run_id"]]["outcome"] == "followed"]
os.makedirs(f"data/explore/w2/lens_sets/c3r{b}_keptB", exist_ok=True)
open(f"data/explore/w2/lens_sets/c3r{b}_keptB/runs.jsonl", "w").write("".join(keep))
print(b, len(keep), "kept-B runs")
PY
    ts readout $b; uv run --project envs/lens python scripts/lens/readout_runs.py --runs data/explore/w2/lens_sets/c3r${b}_keptB/runs.jsonl --out data/readouts/c3r${b}_keptB --validate 3 > $L/c3r_readout$b.log 2>&1
  done; ts readout done;;
esac
