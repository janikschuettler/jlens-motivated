#!/usr/bin/env bash
# F1-T stage 2 on derpy (C3 only): screen fresh items, build the known pool, C3 generation, readout, features.
set -euo pipefail
unset VIRTUAL_ENV; export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../.."
L=data/logs; RD=data/readouts; FE=data/f1t/feat_s2; mkdir -p $FE
G="uv run --project envs/gen python"
echo "$(date +%T) screen"; $G scripts/explore/run_arms.py configs/explore/f1t_screen2.yaml --chunk-size 1600 > $L/f1t2_screen.log 2>&1
$G scripts/explore/w2_known.py data/explore/f1t2/screen2 --arm think_qwen_unhinted_ours --setting A --out data/f1t/screen2_known_all.jsonl
python3 - <<P
import json
old = {json.loads(l)["id"] for l in open("data/f1t/orig_known_items.jsonl")}
rows = [json.loads(l) for l in open("data/f1t/screen2_known_all.jsonl")]
k = [d for d in rows if d["acc"]["A"] >= 0.9 and d["id"] not in old][:150]
open("data/f1t/items2.jsonl", "w").write("".join(json.dumps(d) + "\n" for d in k))
print("screened", len(rows), "in old 200:", sum(d["id"] in old for d in rows), "fresh known:", len(k))
P
echo "$(date +%T) c3"; $G scripts/explore/run_w2.py configs/explore/f1t2_c3.yaml --chunk-size 496 > $L/f1t2_c3.log 2>&1
echo "$(date +%T) select"; python3 scripts/lens/f1t_select.py --data data/explore/f1t2 --known data/f1t/items2.jsonl --out data/f1t/sel_s2
echo "$(date +%T) readout c3"; uv run --project envs/lens python scripts/lens/readout_runs.py --runs data/f1t/sel_s2/c3.jsonl --out $RD/f1t_s2_c3 --validate 3 > $L/f1t_s2_ro_c3.log 2>&1
echo "$(date +%T) features"
uv run --project envs/lens python scripts/lens/competitors_extract.py --kind c3 --readouts $RD/f1t_s2_c3 --labels data/explore/f1t2/c3/labels.jsonl --known data/f1t/items2.jsonl --out $FE/c3.npz --cot-layers "" --workers 16
echo "$(date +%T) stage2 done"
