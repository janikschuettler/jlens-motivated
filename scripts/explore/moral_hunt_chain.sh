#!/usr/bin/env bash
# Moral hunt chain (mayor, overnight): screen -> known pool (merged with octavia's
# w2_moral_screen MoralChoice known items, copied to data/explore/moral_hunt/octavia/) -> M&B hint arm.
# Usage: nohup scripts/explore/moral_hunt_chain.sh DEADLINE_HH:MM > data/explore/logs/moral_hunt.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
DL="${1:-05:57}"
R="uv run --project envs/gen python"
$R scripts/explore/run_w2.py configs/explore/moral_hunt_screen.yaml --chunk-size 3235
$R scripts/explore/w2_known.py data/explore/moral_hunt/screen --arm screen --setting A --out data/explore/moral_hunt/screen_known.jsonl
$R - <<'PY'
import json
from pathlib import Path
d = Path("data/explore/moral_hunt")
old = [json.loads(x) for x in (d / "octavia/moral_known.jsonl").read_text().splitlines() if x.strip()]
new = [json.loads(x) for x in (d / "screen_known.jsonl").read_text().splitlines() if x.strip()]
old = [r for r in old if r["dataset"] == "moralchoice"]
ids = {r["id"] for r in old}
assert not ids & {r["id"] for r in new}, "overlap between octavia's screen and the new screen"
rows = old + new
(d / "known.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
k = [r for r in rows if r["acc"]["A"] >= 0.9]
print(f"pool: {len(old)} octavia + {len(new)} new screened; known {len(k)} "
      f"({sum(r['id'] in ids for r in k)} octavia, {sum(r['id'] not in ids for r in k)} new)")
PY
$R scripts/explore/run_w2.py configs/explore/moral_hunt_nudge.yaml --chunk-size 500 --deadline "$DL"
echo CHAIN DONE
