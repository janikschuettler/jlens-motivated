"""Eval-awareness models screen: "testing me" counts in local grader-bug runs (octavia, read-only).

  python3 -I scripts/explore/evalaware_models_local_counts.py data/explore/models_grader/q14_arms

Uses TEST_ME from evalaware_models_analyze.py (searched in the thinking span, before </think>)
and the run set's labels.jsonl for the final answer. Prints JSON: per arm, runs, "testing me"
hits, and followers (answer == hinted letter) with and without a hit.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalaware_models_analyze import TEST_ME  # noqa: E402

d = Path(sys.argv[1])
lab = {json.loads(l)["run_id"]: json.loads(l) for l in open(d / "labels.jsonl")}
c = Counter()
for l in open(d / "runs.jsonl"):
    r = json.loads(l)
    think = r["completion_text"].split("</think>")[0]
    hit = any(p.search(think) for p in TEST_ME)
    fol = lab.get(r["run_id"], {}).get("answer") == r["hint_letter"]
    a = r["condition"]
    c[(a, "n")] += 1
    c[(a, "test_me")] += hit
    c[(a, "followed_with_test_me")] += hit and fol
    c[(a, "followed_without_test_me")] += (not hit) and fol
out = {}
for (a, k), v in sorted(c.items()):
    out.setdefault(a, {})[k] = v
print(json.dumps({"run_set": str(d), "counts": out}, indent=1))
