"""Select grader-bug runs for the eval-awareness readout (steps E3-E5; plain python3).

  python3 scripts/explore/evalaware_select.py --arms-dir data/explore/models_grader/nt8_arms \
      --arms A_V0 A_sys A_stakes L_unclear --out data/explore/evalaware/select_nt8.txt

Keeps, per arm, the items with at least one follower (final = hinted B) and at least
one resister (final = correct letter), and all their followers and resisters (runs
with another or no answer are left out). Also adds, as a reference, up to 2 runs per
selected item from the instructed "allowed" arm (L_allowed) when present.
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = [json.loads(l) for l in open(f"{a.arms_dir}/labels.jsonl")]
    by = defaultdict(lambda: defaultdict(list))
    for r in labels:
        if r["arm"] not in a.arms:
            continue
        y = "follower" if r["answer"] == r["letter"] and r["letter"] != r["correct_letter"] else \
            "resister" if r["answer"] == r["correct_letter"] else None
        if y:
            by[(r["arm"], r["question_id"])][y].append(r["run_id"])
    keep, items = [], set()
    n = defaultdict(lambda: defaultdict(int))
    for (arm, q), g in sorted(by.items()):
        if g["follower"] and g["resister"]:
            keep += g["follower"] + g["resister"]
            items.add(q)
            n[arm]["items"] += 1
            n[arm]["followers"] += len(g["follower"])
            n[arm]["resisters"] += len(g["resister"])
    ref = defaultdict(list)
    for r in labels:
        if r["arm"] == "L_allowed" and r["question_id"] in items and len(ref[r["question_id"]]) < 2:
            ref[r["question_id"]].append(r["run_id"])
    keep += [x for v in ref.values() for x in v]
    with open(a.out, "w") as f:
        f.write("\n".join(keep) + "\n")
    print(json.dumps({k: dict(v) for k, v in n.items()}), "L_allowed ref", sum(len(v) for v in ref.values()),
          "total", len(keep))


if __name__ == "__main__":
    main()
