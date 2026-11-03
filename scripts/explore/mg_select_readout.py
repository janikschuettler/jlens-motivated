"""Pick the runs to read out for the models x grader-bug lens step (disk budget on octavia).

  python scripts/explore/mg_select_readout.py --runs data/explore/models_grader/q14_arms/runs.jsonl \
      --labels data/explore/models_grader/q14_arms/labels.jsonl --out data/explore/models_grader/q14_arms/readout_runs.jsonl

Rule (fixed before any readout, models_grader lens plan):
  - every run whose final answer is the hinted letter B, in every arm (followers);
  - resisted runs (final = correct) of A_V0, A_stakes and L_unclear on items that have at least one
    follower in that arm: up to 3 per item and arm (lowest sample index first);
  - A_V1_null (matched null): sample 0 of every item (correct runs).
Runs with no parsed answer or truncated are not read out (counted in the printout).
"""

import argparse
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-resisted", type=int, default=3)
    a = ap.parse_args()
    lab = {}
    for line in open(a.labels):
        r = json.loads(line)
        lab[r["run_id"]] = r
    fol_items = defaultdict(set)
    for r in lab.values():
        if r["answer"] is not None and r["answer"] == r["letter"]:
            fol_items[r["arm"]].add(r["question_id"])
    keep, taken, why = set(), defaultdict(int), Counter()
    for rid, r in sorted(lab.items(), key=lambda kv: (kv[1]["arm"], kv[1]["question_id"], kv[1]["sample_idx"])):
        arm, q, ans = r["arm"], r["question_id"], r["answer"]
        if ans is None or r["finish_reason"] != "stop":
            why["no answer / truncated"] += 1
            continue
        if ans == r["letter"]:
            keep.add(rid); why[f"{arm} followed"] += 1
        elif arm in ("A_V0", "A_stakes", "L_unclear") and ans == r["correct_letter"] and q in fol_items[arm] \
                and taken[(arm, q)] < a.max_resisted:
            taken[(arm, q)] += 1
            keep.add(rid); why[f"{arm} resisted"] += 1
        elif arm == "A_V1_null" and r["sample_idx"] == 0 and ans == r["correct_letter"]:
            keep.add(rid); why[f"{arm} correct s0"] += 1
    n = 0
    with open(a.out, "w") as f:
        for line in open(a.runs):
            if json.loads(line)["run_id"] in keep:
                f.write(line)
                n += 1
    for k, v in sorted(why.items()):
        print(f"{k:28s} {v}")
    print(f"{n} runs -> {a.out}")


if __name__ == "__main__":
    main()
