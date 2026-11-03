"""Pick the gpt-oss-20b runs to read out (models x grader-bug Phase 2b; disk budget on octavia).

  python scripts/explore/mg_gptoss_select.py --runs data/explore/models_grader/gptoss_arms/runs.jsonl \
      --labels data/explore/models_grader/gptoss_arms/labels.jsonl --out data/explore/models_grader/gptoss_arms/readout_runs.jsonl

Rule (fixed before any readout; docs/overnight/notes/gptoss_phase2.md):
  - every follower (final = the hinted letter B) in A_V0, A_sys, V0 and L_forbidden;
  - L_allowed followers: sample 0 of every item (the sanctioned reference);
  - resisted runs (final = correct) of A_V0 and A_sys on items with at least one follower in that
    arm: up to 3 per item and arm (lowest sample index first);
  - A_V1_null: sample 0 of every item, if correct.
Runs with no parsed answer or not finished ("stop") are not read out (counted in the printout).
"""

import argparse
import json
from collections import Counter, defaultdict

UNSANCTIONED = ("A_V0", "A_sys")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-resisted", type=int, default=3)
    a = ap.parse_args()
    lab = {json.loads(l)["run_id"]: json.loads(l) for l in open(a.labels)}
    fol_items = defaultdict(set)
    for r in lab.values():
        if r["answer"] is not None and r["answer"] == r["letter"]:
            fol_items[r["arm"]].add(r["question_id"])
    keep, taken, why = set(), defaultdict(int), Counter()
    for rid, r in sorted(lab.items(), key=lambda kv: (kv[1]["arm"], kv[1]["question_id"], kv[1]["sample_idx"])):
        arm, q, ans = r["arm"], r["question_id"], r["answer"]
        if ans is None or r["finish_reason"] != "stop":
            why["no answer / not stop"] += 1
            continue
        if ans == r["letter"] and arm in UNSANCTIONED + ("V0", "L_forbidden"):
            keep.add(rid); why[f"{arm} followed"] += 1
        elif ans == r["letter"] and arm == "L_allowed" and r["sample_idx"] == 0:
            keep.add(rid); why[f"{arm} followed s0"] += 1
        elif arm in UNSANCTIONED and ans == r["correct_letter"] and q in fol_items[arm] \
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
