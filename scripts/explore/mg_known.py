"""Known-items file for a local model's screen (models x grader-bug, Phase 2).

  python scripts/explore/mg_known.py --screen data/explore/models_grader/q14_screen \
      --pool data/explore/known_items.jsonl --setting Q14 --out data/explore/models_grader/q14_known_items.jsonl

Writes every pool item with acc[setting] = (correct / samples) from the screen's labels.jsonl
(unanswered runs count as not correct). The arms spec then filters with min_acc 0.9.
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n, c = defaultdict(int), defaultdict(int)
    for line in open(f"{a.screen}/labels.jsonl"):
        r = json.loads(line)
        n[r["question_id"]] += 1
        c[r["question_id"]] += r["answer"] == r["correct_letter"]
    rows = [json.loads(l) for l in open(a.pool)]
    known = 0
    with open(a.out, "w") as f:
        for row in rows:
            if row["id"] not in n:
                continue
            acc = c[row["id"]] / n[row["id"]]
            row.setdefault("acc", {})[a.setting] = acc
            row.setdefault("n", {})[a.setting] = n[row["id"]]
            row.setdefault("n_correct", {})[a.setting] = c[row["id"]]
            known += acc >= 0.9
            f.write(json.dumps(row) + "\n")
    print(f"{len(n)} items screened, {known} with acc >= 0.9 -> {a.out}")


if __name__ == "__main__":
    main()
