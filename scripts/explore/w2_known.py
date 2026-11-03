"""Known-item pool from a wave-2 screen (e.g. the E6 moral screen), in the format
of data/explore/known_items.jsonl (item row + n_correct/n/acc per setting), so a
later spec can use items: {source: known_pool, path: ..., setting: A, min_acc: 0.9}.
The count goes to n_correct: `correct` stays the Question's correct letter.

Accuracy = correct runs / all runs of the arm (unanswered runs count as not correct),
as in scripts/explore/analyze_gap.py.

uv run --project envs/gen python scripts/explore/w2_known.py data/explore/w2_moral_screen \
    --arm screen --setting A --out data/explore/w2_moral_known.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from jmr import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--arm", required=True)
    p.add_argument("--setting", default="A")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    d = Path(args.dir) if Path(args.dir).is_absolute() else config.REPO_ROOT / args.dir
    items = [json.loads(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    labels = [json.loads(x) for x in (d / "labels.jsonl").read_text().splitlines() if x.strip()]
    labels = [r for r in labels if r["arm"] == args.arm]
    n, k = Counter(), Counter()
    for r in labels:
        n[r["question_id"]] += 1
        k[r["question_id"]] += r["answer"] == r["correct_letter"]
    out = []
    for row in items:
        q = row["id"]
        if n[q]:
            out.append({**row, "n_correct": {args.setting: k[q]}, "n": {args.setting: n[q]},
                        "acc": {args.setting: k[q] / n[q]}})
    path = Path(args.out) if Path(args.out).is_absolute() else config.REPO_ROOT / args.out
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    known = [r for r in out if r["acc"][args.setting] >= 0.9]
    by = Counter(r["dataset"] + ("/" + r["subject"] if r["dataset"] == "mmlu_redux" else "") for r in known)
    print(f"{len(out)} items screened, {len(known)} with >= 9/10 correct: {dict(by)}; wrote {path}")


if __name__ == "__main__":
    main()
