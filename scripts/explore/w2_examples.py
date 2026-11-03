"""Print full examples from a wave-2 run dir for reading by eye (CPU). Never truncates the CoT
unless --max-chars is given.

uv run --project envs/gen python scripts/explore/w2_examples.py data/explore/w2/grader --arm V0_bug --outcome followed --n 3
uv run --project envs/gen python scripts/explore/w2_examples.py data/explore/w2/instructed --arm I3_conceal --where cot_silent=true
"""

import argparse
import json
import random
from pathlib import Path

from jmr import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--arm")
    p.add_argument("--outcome")
    p.add_argument("--where", nargs="*", default=[], help="label=value (true/false/str)")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-chars", type=int)
    args = p.parse_args()
    d = Path(args.dir) if Path(args.dir).is_absolute() else config.REPO_ROOT / args.dir
    labels = {}
    for line in (d / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if args.arm and r["arm"] != args.arm:
            continue
        if args.outcome and r["outcome"] != args.outcome:
            continue
        ok = True
        for w in args.where:
            k, v = w.split("=", 1)
            val = {"true": True, "false": False}.get(v, v)
            got = r.get(k)
            ok &= (bool(got) == val) if isinstance(val, bool) else (str(got) == val)
        if ok:
            labels[r["run_id"]] = r
    ids = sorted(labels)
    random.Random(args.seed).shuffle(ids)
    pick = set(ids[: args.n])
    print(f"{len(labels)} matching runs; showing {len(pick)}")
    with open(d / "runs.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["run_id"] not in pick:
                continue
            lab = labels[row["run_id"]]
            text = row["completion_text"]
            if args.max_chars and len(text) > args.max_chars:
                text = text[: args.max_chars // 2] + "\n[...]\n" + text[-args.max_chars // 2 :]
            print("=" * 100)
            print(row["run_id"])
            print({k: v for k, v in lab.items() if k not in ("run_id",)})
            for m in row["messages"]:
                print(f"--- {m['role']}:\n{m['content']}")
            if row.get("w2", {}).get("prefill_text"):
                print(f"--- prefill:\n{row['w2']['prefill_text']}")
            print(f"--- completion:\n{text}")


if __name__ == "__main__":
    main()
