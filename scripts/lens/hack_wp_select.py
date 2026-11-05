"""Run-ID lists for the W readouts (docs/preregistration.md, "W"). CPU only.

- b002_read.txt: hiding adapter, usable no-hack runs + up to 3 usable hidden-hack runs
  per problem that has a usable no-hack run (lowest sample index first).
- base40.txt: 40 fresh warm-start runs for lens validity (ii): the first 20 usable hidden
  and first 20 usable no-hack runs in problem order (lowest sample index first).
- orig_disp.txt: A8, usable hidden and no-hack runs of the original 480 warm-start runs.

python scripts/lens/hack_wp_select.py --out data/hack_wp/sel
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def usable(gen: Path, arm: str):
    labels = {json.loads(l)["run_id"]: json.loads(l) for l in open(gen / "labels.jsonl")}
    out = []
    for l in open(gen / "labels.jsonl"):
        j = json.loads(l)
        if j["arm"] == arm and j["finish_reason"] == "stop" and j["group"] in ("hidden", "no_hack"):
            k = int(j["run_id"].rsplit("/", 1)[1])
            out.append((j["problem_id"], k, j["run_id"], j["group"]))
    return out, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="data/hack_wp/items.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = {json.loads(l)["problem_id"]: i for i, l in enumerate(open(args.items))}

    rows, _ = usable(Path("data/hack_wp/gen_b002"), "beta002")
    by = defaultdict(lambda: {"hidden": [], "no_hack": []})
    for p, k, rid, g in rows:
        by[p][g].append((k, rid))
    sel = []
    for p, d in by.items():
        if d["no_hack"]:
            sel += [rid for _, rid in sorted(d["no_hack"])]
            sel += [rid for _, rid in sorted(d["hidden"])[:3]]
    (out / "b002_read.txt").write_text("\n".join(sel) + "\n")

    rows, _ = usable(Path("data/hack_wp/gen_disp"), "disposition")
    rows.sort(key=lambda x: (order[x[0]], x[1]))
    base40 = [r[2] for r in rows if r[3] == "hidden"][:20] + [r[2] for r in rows if r[3] == "no_hack"][:20]
    (out / "base40.txt").write_text("\n".join(base40) + "\n")

    rows, _ = usable(Path("data/hack/gen"), "disposition")
    (out / "orig_disp.txt").write_text("\n".join(r[2] for r in rows) + "\n")
    print(json.dumps({"b002": len(sel), "b002_no_hack": sum(len(d["no_hack"]) for d in by.values()),
                      "base40": len(base40), "orig_disp": len(rows)}))


if __name__ == "__main__":
    main()
