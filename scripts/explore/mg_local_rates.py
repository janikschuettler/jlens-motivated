"""Per-arm outcome rates for a local models x grader-bug run set (w2 labels.jsonl), stdlib only.

  python scripts/explore/mg_local_rates.py --labels data/explore/models_grader/qwen8b/labels.jsonl --out X.json

P(final = B), P(correct), P(no answer) per arm with 95% item-bootstrap intervals; heuristic
grader_outcome counts of the followers (jmr.explore.w2_labels); mean completion tokens.
"""

import argparse
import json
import random
from collections import Counter, defaultdict


def boot(per_item, n=2000, seed=0):
    keys = sorted(per_item)
    if not keys:
        return None
    means = {k: sum(v) / len(v) for k, v in per_item.items()}
    point = sum(sum(v) for v in per_item.values()) / sum(len(v) for v in per_item.values())
    rng = random.Random(seed)
    st = sorted(sum(means[rng.choice(keys)] for _ in keys) / len(keys) for _ in range(n))
    return {"v": point, "lo": st[int(0.025 * n)], "hi": st[int(0.975 * n) - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by = defaultdict(lambda: defaultdict(list))
    outc, tok = defaultdict(Counter), defaultdict(list)
    for line in open(a.labels):
        r = json.loads(line)
        by[r["arm"]][r["question_id"]].append(r)
        tok[r["arm"]].append(r.get("completion_tokens") or 0)
        if r["answer"] is not None and r["answer"] == r["letter"]:
            outc[r["arm"]][r.get("grader_outcome")] += 1
    out = {}
    for arm, per in by.items():
        runs = [r for v in per.values() for r in v]
        out[arm] = {
            "runs": len(runs), "items": len(per),
            "p_B": boot({k: [float(r["answer"] is not None and r["answer"] == r["letter"]) for r in v] for k, v in per.items()}),
            "p_correct": boot({k: [float(r["answer"] == r["correct_letter"]) for r in v] for k, v in per.items()}),
            "p_none": boot({k: [float(r["answer"] is None) for r in v] for k, v in per.items()}),
            "n_B": sum(r["answer"] is not None and r["answer"] == r["letter"] for r in runs),
            "n_trunc": sum(r["finish_reason"] == "length" for r in runs),
            "follower_grader_outcome": dict(outc[arm]),
            "mean_tokens": round(sum(tok[arm]) / len(tok[arm])),
        }
    json.dump(out, open(a.out, "w"), indent=1)
    for arm, d in out.items():
        p = d["p_B"]
        print(f"{arm:14s} n={d['runs']:4d} items={d['items']:3d} B {100*p['v']:5.1f}% [{100*p['lo']:.1f}, {100*p['hi']:.1f}] "
              f"none {100*d['p_none']['v']:.1f}% {d['follower_grader_outcome']}")


if __name__ == "__main__":
    main()
