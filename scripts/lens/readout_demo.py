"""Demo table from batch readouts (numpy only; either env): mean J-lens and
logit-lens log-prob of the correct and the hinted letter at one anchor, per
layer, per condition. A pipeline check, not an analysis.

uv run --project envs/lens python scripts/lens/readout_demo.py \
    --readouts data/readouts/temp_check --run-set data/runs/temp_check \
    --conditions unhinted known_evidence [--anchor pre_answer] [--form space]
"""

import argparse
from pathlib import Path

import numpy as np

from jmr.analysis.readouts import Readouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", required=True)
    parser.add_argument("--run-set", required=True, help="dir with labels.jsonl and items.jsonl")
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--anchor", default="pre_answer")
    parser.add_argument("--form", default="space")
    args = parser.parse_args()

    rs = Path(args.run_set)
    r = Readouts(args.readouts, labels=rs / "labels.jsonl", items=rs / "items.jsonl")
    print(f"{len(r)} runs read; anchor {args.anchor}, form {args.form!r}; mean log-prob (nats)\n")
    cols = []
    for cond in args.conditions:
        idx = r.select(condition=cond)
        for lens in ("jlens", "logit"):
            for role in ("correct", "hinted"):
                x = r.at_anchor(lens, role, args.anchor, form=args.form, runs=idx)
                cols.append((f"{cond[:10]} {lens} {role[:4]}", x, int(np.isfinite(x[:, 0]).sum())))
    print("| layer | " + " | ".join(c[0] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    print("| n | " + " | ".join(str(c[2]) for c in cols) + " |")
    for li, layer in enumerate(r.layers):
        print(f"| {layer} | " + " | ".join(f"{np.nanmean(c[1][:, li]):.2f}" for c in cols) + " |")
    # The model's own distribution (block 35) at the same anchor.
    row = []
    for cond in args.conditions:
        idx = r.select(condition=cond)
        for role in ("correct", "hinted"):
            row.append(f"{np.nanmean(r.at_anchor('model', role, args.anchor, form=args.form, runs=idx)):.2f}")
    print("\nmodel (block 35), per condition correct/hinted: " + ", ".join(row))


if __name__ == "__main__":
    main()
