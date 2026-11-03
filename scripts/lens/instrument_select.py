"""Select the runs for the instrument check (Q1/Q3; stdlib only, any env).

Groups (thinking mode, home setting "ours"; known = >= 9/10 correct in the gap
screen's setting A, arm think_qwen_unhinted_ours):
  train_unhinted   screen A, known item, final = correct; TRAIN_PER_ITEM per item (seed 0)
  honest_error     screen A, item with 2-8/10 correct, final = a wrong letter (all)
  unstable_correct the same items, final = correct (up to UNSTABLE_PER_ITEM per item)
  kb_I1, kb_I3     w2/instructed I1_argue / I3_conceal, known item, final = the instructed letter B (all)
  i2_says_b        I2_debate, final = B (all)
  i2_knows         I2_debate, final = correct while arguing B (all)
  w2_unhinted      w2/instructed unhinted, final = correct (all)
  wording_switch   w2/wording, known item, final = B (all)        [for later use]
  wording_resisted w2/wording, known item, final = correct (per arm, as many as switches; seed 0)
  c3_kept_b, c3_recovered  w2/prefill C3_think_prefill, final = B / correct (all)

Runs are split by whether their item has content tokens (configs/content_tokens.json):
<out>/ct/runs.jsonl (read out with --content-tokens) and <out>/nc/runs.jsonl.
Each line is the original run record plus "instr_group" and "source_set"
(kept by the readout in meta["extras"]).

python3 scripts/lens/instrument_select.py --data data --out data/instrument [--groups ...]
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_PER_ITEM = 3
UNSTABLE_PER_ITEM = 3
SCREEN = "explore/gap_factorial/screen"
SCREEN_ARM = "think_qwen_unhinted_ours"
INSTRUCTED = "from_mayor/explore/w2/instructed"
WORDING = "explore/w2/wording"
PREFILL = "explore/w2/prefill"
PRIORITY = ["train_unhinted", "honest_error", "unstable_correct", "kb_I1", "kb_I3", "i2_says_b", "i2_knows",
            "w2_unhinted"]
LATER = ["wording_switch", "wording_resisted", "c3_kept_b", "c3_recovered"]


def rows(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p) if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--content", default="configs/content_tokens.json")
    ap.add_argument("--groups", nargs="+", default=PRIORITY + LATER)
    args = ap.parse_args()
    D, out = Path(args.data), Path(args.out)
    rng = random.Random(0)
    content = set(json.loads(Path(args.content).read_text())["items"])

    lab = [d for d in rows(D / SCREEN / "labels.jsonl") if d["arm"] == SCREEN_ARM]
    n_correct = Counter(d["question_id"] for d in lab if d["answer"] == d["correct_letter"])
    n_total = Counter(d["question_id"] for d in lab)
    if set(n_total.values()) != {10}:
        raise ValueError(f"screen A: samples per item {Counter(n_total.values())}, expected 10")
    known = {q for q in n_total if n_correct[q] >= 9}
    unstable = {q for q in n_total if 2 <= n_correct[q] <= 8}

    chosen: dict[str, str] = {}  # run_id -> group
    source: dict[str, str] = {}

    def take(rid, g, src):
        if g in args.groups:
            chosen[rid] = g
            source[rid] = src

    # screen A
    by_item = defaultdict(list)
    for d in lab:
        by_item[d["question_id"]].append(d)
    for q in sorted(by_item):
        ds = sorted(by_item[q], key=lambda d: d["sample_idx"])
        corr = [d for d in ds if d["answer"] == d["correct_letter"]]
        wrong = [d for d in ds if d["answer"] in ("A", "B", "C", "D") and d["answer"] != d["correct_letter"]]
        if q in known:
            for d in rng.sample(corr, min(TRAIN_PER_ITEM, len(corr))):
                take(d["run_id"], "train_unhinted", SCREEN)
        elif q in unstable:
            for d in wrong:
                take(d["run_id"], "honest_error", SCREEN)
            for d in rng.sample(corr, min(UNSTABLE_PER_ITEM, len(corr))):
                take(d["run_id"], "unstable_correct", SCREEN)

    # instructed
    for d in rows(D / INSTRUCTED / "labels.jsonl"):
        q, A, C, B, arm = d["question_id"], d["answer"], d["correct_letter"], d["letter"], d["arm"]
        if q not in known or A is None:
            continue
        if arm in ("I1_argue", "I3_conceal") and A == B:
            take(d["run_id"], "kb_I1" if arm == "I1_argue" else "kb_I3", INSTRUCTED)
        elif arm == "I2_debate":
            if A == B:
                take(d["run_id"], "i2_says_b", INSTRUCTED)
            elif A == C:
                take(d["run_id"], "i2_knows", INSTRUCTED)
        elif arm == "unhinted" and A == C:
            take(d["run_id"], "w2_unhinted", INSTRUCTED)

    # wording (known items): all switches, and per arm as many resisted runs
    wl = [d for d in rows(D / WORDING / "labels.jsonl") if d["question_id"] in known and d["answer"]]
    by_arm = defaultdict(lambda: ([], []))
    for d in wl:
        if d["answer"] == d["hint_letter"]:
            by_arm[d["arm"]][0].append(d)
        elif d["answer"] == d["correct_letter"]:
            by_arm[d["arm"]][1].append(d)
    for arm in sorted(by_arm):
        sw, rs = by_arm[arm]
        for d in sw:
            take(d["run_id"], "wording_switch", WORDING)
        for d in rng.sample(sorted(rs, key=lambda d: d["run_id"]), min(len(sw), len(rs))):
            take(d["run_id"], "wording_resisted", WORDING)

    # C3
    for d in rows(D / PREFILL / "labels.jsonl"):
        if d["arm"] != "C3_think_prefill" or d["question_id"] not in known or not d["answer"]:
            continue
        if d["answer"] == d["letter"]:
            take(d["run_id"], "c3_kept_b", PREFILL)
        elif d["answer"] == d["correct_letter"]:
            take(d["run_id"], "c3_recovered", PREFILL)

    (out / "ct").mkdir(parents=True, exist_ok=True)
    (out / "nc").mkdir(parents=True, exist_ok=True)
    counts = Counter()
    n_written = 0
    with open(out / "ct" / "runs.jsonl", "w") as fct, open(out / "nc" / "runs.jsonl", "w") as fnc:
        for src in (SCREEN, INSTRUCTED, WORDING, PREFILL):
            for line in open(D / src / "runs.jsonl"):
                r = json.loads(line)
                g = chosen.get(r["run_id"])
                if g is None or source[r["run_id"]] != src:
                    continue
                r["instr_group"], r["source_set"] = g, src
                ct = r["question_id"] in content
                (fct if ct else fnc).write(json.dumps(r) + "\n")
                counts[(g, "ct" if ct else "nc")] += 1
                n_written += 1
    if n_written != len(chosen):
        raise ValueError(f"wrote {n_written} runs, selected {len(chosen)}")
    summary = {"known_items": len(known), "unstable_items": len(unstable),
               "counts": {f"{g}/{s}": n for (g, s), n in sorted(counts.items())}, "total": n_written}
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
