"""F1-T: pick the runs to read out (stdlib). Mirrors scripts/lens/instrument_select.py for the wording and C3
groups: known items only; wording = all followers (answer == hint letter) plus, per arm, as many resisters
(answer == correct) sampled with random.Random(0) from run_id-sorted lists; C3 = kept (answer == planted letter)
and recovered (answer == correct) runs of arm C3_think_prefill.
    python3 scripts/lens/f1t_select.py --data data/explore/f1t --items data/f1t/items_w.jsonl --out DIR
Writes DIR/wording.jsonl, DIR/c3.jsonl (runs.jsonl records) and DIR/selection.json.
"""
import argparse, json, random
from collections import defaultdict, Counter
from pathlib import Path


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/explore/f1t")
ap.add_argument("--known", nargs="+", required=True, help="items jsonl files (known pool, acc.A >= 0.9)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
D, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
known = {d["id"] for f in a.known for d in rows(f) if (d.get("acc") or {}).get("A", 0) >= 0.9}
rng = random.Random(0)
summary = {"known_items": len(known)}

# wording
lab = [d for d in rows(D / "wording/labels.jsonl") if d["question_id"] in known and d["answer"]] if (D / "wording/labels.jsonl").exists() else []
by_arm = defaultdict(lambda: ([], []))
for d in lab:
    if d["answer"] == d["hint_letter"]:
        by_arm[d["arm"]][0].append(d)
    elif d["answer"] == d["correct_letter"]:
        by_arm[d["arm"]][1].append(d)
chosen = {}
for arm in sorted(by_arm):
    sw, rs = by_arm[arm]
    for d in sw:
        chosen[d["run_id"]] = "wording_switch"
    for d in rng.sample(sorted(rs, key=lambda d: d["run_id"]), min(len(sw), len(rs))):
        chosen[d["run_id"]] = "wording_resisted"
    summary[f"wording/{arm}"] = [len(sw), len(rs)]
n = 0
with open(out / "wording.jsonl", "w") as f:
    for r in (rows(D / "wording/runs.jsonl") if lab else []):
        g = chosen.get(r["run_id"])
        if g:
            r["instr_group"], r["source_set"] = g, "f1t/wording"
            f.write(json.dumps(r) + "\n"); n += 1
summary["wording_selected"] = n
# c3
c3 = {}
for d in rows(D / "c3/labels.jsonl"):
    if d["arm"] != "C3_think_prefill" or d["question_id"] not in known or not d["answer"]:
        continue
    if d["answer"] == d["letter"]:
        c3[d["run_id"]] = "c3_kept_b"
    elif d["answer"] == d["correct_letter"]:
        c3[d["run_id"]] = "c3_recovered"
n = 0
with open(out / "c3.jsonl", "w") as f:
    for r in rows(D / "c3/runs.jsonl"):
        g = c3.get(r["run_id"])
        if g:
            r["instr_group"], r["source_set"] = g, "f1t/c3"
            f.write(json.dumps(r) + "\n"); n += 1
summary["c3_selected"] = dict(Counter(c3.values()))
(out / "selection.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
