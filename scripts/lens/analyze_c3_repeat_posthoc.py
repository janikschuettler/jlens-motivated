"""R-C3 post-hoc item checks (NOT pre-registered; run after the primary result was known).
AUROC of S restricted to items whose fresh-context control (question alone, thinking off) is
correct / wrong, and within items with both outcomes (S centred on the item mean).

uv run --project envs/lens python scripts/lens/analyze_c3_repeat_posthoc.py   # after analyze_c3_repeat.py
"""
import json, sys, collections
import numpy as np
sys.path.insert(0, "scripts/lens")
from analyze_main import auroc, auroc_boot
r = json.load(open("data/explore/c3rep/c3_lead_repeat.json"))
runs = r["runs"]
ctrl, cls = {}, collections.Counter()
for d in ["forks_reelicit_c3rep", "forks_reelicit_c3rep2"]:
    for f in map(json.loads, open(f"data/explore/w2/{d}/forks.jsonl")):
        if f["kind"] == "control":
            ctrl.setdefault(f["question_id"], []).append(f["argmax"] == f["correct_letter"])
        elif f["mode"] == "template":
            cls[(f["source_class"], f["argmax"] == f["correct_letter"])] += 1
print("template re-elicitation by source class (class, correct):", dict(cls))
ctrl = {k: all(v) for k, v in ctrl.items()}
print("controls per item consistent:", len(ctrl))
out = {}
idx = list(range(len(runs)))
items = {i: runs[i]["item"] for i in idx}
for lens in ("jlens", "logit"):
    sc = {i: runs[i]["S"][lens] for i in idx}
    for name, keep in (("fresh-context control correct", True), ("fresh-context control wrong", False)):
        sel = [i for i in idx if ctrl.get(runs[i]["item"]) == keep]
        pos = [i for i in sel if runs[i]["pos"]]; neg = [i for i in sel if not runs[i]["pos"]]
        out[f"{lens} / items where {name}"] = auroc_boot(sc, items, pos, neg, np.random.default_rng(0))
    # item-centred, items with both outcomes
    by = collections.defaultdict(list)
    for i in idx: by[runs[i]["item"]].append(i)
    both = [i for q, v in by.items() if len({runs[j]["pos"] for j in v}) == 2 for i in v]
    cen = {i: sc[i] - np.mean([sc[j] for j in by[runs[i]["item"]]]) for i in both}
    out[f"{lens} / within items (item-centred)"] = auroc_boot(cen, items, [i for i in both if runs[i]["pos"]],
                                                             [i for i in both if not runs[i]["pos"]], np.random.default_rng(0))
# how much of the outcome is the item: share of kept-B runs on items whose control is wrong, by group
for g in (True, False):
    sel = [x for x in runs if x["pos"] == g]
    print("pos" if g else "neg", "share on items with wrong fresh-context control:",
          round(np.mean([not ctrl.get(x["item"], True) for x in sel]), 3), len(sel))
print(json.dumps(out, indent=0))
json.dump({"posthoc": out, "template_by_class": {f"{k[0]}|{k[1]}": v for k, v in cls.items()}},
          open("data/explore/c3rep/posthoc.json", "w"), indent=1)
