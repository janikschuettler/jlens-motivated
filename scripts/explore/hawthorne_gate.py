"""Behaviour gate for the 14B test-awareness replication (CPU).
Awareness = regex on reasoning (AWARE_RE, from hawthorne_analyze.py). Compliance: actions = tool call
after </think> (paper's rule); triggers = judge verdict in data/hawthorne/q14/compl_judge.jsonl if present.
  python scripts/explore/hawthorne_gate.py data/hawthorne/q14 docs/experiments/test-awareness-14b/data/gate.json
"""
import importlib.util, json, random, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("ha", Path(__file__).with_name("hawthorne_analyze.py"))
ha = importlib.util.module_from_spec(spec); spec.loader.exec_module(ha)
d, out = Path(sys.argv[1]), sys.argv[2]
cj = {}
if (d / "compl_judge.jsonl").exists():
    for l in open(d / "compl_judge.jsonl"):
        x = json.loads(l); cj[x["run_id"]] = x["performed"]
rows = []
for l in open(d / "runs.jsonl"):
    r = json.loads(l)
    reas, ans = ha.split(r["completion_text"])
    row = dict(run_id=r["run_id"], q=r["question_id"], ds=r["dataset"], var=r["variant"], done=reas is not None)
    if reas is not None:
        row["aware"] = bool(ha.AWARE_RE.search(reas))
        row["nreas"] = len(r["completion_token_ids"])
        if r["dataset"] == "actions":
            row["comp"] = bool(ha.TOOL_RE.search(ans))
        elif r["run_id"] in cj:
            row["comp"] = cj[r["run_id"]]
    rows.append(row)
json.dump(rows, open(out.replace("gate.json", "labels.json"), "w"))
rng = random.Random(0)
def boot(vals, B=2000):
    bs = [np.mean([rng.choice(vals) for _ in vals]) for _ in range(B)]
    return float(np.mean(vals)), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
res = {}
for var in ("real", "test"):
    for ds in ("actions", "triggers", "all"):
        rs = [r for r in rows if r["var"] == var and r["done"] and (ds == "all" or r["ds"] == ds)]
        allv = [r for r in rows if r["var"] == var and (ds == "all" or r["ds"] == ds)]
        by = defaultdict(list)
        for r in rs: by[r["q"]].append(r)
        mixed = [q for q, x in by.items() if any(i["aware"] for i in x) and any(not i["aware"] for i in x)]
        diffs = []
        for q in mixed:
            a = [i["comp"] for i in by[q] if i["aware"] and "comp" in i]; u = [i["comp"] for i in by[q] if not i["aware"] and "comp" in i]
            if a and u: diffs.append(np.mean(a) - np.mean(u))
        ca = [i["comp"] for i in rs if i["aware"] and "comp" in i]; cu = [i["comp"] for i in rs if not i["aware"] and "comp" in i]
        res[f"{var}/{ds}"] = dict(n_runs=len(allv), n_finished=len(rs), aware_share=float(np.mean([i["aware"] for i in rs])),
            n_items=len(by), n_mixed=len(mixed), comp_aware=float(np.mean(ca)) if ca else None, comp_unaware=float(np.mean(cu)) if cu else None,
            n_aware=len(ca), within_item_diff=(boot(diffs) if diffs else None), n_items_with_diff=len(diffs))
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items(): print(k, json.dumps(v))
