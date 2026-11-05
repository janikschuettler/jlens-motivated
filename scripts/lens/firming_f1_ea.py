"""F1 case 5: Qwen3.6-27B smoke test (grader-bug runs that later say "testing me" vs runs that don't), the paper's
eval score (mean of " fictional", " simulation" minus the rest-of-vocabulary slot, layer mean) before the first
"testing me" sentence (the report's main window = evalaware_big_analyze.py 'own_cutoff'), J-lens and logit lens,
clusters = the 19 questions. Same TOST rules as firming_f1_cases.py. Re-implements the window from the stored
readout (runs.npz) and the stored run texts; uses only first_test_me's regexes from evalaware_big_analyze.py.

    python scripts/lens/firming_f1_ea.py --dir data/overnight/firming/ea27 --runs api_runs.jsonl --out out.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "explore"))
from firming_f1_scores import run
from evalaware_big_analyze import first_test_me

ap = argparse.ArgumentParser(); ap.add_argument("--dir"); ap.add_argument("--runs"); ap.add_argument("--out"); a = ap.parse_args()
d = Path(a.dir); meta = json.load(open(d / "meta.json")); K = len(meta["tracked"])
col = {t["text"]: i for i, t in enumerate(meta["tracked"])}; prim = [col[" fictional"], col[" simulation"]]
texts = {json.loads(l)["run_id"]: json.loads(l) for l in open(a.runs)}
z = np.load(d / "runs.npz"); rows = []
for r in json.load(open(d / "runs_index.json")):
    t = texts[r["run_id"]]; reason = (t["reasoning"] or "").strip(); fr = first_test_me(reason)
    lp = z[r["key"] + "_lp"].astype(np.float32)  # [lens, layers, pos, K+1]
    keep = []
    for i in range(2, len(r["positions"]) - 1):
        c = r["pos_char"][i] - r["reasoning_char0"]
        if fr is None or c < fr - 1:
            keep.append(i)
    if not keep:
        continue
    sc = [float((lp[li][:, keep][..., prim].mean(-1) - lp[li][:, keep][..., K]).mean()) for li in (0, 1)]
    rows.append({"item": r["item_id"], "pos": int(fr is not None), "jlens": sc[0], "logit": sc[1]})
res = run(rows); res["n_runs_used"] = len(rows)
json.dump({"ea27_smoke_before_first_testing_me": res}, open(a.out, "w"), indent=1)
print(res["n_pos"], res["n_neg"], res["n_clusters"], {l: (round(res[l]["point"], 3), [round(x, 3) for x in res[l]["ci90"]]) for l in ("jlens", "logit", "diff")})
