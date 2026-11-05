"""Q-ah: within-item silent vs open AUROC of S (J-lens, logit lens), J minus logit, length baselines (CPU).
uv run --project envs/judge python scripts/lens/qah_analyze.py --lens DIR --runs runs.jsonl --flags flags.json --out out.json [--arm answer_hint]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from qsq_analyze import load

B = 2000
def pairs(sil, opn, ratio=None):
    """sil, opn: lists of (value, length). Returns (wins, n) over pairs."""
    w = n = 0.0
    for v, l in sil:
        for u, m in opn:
            if ratio and not (max(l, m) / min(l, m) <= 1 + ratio):
                continue
            n += 1; w += 1.0 if v > u else 0.5 if v == u else 0.0
    return w, n

def within(per, ratio=None):
    W = N = 0.0
    for it in per:
        w, n = pairs(per[it]["s"], per[it]["o"], ratio); W += w; N += n
    return W / N if N else float("nan")

def boot(per, stat):
    rng = np.random.default_rng(0); keys = sorted(per); est = stat(per); bs = []
    for _ in range(B):
        ks = rng.choice(keys, size=len(keys)); p = {i: per[k] for i, k in enumerate(ks)}
        v = stat(p)
        if np.isfinite(v): bs.append(v)
    return {"est": round(float(est), 4), "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4)}

def pooled(per, k=0):
    s = [v for i in per.values() for v, _ in i["s"]]; o = [v for i in per.values() for v, _ in i["o"]]
    return sum((a > b) + 0.5 * (a == b) for a in s for b in o) / (len(s) * len(o)) if s and o else float("nan")

def main():
    ap = argparse.ArgumentParser()
    for x in ("--lens", "--runs", "--flags", "--out"): ap.add_argument(x, nargs="+" if x in ("--lens", "--runs") else None, required=True)
    ap.add_argument("--arm", default="answer_hint")
    a = ap.parse_args()
    flags = json.load(open(a.flags)); idx, runs = load(a.lens, a.runs, flags)
    sw = [d for d in runs if d["arm"] == a.arm and d["group"] in ("silent", "open")]
    items = defaultdict(lambda: {"s": [], "o": []})
    for d in sw: items[d["item"]]["s" if d["group"] == "silent" else "o"].append(d)
    both = {i: v for i, v in items.items() if v["s"] and v["o"]}
    res = {"n_silent": sum(d["group"] == "silent" for d in sw), "n_open": sum(d["group"] == "open" for d in sw),
           "items_with_both": len(both), "underpowered_lt15_items": len(both) < 15}
    def mk(score):  # score(d) -> value
        return {i: {"s": [(score(d), d["tokens"]) for d in v["s"]], "o": [(score(d), d["tokens"]) for d in v["o"]]} for i, v in both.items()}
    per = {nm: mk(lambda d, li=li: float(d["S"][li])) for li, nm in enumerate(["J", "logit"])}
    per["length"] = mk(lambda d: float(d["tokens"]))
    for nm, p in per.items():
        res[f"within_item_AUROC_{nm}"] = boot(p, within); res[f"pooled_AUROC_{nm}"] = boot(p, pooled)
    # J minus logit: same bootstrap sample for both
    both_jl = {i: {"J": per["J"][i], "L": per["logit"][i]} for i in both}
    res["within_item_J_minus_logit"] = boot(both_jl, lambda p: within({i: v["J"] for i, v in p.items()}) - within({i: v["L"] for i, v in p.items()}))
    # length-controlled: (1) residualise S on log length within item; (2) length-matched pairs (ratio <= 1.25)
    for li, nm in enumerate(["J", "logit"]):
        xs, ys = [], []
        for i, v in both.items():
            ds = v["s"] + v["o"]; x = np.log([d["tokens"] for d in ds]); y = np.array([d["S"][li] for d in ds])
            xs += list(x - x.mean()); ys += list(y - y.mean())
        slope = float(np.dot(xs, ys) / np.dot(xs, xs)) if np.dot(xs, xs) else 0.0
        def resid(d_, v_=None): pass
        pr = {}
        for i, v in both.items():
            ds = v["s"] + v["o"]; xm = np.mean(np.log([d["tokens"] for d in ds])); ym = np.mean([d["S"][li] for d in ds])
            f = lambda d: (float(d["S"][li]) - ym) - slope * (np.log(d["tokens"]) - xm)
            pr[i] = {"s": [(f(d), d["tokens"]) for d in v["s"]], "o": [(f(d), d["tokens"]) for d in v["o"]]}
        res[f"length_residualised_within_AUROC_{nm}"] = boot(pr, within); res[f"length_residual_slope_{nm}"] = round(slope, 4)
        res[f"length_matched25_within_AUROC_{nm}"] = boot(per[nm], lambda p: within(p, 0.25))
    both_m = {i: {"J": per["J"][i], "L": per["logit"][i]} for i in both}
    res["length_matched25_J_minus_logit"] = boot(both_m, lambda p: within({i: v["J"] for i, v in p.items()}, 0.25) - within({i: v["L"] for i, v in p.items()}, 0.25))
    res["length_matched25_pairs"] = sum(pairs(v["s"], v["o"], 0.25)[1] for v in per["J"].values())
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
main()
