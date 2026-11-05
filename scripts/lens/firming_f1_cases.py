"""F1 (docs/preregistration.md, F1-F3), case 1: TOST equivalence tests on the stored per-run
features of the J-lens-vs-competitors cases (competitors_extract.py npz files). CPU, numpy+scipy.

Score: D = lp(correct) - lp(pushed letter), mean over layers 14-32 (lens[:, pos, 0, lens, :]).
Positions: last_prompt, mid, think_end (where the case has it); pre_answer only for 'silent'.
5,000 item bootstraps, seed 0, same resamples for both lenses (paired). Bound +-0.10 around 0.5.

    python scripts/lens/firming_f1_cases.py --feat DIR --out OUT.json
"""
import argparse, json
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

N_BOOT, SEED, BOUND = 5000, 0, 0.10
POS = ("last_prompt", "mid", "think_end", "pre_answer")
CASES = {"instr": ["instr"], "wording": ["wording"], "silent": ["silent_a", "silent_b"], "c3": ["c3"],
         "grader_q14": ["grader_q14"], "grader_gptoss": ["grader_gptoss"],
         "grader_nt8": ["grader_nt8"], "grader_nt14": ["grader_nt14"]}


def auroc(p, n):
    if not len(p) or not len(n):
        return np.nan
    r = rankdata(np.concatenate([p, n]))
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def load(feat, names):
    runs, lens = [], []
    for n in names:
        z = np.load(f"{feat}/{n}.npz")
        runs += json.loads(str(z["meta_json"]))["runs"]
        lens.append(z["lens"])
    return runs, np.concatenate(lens)


def run_case(name, feat, rng):
    runs, lens = load(feat, CASES[name])
    items = np.array([r["item"] for r in runs]); g = np.array([r["group"] for r in runs])
    keep = (g == 0) | (g == 1)
    items, g, lens = items[keep], g[keep], lens[keep]
    rows = defaultdict(list)
    for i, q in enumerate(items):
        rows[q].append(i)
    keys = sorted(rows)
    samples = [np.concatenate([rows[keys[k]] for k in rng.choice(len(keys), len(keys))]) for _ in range(N_BOOT)]
    positions = [0, 1, 2] + ([3] if name == "silent" else [])
    out = {"n_pos": int((g == 1).sum()), "n_neg": int((g == 0).sum()), "n_items": len(keys), "cells": {}}
    for pi in positions:
        sc = {ln: np.nanmean(lens[:, pi, 0, li, :], -1) for li, ln in enumerate(("jlens", "logit"))}
        if not np.isfinite(sc["jlens"]).any():
            continue
        ok = np.isfinite(sc["jlens"]) & np.isfinite(sc["logit"])
        bs = {ln: [] for ln in sc}
        for s in samples:
            s = s[ok[s]]; gg = g[s]
            for ln in sc:
                bs[ln].append(auroc(sc[ln][s][gg == 1], sc[ln][s][gg == 0]))
        pt = {ln: float(auroc(sc[ln][ok & (g == 1)], sc[ln][ok & (g == 0)])) for ln in sc}
        bs = {ln: np.array(v) for ln, v in bs.items()}
        bs["diff"] = bs["jlens"] - bs["logit"]; pt["diff"] = pt["jlens"] - pt["logit"]
        cell = {}
        for k in ("jlens", "logit", "diff"):
            b = bs[k][np.isfinite(bs[k])]
            q = np.percentile(b, [2.5, 5, 95, 97.5])
            c = {"point": pt[k], "ci95": [q[0], q[3]], "ci90": [q[1], q[2]]}
            if k != "diff":
                c["equiv_chance"] = bool(q[1] >= .5 - BOUND and q[2] <= .5 + BOUND)
                c["differs_from_chance"] = bool(q[0] > .5 or q[3] < .5)
                c["smallest_bound"] = float(max(abs(q[1] - .5), abs(q[2] - .5)))
            else:
                c["advantage_ge_010_ruled_out"] = bool(q[2] < BOUND)
                c["equivalent_lenses"] = bool(q[1] > -BOUND and q[2] < BOUND)
            cell[k] = c
        out["cells"][POS[pi]] = cell
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--feat"); ap.add_argument("--out"); a = ap.parse_args()
    res = {}
    for name in CASES:
        res[name] = run_case(name, a.feat, np.random.default_rng(SEED))
        print(name, {p: (round(c["jlens"]["point"], 3), round(c["jlens"]["ci90"][0], 3), round(c["jlens"]["ci90"][1], 3)) for p, c in res[name]["cells"].items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
