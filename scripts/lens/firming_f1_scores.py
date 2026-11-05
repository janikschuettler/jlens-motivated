"""F1 (docs/preregistration.md, F1-F3), cases 2-4, 6, 7: TOST equivalence tests on stored per-run lens scores.
CPU, numpy+scipy. Same rules as firming_f1_cases.py: 5,000 item bootstraps, seed 0, same resamples for both lenses,
bound +-0.10 around 0.5, 90% interval for equivalence, 95% for 'differs'.
Inputs: t1b score jsonl (oct, mayor), claim_lead_repeat.json, c3_lead_repeat.json, t2.jsonl (firming_recompute.py t2),
hack.npz meta (competitors_hack.py extract; concept score per run and position).

    python scripts/lens/firming_f1_scores.py --root . --comp DIR --t2 t2.jsonl --out OUT.json
"""
import argparse, json
from collections import defaultdict
import numpy as np
from firming_f1_cases import auroc, N_BOOT, SEED, BOUND


def tost(rows, boot_sets, group_key="item"):
    """rows: list of dict(item,pos,jlens,logit). boot_sets: list of arrays of row indices."""
    it = np.array([r[group_key] for r in rows]); y = np.array([r["pos"] for r in rows], int)
    sc = {k: np.array([r[k] for r in rows], float) for k in ("jlens", "logit")}
    bs = {k: [] for k in sc}
    for s in boot_sets:
        yy = y[s]
        for k in sc:
            bs[k].append(auroc(sc[k][s][yy == 1], sc[k][s][yy == 0]))
    return summarize({k: float(auroc(sc[k][y == 1], sc[k][y == 0])) for k in sc}, {k: np.array(v) for k, v in bs.items()},
                     int(y.sum()), int((1 - y).sum()), len(set(it)))


def summarize(pt, bs, npos, nneg, nitems):
    bs["diff"] = bs["jlens"] - bs["logit"]; pt["diff"] = pt["jlens"] - pt["logit"]
    out = {"n_pos": npos, "n_neg": nneg, "n_clusters": nitems}
    for k in ("jlens", "logit", "diff"):
        b = bs[k][np.isfinite(bs[k])]; q = np.percentile(b, [2.5, 5, 95, 97.5])
        c = {"point": pt[k], "ci95": [float(q[0]), float(q[3])], "ci90": [float(q[1]), float(q[2])]}
        if k != "diff":
            c["equiv_chance"] = bool(q[1] >= .5 - BOUND and q[2] <= .5 + BOUND)
            c["differs_from_chance"] = bool(q[0] > .5 or q[3] < .5)
            c["smallest_bound"] = float(max(abs(q[1] - .5), abs(q[2] - .5)))
        else:
            c["advantage_ge_010_ruled_out"] = bool(q[2] < BOUND); c["equivalent_lenses"] = bool(q[1] > -BOUND and q[2] < BOUND)
        out[k] = c
    return out


def boots_for(items, rng):
    rows = defaultdict(list)
    for i, q in enumerate(items):
        rows[q].append(i)
    keys = sorted(rows)
    return [np.concatenate([rows[keys[k]] for k in rng.choice(len(keys), len(keys))]) for _ in range(N_BOOT)]


def run(rows):
    return tost(rows, boots_for([r["item"] for r in rows], np.random.default_rng(SEED)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--comp"); ap.add_argument("--t2"); ap.add_argument("--out")
    a = ap.parse_args(); R = a.root; res = {}
    # T1b: silent (pos) vs open; S = pre_answer margin vs uninvolved letters, layers 14-32 mean (S[0] J, S[1] logit)
    t1 = {}
    for nm in ("oct", "mayor"):
        L = [json.loads(x) for x in open(f"{R}/data/overnight/firming/oct/t1b_{nm}_scores.jsonl")][1:]
        t1[nm] = [{"item": d["item"], "pos": int(d["group"] == "silent"), "jlens": d["S"][0], "logit": d["S"][1]} for d in L]
    res["t1b_pooled_preregistered"] = run(t1["oct"] + t1["mayor"])
    res["t1b_octavia_half"] = run(t1["oct"]); res["t1b_mayor_half"] = run(t1["mayor"])
    # R-CL (deferred = pos vs persuaded) and R-C3 (names correct = pos) from derived per-run scores
    for nm, f in (("rcl_claimed_answer_key", "claim_lead_repeat"), ("rc3_planted_claim_repeat", "c3_lead_repeat")):
        runs = json.load(open(f"{R}/docs/experiments/replicate-leads/data/{f}.json"))["runs"]
        res[nm] = run([{"item": d["item"], "pos": int(d["pos"]), "jlens": d["S"]["jlens"], "logit": d["S"]["logit"]} for d in runs])
    # T2 (recomputed from raw readouts by firming_recompute.py t2)
    T = [json.loads(x) for x in open(a.t2)][1:]
    res["t2_followers_still_know_vs_persuaded"] = run([{"item": d["item"], "pos": int(d["pos"]), "jlens": d["jlens"], "logit": d["logit"]} for d in T])
    # hack: problem-level means over problems with both groups, 4 positions; plus within-problem pairs at mid
    z = np.load(f"{a.comp}/hack.npz"); runs = json.loads(str(z["meta_json"]))["runs"]
    for posn in ("mid", "last_prompt", "think_end", "pre_code"):
        by = defaultdict(lambda: {0: [], 1: []})
        for d in runs:
            by[d["item"]][d["group"]].append(d[f"lens_{posn}"])
        prob = []
        for q, g in sorted(by.items()):
            if g[0] and g[1]:
                for s in (0, 1):
                    m = np.mean(g[s], 0); prob.append({"item": q, "pos": s, "jlens": float(m[0]), "logit": float(m[1])})
        res[f"hack_problem_means_{posn}"] = run(prob)
    # within-problem pair AUROC at mid (W's metric): pairs inside the same problem only
    by = defaultdict(lambda: {0: [], 1: []})
    for d in runs:
        by[d["item"]][d["group"]].append(d["lens_mid"])
    qs = [q for q in sorted(by) if by[q][0] and by[q][1]]

    def wp(sel, k):
        u = n = 0
        for q in sel:
            p = np.array([x[k] for x in by[q][1]]); m = np.array([x[k] for x in by[q][0]])
            dd = p[:, None] - m[None, :]; u += (dd > 0).sum() + .5 * (dd == 0).sum(); n += dd.size
        return u / n
    rng = np.random.default_rng(SEED); bs = {"jlens": [], "logit": []}
    for _ in range(N_BOOT):
        sel = [qs[j] for j in rng.choice(len(qs), len(qs))]
        for k, ki in (("jlens", 0), ("logit", 1)):
            bs[k].append(wp(sel, ki))
    res["hack_within_problem_mid"] = summarize({"jlens": wp(qs, 0), "logit": wp(qs, 1)}, {k: np.array(v) for k, v in bs.items()},
                                               sum(len(by[q][1]) for q in qs), sum(len(by[q][0]) for q in qs), len(qs))
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, v["n_pos"], v["n_neg"], v["n_clusters"], {l: (round(v[l]["point"], 3), [round(x, 3) for x in v[l]["ci90"]]) for l in ("jlens", "logit", "diff")})


if __name__ == "__main__":
    main()
