"""F2 (docs/preregistration.md, F1-F3): does the planted-claim positive (R-C3, AUROC 0.71) survive a
question-difficulty control? CPU, numpy only. Inputs: docs/experiments/replicate-leads/data/c3_lead_repeat.json
(per-run S, item, pos) and a small extract of octavia's data/explore (known_items, control forks, labels).

    python scripts/lens/firming_f2.py --inp f2_in.json --out out.json
"""
import argparse, json
from collections import defaultdict
import numpy as np

NB, NP, SEED = 2000, 10000, 0
ap = argparse.ArgumentParser(); ap.add_argument("--inp"); ap.add_argument("--runs", default="docs/experiments/replicate-leads/data/c3_lead_repeat.json")
ap.add_argument("--out"); ap.add_argument("--drop-uninformative-control", action="store_true"); a = ap.parse_args()
rng = np.random.default_rng(SEED)
I = json.load(open(a.inp)); R = json.load(open(a.runs))["runs"]
known = {k["id"]: k for k in I["known"]}
ctrl = {(c["batch"], c["question_id"]): c for c in I["control"]}
rec = defaultdict(lambda: [0, 0])  # question -> [n correct, n runs]
for l in I["labels"]:
    rec[l["question_id"]][1] += 1; rec[l["question_id"]][0] += l["outcome"] == "correct"

rows = []
for r in R:
    q = r["item"]; k = known[q]; b = "" if r["source"] == "new" else "2"
    c = ctrl[(b, q)]
    rows.append(dict(item=q, y=int(r["pos"]), sj=r["S"]["jlens"], sl=r["S"]["logit"],
                     miss20=(k["n"]["A"] - k["n_correct"]["A"]) + (k["n"]["B"] - k["n_correct"]["B"]),
                     miss_off=k["n"]["B"] - k["n_correct"]["B"], cwrong=int(c["argmax"] != c["correct_letter"]),
                     pcorr=(c["p_norm"][c["correct_letter"]] if c["p_norm"] else 0.25), recov=rec[q][0] / rec[q][1], plet=c["p_letters"]))
# Control with argmax None (all four letter probabilities underflow to 0): literal rule counts it as "wrong"
# (None != correct); p(correct) is set to the uninformative 0.25. --drop-uninformative-control removes runs whose
# control put < 0.5 of its mass on the letters (sensitivity analysis, not the pre-registered one).
if a.drop_uninformative_control:
    rows = [x for x in rows if x["plet"] >= 0.5]
col = lambda n: np.array([x[n] for x in rows], float)
items = np.array([x["item"] for x in rows]); y = col("y").astype(int)
sj, sl, miss20, miss_off, cw, pc, rc = (col(n) for n in ("sj", "sl", "miss20", "miss_off", "cwrong", "pcorr", "recov"))
keys = sorted(set(items)); idx = {k: np.where(items == k)[0] for k in keys}
boots = [np.concatenate([idx[keys[j]] for j in rng.choice(len(keys), len(keys))]) for _ in range(NB)]


def U(p, n):  # concordant + 0.5 ties, number of pairs
    if not len(p) or not len(n):
        return 0.0, 0
    d = p[:, None] - n[None, :]
    return float((d > 0).sum() + 0.5 * (d == 0).sum()), len(p) * len(n)


def auroc(s, yy):
    u, n = U(s[yy == 1], s[yy == 0]); return u / n if n else np.nan


def ci(v):
    v = np.array([x for x in v if np.isfinite(x)]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def stratum_ids(m20, cwr):
    g = np.where(m20 == 0, 0, np.where(m20 == 1, 1, np.where(m20 <= 3, 2, 3)))
    return g * 2 + cwr.astype(int)


def matched(s, yy, st):
    u = n = 0
    for g in np.unique(st):
        m = st == g; uu, nn = U(s[m & (yy == 1)], s[m & (yy == 0)]); u += uu; n += nn
    return u / n if n else np.nan


def within(s, yy, it):
    u = n = 0
    for q in np.unique(it):
        m = it == q
        if yy[m].min() != yy[m].max():
            uu, nn = U(s[m & (yy == 1)], s[m & (yy == 0)]); u += uu; n += nn
    return u / n if n else np.nan


def logit_fit(X, yy):
    X = np.column_stack([np.ones(len(yy)), X]); w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30))); W = np.maximum(p * (1 - p), 1e-6)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        w2 = w + np.linalg.solve(H, X.T @ (yy - p))
        if np.abs(w2 - w).max() < 1e-8:
            w = w2; break
        w = w2
    return w


z = lambda v: (v - v.mean()) / v.std()
st = stratum_ids(miss20, cw)
out = {"n_runs": len(rows), "n_items": len(keys), "n_pos": int(y.sum()),
       "control_p_letters_share_below_0.5": float(np.mean(col("plet") < 0.5)),
       "control_wrong_share": float(cw.mean()), "miss20_dist": {int(k): int((miss20 == k).sum()) for k in np.unique(miss20)}}
# (a) covariates alone
cov = {"miss20 (easier -> names correct)": -miss20, "control right": 1 - cw, "thinking-off misses (neg)": -miss_off,
       "control p(correct)": pc, "recovery rate": rc, "J-lens S": sj, "logit-lens S": sl}
out["a_alone"] = {}
for n, s in cov.items():
    v = [auroc(s[b], y[b]) for b in boots]; out["a_alone"][n] = {"auroc": auroc(s, y), "ci95": ci(v)}
# (b) logistic models
def models(b):
    res = {}
    yy = y[b]
    for ln, s in (("jlens", sj), ("logit", sl)):
        zs = z(s)  # z-scored on the full sample (fixed scale across resamples)
        for mn, covs in (("primary", [miss20, cw]), ("secondary", [miss20, cw, miss_off, pc, rc])):
            X = np.column_stack([zs] + covs)[b]
            res[f"{ln}|{mn}"] = float(logit_fit(X, yy)[1])
        res[f"{ln}|alone"] = float(logit_fit(zs[b][:, None], yy)[1])
    return res
pt = models(np.arange(len(rows))); bs = [models(b) for b in boots]
out["b_logistic_S_coef_logodds_per_SD"] = {k: {"coef": pt[k], "ci95": ci([r[k] for r in bs])} for k in pt}
# (c) difficulty-matched
easy = (miss20 == 0) & (cw == 0)
out["c_matched"] = {}
for ln, s in (("jlens", sj), ("logit", sl)):
    v = [matched(s[b], y[b], st[b]) for b in boots]
    ve = [auroc(s[b][easy[b]], y[b][easy[b]]) for b in boots]
    out["c_matched"][ln] = {"matched_auroc": matched(s, y, st), "ci95": ci(v),
                            "easy_auroc": auroc(s[easy], y[easy]), "easy_ci95": ci(ve), "easy_n": int(easy.sum()),
                            "easy_n_pos": int(y[easy].sum()), "easy_items": len(set(items[easy]))}
out["c_strata_counts"] = {int(g): [int(((st == g) & (y == 1)).sum()), int(((st == g) & (y == 0)).sum())] for g in np.unique(st)}
# (d) within question
both = [q for q in keys if 0 < y[idx[q]].sum() < len(idx[q])]
out["d_within"] = {"n_questions_both": len(both), "n_runs": int(sum(len(idx[q]) for q in both))}
m = np.isin(items, both)
cc = {}
for ln, s in (("jlens", sj), ("logit", sl)):
    obs = within(s[m], y[m], items[m])
    v = []
    bk = {q: idx[q] for q in both}
    for _ in range(NB):
        pick = rng.choice(len(both), len(both)); sel = np.concatenate([bk[both[j]] for j in pick])
        lab = np.concatenate([[i] * len(bk[both[j]]) for i, j in enumerate(pick)])
        v.append(within(s[sel], y[sel], lab))
    # permutation within question
    sm, ym, im = s[m], y[m], items[m]; cnt = 0
    grp = [np.where(im == q)[0] for q in both]
    for _ in range(NP):
        yp = ym.copy()
        for g in grp:
            yp[g] = rng.permutation(ym[g])
        cnt += within(sm, yp, im) >= obs - 1e-12
    cc[ln] = {"auroc": obs, "ci95": ci(v), "perm_p_onesided": (cnt + 1) / (NP + 1)}
    per_q = [auroc(s[idx[q]], y[idx[q]]) for q in both]
    cc[ln]["questions_in_expected_direction"] = int(sum(x > 0.5 for x in per_q)); cc[ln]["questions_tied"] = int(sum(x == 0.5 for x in per_q))
# paired J - logit within question
bk = {q: idx[q] for q in both}; dv = []
for _ in range(NB):
    pick = rng.choice(len(both), len(both)); sel = np.concatenate([bk[both[j]] for j in pick])
    lab = np.concatenate([[i] * len(bk[both[j]]) for i, j in enumerate(pick)])
    dv.append(within(sj[sel], y[sel], lab) - within(sl[sel], y[sel], lab))
cc["jlens_minus_logit"] = {"diff": cc["jlens"]["auroc"] - cc["logit"]["auroc"], "ci95": ci(dv)}
out["d_within"].update(cc)
# whole-sample check
out["check_overall_auroc"] = {"jlens": auroc(sj, y), "logit": auroc(sl, y)}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
