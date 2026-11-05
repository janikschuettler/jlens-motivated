"""TA14 exploratory add-on (numpy only; any env): window-length confound. NOT pre-registered.

  python scripts/explore/ta14_extra.py --readouts data/readouts/hawthorne_q14 --out data/ta14/ta14_extra.json

Score as in TA14-1 (eval_primary mean log-prob minus control-noun mean, layer-averaged), per window position.
(a) length-matched: for every (aware, unaware) pair of one prompt, both runs are cut to m = min(n_aware, n_unaware)
    tokens, either the m positions closest to the cut ("last") or the first m ("first"); AUROC over pairs.
(b) residualised: score minus beta * log(window length), beta = pooled slope of score on log length within
    prompt x group cells (so the label's own effect is not absorbed); re-estimated in every bootstrap resample.
Both: 2,000 bootstrap resamples of prompts; J-lens, logit lens, J minus logit.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

B = 2000
ap = argparse.ArgumentParser(); ap.add_argument("--readouts", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); d = Path(a.readouts)
tk = json.loads((d / "tokens.json").read_text()); cols = {c: i for i, c in enumerate(tk["columns"])}
prim = [cols[" fictional"], cols[" simulation"]]; ctrl = cols["control_mean"]
by = defaultdict(lambda: {"aware": [], "unaware": []})
for l in open(d / "index.jsonl"):
    m = json.loads(l); z = np.load(d / m["file"]); nw = int(z["n_window"])
    lp = z["lp"][:, :, :nw, :].astype(np.float32)
    s = (lp[..., prim].mean(-1) - lp[..., ctrl]).mean(1)  # [lens, pos]
    by[m["question_id"]][m["group"]].append(s)
by = {q: g for q, g in by.items() if g["aware"] and g["unaware"]}
Q = sorted(by); print(len(Q), "prompts")

def pair_stats(fn):
    """fn(a_series, u_series) -> [2] pair scores diff; returns per-prompt (num[2], den)."""
    num = np.zeros((len(Q), 2)); den = np.zeros(len(Q))
    for k, q in enumerate(Q):
        for sa in by[q]["aware"]:
            for su in by[q]["unaware"]:
                x, y = fn(sa, su)
                num[k] += (x > y) + 0.5 * (x == y); den[k] += 1
    return num, den

def summarize(num, den, rng):
    est = num.sum(0) / den.sum()
    bs = np.zeros((B, 2))
    for b in range(B):
        i = rng.integers(0, len(Q), len(Q)); bs[b] = num[i].sum(0) / den[i].sum()
    out = {}
    for j, n in enumerate(("jlens", "logit")):
        out[n] = {"auroc": float(est[j]), "ci95": np.quantile(bs[:, j], [.025, .975]).tolist(), "ci90": np.quantile(bs[:, j], [.05, .95]).tolist()}
    df = bs[:, 0] - bs[:, 1]
    out["jlens_minus_logit"] = {"est": float(est[0] - est[1]), "ci95": np.quantile(df, [.025, .975]).tolist(), "ci90": np.quantile(df, [.05, .95]).tolist()}
    return out

rng = np.random.default_rng(0)
res = {"n_prompts": len(Q), "n_pairs": 0}
def full(sa, su): return sa.mean(1), su.mean(1)
def last(sa, su):
    m = min(sa.shape[1], su.shape[1]); return sa[:, -m:].mean(1), su[:, -m:].mean(1)
def first(sa, su):
    m = min(sa.shape[1], su.shape[1]); return sa[:, :m].mean(1), su[:, :m].mean(1)
for name, fn in (("full_window_check", full), ("length_matched_last", last), ("length_matched_first", first)):
    num, den = pair_stats(fn); res[name] = summarize(num, den, rng); res["n_pairs"] = int(den.sum())
# window-length AUROC as a baseline (full windows)
ln = np.zeros(len(Q)); dn = np.zeros(len(Q))
for k, q in enumerate(Q):
    for sa in by[q]["aware"]:
        for su in by[q]["unaware"]:
            ln[k] += (sa.shape[1] > su.shape[1]) + 0.5 * (sa.shape[1] == su.shape[1]); dn[k] += 1
bs = [ln[i].sum() / dn[i].sum() for i in (rng.integers(0, len(Q), len(Q)) for _ in range(B))]
res["length_auroc"] = {"auroc": float(ln.sum() / dn.sum()), "ci95": np.quantile(bs, [.025, .975]).tolist()}
# residualised
data = []  # per prompt: score[2, n], loglen[n], is_aware[n], cell id
for q in Q:
    sc = [x.mean(1) for x in by[q]["aware"] + by[q]["unaware"]]
    L = [np.log(x.shape[1]) for x in by[q]["aware"] + by[q]["unaware"]]
    g = [1] * len(by[q]["aware"]) + [0] * len(by[q]["unaware"])
    data.append((np.array(sc).T, np.array(L), np.array(g)))
cell = []  # per prompt: demeaned within group
for sc, L, g in data:
    sxy = np.zeros(2); sxx = 0.0
    for v in (0, 1):
        k = g == v
        if k.sum():
            x = L[k] - L[k].mean(); sxx += (x * x).sum(); sxy += ((sc[:, k] - sc[:, k].mean(1, keepdims=True)) * x).sum(1)
    cell.append((sxy, sxx))
cell_xy = np.array([c[0] for c in cell]); cell_xx = np.array([c[1] for c in cell])
def pairs_resid(k, beta):
    sc, L, g = data[k]; r = sc - beta[:, None] * L[None]
    a, u = r[:, g == 1], r[:, g == 0]
    diff = a[:, :, None] - u[:, None, :]
    return ((diff > 0) + 0.5 * (diff == 0)).sum((1, 2)), diff.shape[1] * diff.shape[2]
beta0 = cell_xy.sum(0) / cell_xx.sum()
pp = [pairs_resid(k, beta0) for k in range(len(Q))]
est = np.sum([p[0] for p in pp], 0) / sum(p[1] for p in pp)
bs = np.zeros((B, 2))
for b in range(B):
    i = rng.integers(0, len(Q), len(Q)); beta = cell_xy[i].sum(0) / cell_xx[i].sum()
    n = np.zeros(2); dd = 0
    for k in i:
        x, y = pairs_resid(k, beta); n += x; dd += y
    bs[b] = n / dd
o = {}
for j, n in enumerate(("jlens", "logit")):
    o[n] = {"auroc": float(est[j]), "ci95": np.quantile(bs[:, j], [.025, .975]).tolist(), "ci90": np.quantile(bs[:, j], [.05, .95]).tolist(), "beta_per_log_token": float(beta0[j])}
df = bs[:, 0] - bs[:, 1]
o["jlens_minus_logit"] = {"est": float(est[0] - est[1]), "ci95": np.quantile(df, [.025, .975]).tolist(), "ci90": np.quantile(df, [.05, .95]).tolist()}
res["residualised_on_log_length"] = o
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
