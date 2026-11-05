"""EA27-L: lens-quality check (pre-registration EA27-L). Reads positions.npz from ea27_readout.py (prompts step).
Primary: top-1 agreement of each lens with the model's own top-1, pooled over every position of the deployment
prompts (<= 2048 tokens), averaged over blocks 33-49; 95% bootstrap (2,000 resamples, seed 0) over prompts.
Secondary: mean of per-prompt agreement rates. Output: docs/experiments/ea27-replication/data/ea27_L.json"""
import json, sys
import numpy as np
d = sys.argv[1]
OUT = "docs/experiments/ea27-replication/data/ea27_L.json"
pz = np.load(f"{d}/positions.npz")
meta = json.load(open(f"{d}/meta.json"))
ws = meta["workspace_layers"]; lenses = meta["lenses"]
sel = [i for i, l in enumerate(ws) if 33 <= l <= 49]
keys = sorted({k.split("|")[0] for k in pz.files if k.startswith(("oasst/", "wildchat/"))})
A, N, excl = [], [], 0   # A[p]: agree counts [lens, layer]; N[p]: positions
for k in keys:
    ids = pz[k + "|ids"]
    if len(ids) > 2048:
        excl += 1; continue
    ag = (pz[k + "|top1"] == pz[k + "|own"][None, None]).sum(-1)
    A.append(ag); N.append(len(ids))
A = np.stack(A).astype(float); N = np.array(N, float)  # [P, lens, layer]
P = len(N)
def est(idx):
    a = A[idx][:, :, sel].mean(-1).sum(0) / N[idx].sum()   # pooled, layer-averaged
    return a
def est2(idx):
    return (A[idx][:, :, sel].mean(-1) / N[idx][:, None]).mean(0)
rng = np.random.default_rng(0)
B = [rng.integers(0, P, P) for _ in range(2000)]
def ci(f, fn):
    v = np.array([fn(b) for b in B]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
al = np.arange(P); L = {n: i for i, n in enumerate(lenses)}
res = {"n_prompts": P, "n_excluded_over_2048": excl, "n_positions": int(N.sum()), "blocks": [ws[i] for i in sel]}
for name, fn in (("pooled", est), ("per_prompt_mean", est2)):
    r = {}
    for n, i in L.items():
        r[n] = {"agree": float(fn(al)[i]), "ci": ci(None, lambda b, i=i: fn(b)[i])}
    for a, b_ in (("jlens", "logit"), ("jpp", "jlens"), ("jpp", "logit")):
        r[f"{a}_minus_{b_}"] = {"diff": float(fn(al)[L[a]] - fn(al)[L[b_]]),
                               "ci": ci(None, lambda b, a=a, b_=b_: fn(b)[L[a]] - fn(b)[L[b_]])}
    res[name] = r
res["per_layer_pooled"] = {ws[j]: {n: float(A[:, i, j].sum() / N.sum()) for n, i in L.items()} for j in range(len(ws))}
d1 = res["pooled"]["jlens_minus_logit"]
res["verdict_primary"] = "works" if d1["ci"][0] > 0 else "fails"
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n_prompts", "n_excluded_over_2048", "n_positions", "verdict_primary")}))
print(json.dumps(res["pooled"], indent=0)[:1500])
