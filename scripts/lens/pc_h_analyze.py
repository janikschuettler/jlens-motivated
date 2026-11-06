"""probe-control PC-H analysis (CPU; lens env): test (a) on the test-awareness runs of Qwen3-14B. Pre-registered as PC-H.

  python scripts/lens/pc_h_analyze.py --probe data/probe_control/readout_h --lens data/readouts/hawthorne_q14 \
      --runs data/hawthorne/q14/runs.jsonl --labels data/ta14j --out <json>

Pairs = all (aware, unaware) runs of one prompt (TA14-1 style); within-prompt AUROC; 2,000 prompt-level bootstrap
resamples (seed 0). Scores per run: Nguyen (layer 2) and Hua (layer 14) window-mean projection, J-lens and logit-lens
eval-word score of TA14-1 (read from the existing readout), window length, word 1-2-gram TF-IDF text classifier
(prompts held out, 5 folds), 100 random directions at the probe layers (band, window-mean AUROC).
Exploratory (declared in PC-H): length-matched (first m / last m), residualised on log window length (fixed slope).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

B = 2000
ap = argparse.ArgumentParser()
for k in ("probe", "lens", "runs", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--labels", default=None)
a = ap.parse_args()
pd_, ld = Path(a.probe), Path(a.lens)
tk = json.loads((ld / "tokens.json").read_text()); cols = {c: i for i, c in enumerate(tk["columns"])}
prim = [cols[" fictional"], cols[" simulation"]]; ctrl = cols["control_mean"]
summ = json.loads((pd_ / "summary.json").read_text())
lens_idx = {json.loads(l)["run_id"]: json.loads(l) for l in open(ld / "index.jsonl")}
NAMES = ["nguyen", "hua", "jlens", "logit"]
R = {}  # run_id -> dict(group, q, series{name: [T]}, rnd [T,200])
for l in open(pd_ / "index.jsonl"):
    m = json.loads(l); z = np.load(pd_ / m["file"])
    T = m["hi"] - m["lo"]
    li = lens_idx[m["run_id"]]
    assert li["n_window"] == T and li["lo"] == m["lo"] and li["hi"] == m["hi"], m["run_id"]
    zl = np.load(ld / li["file"]); lp = zl["lp"][:, :, :T, :].astype(np.float32)
    s = (lp[..., prim].mean(-1) - lp[..., ctrl]).mean(1)  # [2, T]
    proj = z["proj"].astype(np.float32)  # [T, 40, 2]
    Ln, Lh = summ["layers"]["nguyen"], summ["layers"]["hua"]
    R[m["run_id"]] = dict(group=m["group"], q=m["question_id"], T=T, dataset=m["dataset"],
                          s={"nguyen": proj[:, Ln, 0], "hua": proj[:, Lh, 1], "jlens": s[0], "logit": s[1]},
                          rnd=np.concatenate([z["rnd_n"].astype(np.float32), z["rnd_h"].astype(np.float32)], 1))
print(len(R), "runs", flush=True)

# decoded window text
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", revision="40c069824f4251a91eefaf281ebe4c544efd3e18")
comp = {}
for l in open(a.runs):
    r = json.loads(l)
    if r["run_id"] in R:
        comp[r["run_id"]] = r["completion_token_ids"]
for rid, v in R.items():
    v["ids"] = comp[rid][: v["T"]]
    v["text"] = tok.decode(v["ids"])


def auroc_ci(num, den, rng_idx):
    """num [Q, K], den [Q]; rng_idx [B, Q] resample counts."""
    est = num.sum(0) / den.sum()
    bn = rng_idx @ num; bd = rng_idx @ den
    bs = bn / bd[:, None]
    return est, bs


def run_mode(runs, ids_by_label, textscore=None, textfn=None):
    """runs: dict run_id -> record (with group). Returns dict of results for one label set."""
    byq = defaultdict(lambda: {"aware": [], "unaware": []})
    for rid, v in runs.items():
        byq[v["q"]][v["group"]].append(rid)
    Q = sorted(q for q, g in byq.items() if g["aware"] and g["unaware"])
    rng = np.random.default_rng(0)
    cnt = np.stack([np.bincount(rng.integers(0, len(Q), len(Q)), minlength=len(Q)) for _ in range(B)]).astype(float)
    out = {"n_prompts": len(Q), "n_aware": sum(len(byq[q]["aware"]) for q in Q), "n_unaware": sum(len(byq[q]["unaware"]) for q in Q)}

    def collect(scorefn, keys):
        num = np.zeros((len(Q), len(keys))); den = np.zeros(len(Q))
        for k, q in enumerate(Q):
            for ra in byq[q]["aware"]:
                for ru in byq[q]["unaware"]:
                    x, y = scorefn(ra, ru)  # arrays [K]
                    num[k] += (x > y) + 0.5 * (x == y); den[k] += 1
        return num, den

    def summarize(num, den, keys, diffs=()):
        est, bs = auroc_ci(num, den, cnt)
        res = {}
        for j, k in enumerate(keys):
            res[k] = {"auroc": float(est[j]), "ci95": np.quantile(bs[:, j], [.025, .975]).tolist(),
                      "ci90": np.quantile(bs[:, j], [.05, .95]).tolist()}
        ix = {k: j for j, k in enumerate(keys)}
        for p, q2 in diffs:
            if p in ix and q2 in ix:
                d = bs[:, ix[p]] - bs[:, ix[q2]]
                res[f"{p}_minus_{q2}"] = {"est": float(est[ix[p]] - est[ix[q2]]), "ci95": np.quantile(d, [.025, .975]).tolist(),
                                          "ci90": np.quantile(d, [.05, .95]).tolist()}
        return res

    # --- text classifier (prompts held out)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    rids = [rid for q in Q for g in ("aware", "unaware") for rid in byq[q][g]]
    y = np.array([1 if runs[r]["group"] == "aware" else 0 for r in rids]); qid = np.array([runs[r]["q"] for r in rids])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, qid))
    oof = {}; models = {}
    for tr, te in folds:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        X = v.fit_transform([runs[rids[i]]["text"] for i in tr])
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y[tr])
        dec = m.decision_function(v.transform([runs[rids[i]]["text"] for i in te]))
        for i, d in zip(te, dec):
            oof[rids[i]] = float(d)
        for i in te:
            models[qid[i]] = (v, m)
    Tn = lambda r: runs[r]["T"]
    DIFFS = [("jlens", "nguyen"), ("jlens", "hua"), ("jlens", "logit"), ("jlens", "text"), ("nguyen", "text"), ("hua", "text"), ("logit", "text")]
    KEYS = NAMES + ["length", "text"]

    def full(ra, ru):
        return (np.array([runs[ra]["s"][n].mean() for n in NAMES] + [Tn(ra), oof[ra]]),
                np.array([runs[ru]["s"][n].mean() for n in NAMES] + [Tn(ru), oof[ru]]))
    out["primary"] = summarize(*collect(full, KEYS), KEYS, DIFFS)
    # --- random directions band (window-mean AUROC per direction, pooled pairs)
    rn = np.zeros(200); dn = 0
    for q in Q:
        A = np.stack([runs[r]["rnd"].mean(0) for r in byq[q]["aware"]]); U = np.stack([runs[r]["rnd"].mean(0) for r in byq[q]["unaware"]])
        d = A[:, None, :] - U[None, :, :]
        rn += ((d > 0) + 0.5 * (d == 0)).sum((0, 1)); dn += d.shape[0] * d.shape[1]
    rn /= dn
    out["random_band"] = {"nguyen_layer": {"q025": float(np.quantile(rn[:100], .025)), "q975": float(np.quantile(rn[:100], .975))},
                          "hua_layer": {"q025": float(np.quantile(rn[100:], .025)), "q975": float(np.quantile(rn[100:], .975))}}
    # --- secondary windows
    for nm, sl in (("last32", lambda x: x[-32:]), ("first64", lambda x: x[:64])):
        def f(ra, ru, sl=sl):
            return (np.array([sl(runs[ra]["s"][n]).mean() for n in NAMES]), np.array([sl(runs[ru]["s"][n]).mean() for n in NAMES]))
        out[nm] = summarize(*collect(f, NAMES), NAMES, [("jlens", "logit")])
    # --- exploratory: length-matched
    for nm, sl in (("matched_first", lambda x, m: x[:m]), ("matched_last", lambda x, m: x[-m:])):
        def f(ra, ru, sl=sl):
            m = min(Tn(ra), Tn(ru))
            return (np.array([sl(runs[ra]["s"][n], m).mean() for n in NAMES]), np.array([sl(runs[ru]["s"][n], m).mean() for n in NAMES]))
        out[nm] = summarize(*collect(f, NAMES), NAMES, [("jlens", "logit"), ("jlens", "nguyen"), ("jlens", "hua")])
    # matched text: first m tokens, classifier of the fold holding the prompt out
    def ftext(ra, ru):
        m = min(Tn(ra), Tn(ru)); v, mod = models[runs[ra]["q"]]
        da, du = mod.decision_function(v.transform([tok.decode(runs[ra]["ids"][:m]), tok.decode(runs[ru]["ids"][:m])]))
        return np.array([da]), np.array([du])
    out["matched_first"]["text"] = summarize(*collect(ftext, ["text"]), ["text"])["text"]
    # residualised on log length (fixed within-cell slope)
    beta = {}
    for n in NAMES:
        sxy = sxx = 0.0
        for q in Q:
            for g in ("aware", "unaware"):
                rr = byq[q][g]
                if len(rr) < 2:
                    continue
                L = np.log([Tn(r) for r in rr]); S = np.array([runs[r]["s"][n].mean() for r in rr])
                L -= L.mean(); S -= S.mean(); sxy += (S * L).sum(); sxx += (L * L).sum()
        beta[n] = sxy / sxx
    def fres(ra, ru):
        return (np.array([runs[ra]["s"][n].mean() - beta[n] * np.log(Tn(ra)) for n in NAMES]),
                np.array([runs[ru]["s"][n].mean() - beta[n] * np.log(Tn(ru)) for n in NAMES]))
    out["residualised"] = summarize(*collect(fres, NAMES), NAMES, [("jlens", "logit"), ("jlens", "nguyen"), ("jlens", "hua")])
    out["residualised"]["slopes"] = {k: float(v) for k, v in beta.items()}
    out["mean_window_len"] = {g: float(np.mean([runs[r]["T"] for r in rids if runs[r]["group"] == g])) for g in ("aware", "unaware")}
    return out


res = {"probe_layers": summ["layers"], "n_runs_read": len(R)}
res["keyword"] = run_mode(R, None)
for name, path in (("strict_yes_vs_no", "labels_strict_nano.json"), ("yesmaybe_vs_no", "labels_yesmaybe_nano.json")):
    if a.labels is None:
        continue
    lab = json.loads((Path(a.labels) / path).read_text())
    sub = {rid: dict(v, group=lab[rid]) for rid, v in R.items() if rid in lab}
    res[name] = run_mode(sub, None)
Path(a.out).write_text(json.dumps(res, indent=1))
for k in ("keyword", "strict_yes_vs_no", "yesmaybe_vs_no"):
    if k in res:
        print(k, res[k]["n_prompts"], {n: round(v["auroc"], 3) for n, v in res[k]["primary"].items() if "auroc" in v}, res[k]["random_band"])
