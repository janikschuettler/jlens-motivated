"""probe-control test (a) analysis (CPU). Pre-registered as PC (docs/preregistration.md).

  python scripts/lens/pc_analyze.py --dir data/probe_control --out docs/experiments/probe-control/data/pc_test_a.json

Reads data/probe_control/{pc_select.jsonl, runs.jsonl, build/build.json, readout/, lens/}. Samples = (run, cut index);
pair = (aware run a, unaware sibling u in the same identical-prompt group) read over the first c_a reasoning
tokens (sibling capped at its own </think>). Within-group AUROC = share of pairs where the aware score is higher
(ties count half); bootstrap over questions (2,000, seed 0).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

B = 2000


class Pairs:
    def __init__(self, sel):
        self.samples = []          # (run_id, k)
        sid = {}
        for s in sel:
            for k in range(len(s["cuts"])):
                sid[(s["run_id"], k)] = len(self.samples)
                self.samples.append((s["run_id"], k))
        by = defaultdict(lambda: {"aware": [], "unaware": []})
        for s in sel:
            by[s["question_id"]][s["group"]].append(s)
        sa, su, qq, grp = [], [], [], []
        for key, g in by.items():
            for a in g["aware"]:
                for u in g["unaware"]:
                    k = u["src_cuts"].index(a["cuts"][0])
                    sa.append(sid[(a["run_id"], 0)]); su.append(sid[(u["run_id"], k)])
                    qq.append(key.split("|")[0]); grp.append(key)
        self.sa, self.su = np.array(sa), np.array(su)
        self.qs = sorted(set(qq)); qi = {q: i for i, q in enumerate(self.qs)}
        self.q = np.array([qi[x] for x in qq]); self.grp = np.array(grp)
        self.sid = sid
        self.grp_of_sample = {}
        for s_ in sel:
            for k in range(len(s_["cuts"])):
                self.grp_of_sample[sid[(s_["run_id"], k)]] = s_["question_id"]
        self.n = len(sa)
        self.nq = len(self.qs)
        self.npairs_q = np.bincount(self.q, minlength=self.nq).astype(float)
        rng = np.random.default_rng(0)
        self.boot = np.stack([np.bincount(rng.integers(0, self.nq, self.nq), minlength=self.nq) for _ in range(B)]).astype(float)

    def subset(self, keep):
        import copy
        S = copy.copy(self)
        S.sa, S.su, S.q, S.grp = self.sa[keep], self.su[keep], self.q[keep], self.grp[keep]
        S.n = int(keep.sum())
        S.npairs_q = np.bincount(S.q, minlength=S.nq).astype(float)
        return S

    def wins(self, score):
        d = score[self.sa] - score[self.su]
        return (d > 0) + 0.5 * (d == 0)

    def summ_w(self, w):
        """summary for a per-pair wins vector (0/0.5/1)."""
        pq = np.bincount(self.q, weights=w, minlength=self.nq)
        bs = (self.boot @ pq) / (self.boot @ self.npairs_q)
        return {"auroc": round(float(w.mean()), 4), "ci95": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)],
                "ci90": [round(float(np.quantile(bs, 0.05)), 4), round(float(np.quantile(bs, 0.95)), 4)]}

    def wins_ab(self, xa, xu):
        d = xa - xu
        return (d > 0) + 0.5 * (d == 0)

    def resid(self, score, clen):
        """exploratory: remove the linear effect of log window length (pooled slope after demeaning within identical-prompt group)."""
        g = np.array([self.grp_of_sample[i] for i in range(len(score))])
        x = np.log(clen); y = score.astype(float).copy()
        for gg in np.unique(g):
            m = g == gg
            x[m] -= x[m].mean(); y[m] -= y[m].mean()
        b = (x * y).sum() / (x * x).sum()
        return y - b * x

    def auroc(self, score):
        return float(self.wins(score).mean())

    def per_q(self, score):
        return np.bincount(self.q, weights=self.wins(score), minlength=self.nq)

    def boot_auc(self, score):
        w = self.per_q(score)
        return (self.boot @ w) / (self.boot @ self.npairs_q)

    def summarize(self, score, level=0.95):
        bs = self.boot_auc(score)
        lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
        return {"auroc": round(self.auroc(score), 4), "ci95": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)],
                "ci90": [round(float(np.quantile(bs, 0.05)), 4), round(float(np.quantile(bs, 0.95)), 4)]}

    def diff(self, s1, s2):
        a1, a2 = self.boot_auc(s1), self.boot_auc(s2)
        d = a1 - a2
        return {"diff": round(self.auroc(s1) - self.auroc(s2), 4), "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]}


def verdict(res):
    lo90, hi90 = res["ci90"]
    if lo90 > 0.42 and hi90 < 0.58:
        return "no signal at detectable size (90% interval inside 0.42-0.58)"
    if res["ci95"][0] > 0.5 and res["auroc"] >= 0.58:
        return "signal (AUROC >= 0.58, interval excludes 0.5)"
    if res["ci95"][1] < 0.5:
        return "below chance (interval excludes 0.5)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/probe_control")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", action="store_true", help="also run the text baselines and the supervised ceiling (needs sklearn)")
    a = ap.parse_args()
    d = Path(a.dir)
    sel = [json.loads(l) for l in open(d / "pc_select.jsonl")]
    P = Pairs(sel)
    bj = json.loads((d / "build/build.json").read_text())
    Ln, Lh = bj["validation"]["nguyen"]["chosen_layer"], bj["validation"]["hua"]["chosen_layer"]
    idx = {json.loads(l)["run_id"]: json.loads(l) for l in open(d / "readout/index.jsonl")}
    N = len(P.samples)
    clen = np.zeros(N)
    nw = {"nguyen": np.zeros(N), "hua": np.zeros(N)}              # window mean, chosen layer
    last = {"nguyen": np.zeros(N), "hua": np.zeros(N)}            # last window position
    early = {"nguyen": np.zeros(N), "hua": np.zeros(N)}           # window ending 64 tokens earlier
    first25 = {"nguyen": np.zeros(N), "hua": np.zeros(N)}         # first 25% of the window
    alll = {"nguyen": np.zeros((N, 40)), "hua": np.zeros((N, 40))}
    rnd = {"nguyen": np.zeros((N, 100)), "hua": np.zeros((N, 100))}
    prompt = {"nguyen": {}, "hua": {}}
    sup = {}
    ser = {}
    for s in sel:
        z = np.load(d / "readout" / idx[s["run_id"]]["file"])
        proj = z["proj"].astype(np.float32)  # [T,40,2]
        rn, rh = z["rnd_n"].astype(np.float32), z["rnd_h"].astype(np.float32)
        for k, c in enumerate(s["cuts"]):
            i = P.sid[(s["run_id"], k)]
            clen[i] = c
            for pi, name in enumerate(("nguyen", "hua")):
                L = Ln if pi == 0 else Lh
                x = proj[:c, :, pi]
                nw[name][i] = x[:, L].mean(); last[name][i] = x[c - 1, L]
                early[name][i] = x[: max(c - 64, 1), L].mean()
                first25[name][i] = x[: max(int(0.25 * c), 1), L].mean()
                alll[name][i] = x.mean(0)
            rnd["nguyen"][i] = rn[:c].mean(0); rnd["hua"][i] = rh[:c].mean(0)
            sup[i] = z["sup"][k]
        ser[s["run_id"]] = {"nguyen": proj[:, Ln, 0], "hua": proj[:, Lh, 1], "rnd_n": rn, "rnd_h": rh}
        prompt["nguyen"][s["run_id"]] = z["prompt_n"]; prompt["hua"][s["run_id"]] = z["prompt_h"]
    out = {"n_pairs": P.n, "n_questions": P.nq, "n_groups": len(set(P.grp)), "layers": {"nguyen": Ln, "hua": Lh},
           "build": bj["validation"] and {k: {kk: v[kk] for kk in ("chosen_layer", "auroc_select_half", "auroc_heldout_half", "passes_0.70")} for k, v in bj["validation"].items()}}
    # as published: prompt-level score is identical within a group
    pub = {}
    for name in ("nguyen", "hua"):
        L = Ln if name == "nguyen" else Lh
        sc = np.array([prompt[name][r][L] for r, k in P.samples])
        pub[name] = {"auroc": round(P.auroc(sc), 4), "max_within_group_spread": float(max(np.ptp([prompt[name][r][L] for r in {s["run_id"] for s in sel if s["question_id"] == g}]) for g in set(P.grp)))}
    out["as_published_prompt_level"] = pub
    prim = {}
    for name in ("nguyen", "hua"):
        r = P.summarize(nw[name]); r["verdict"] = verdict(r)
        rb = np.array([P.auroc(rnd[name][:, j]) for j in range(100)])
        r["random_directions_auroc_band_2.5_97.5"] = [round(float(np.quantile(rb, 0.025)), 4), round(float(np.quantile(rb, 0.975)), 4)]
        r["random_directions_sd"] = round(float(rb.std()), 4)
        r["above_random_band"] = bool(r["auroc"] > np.quantile(rb, 0.975))
        prim[name] = r
    out["primary_window_mean"] = prim
    out["secondary"] = {}
    for tag, dct in (("last_position", last), ("window_ending_64_earlier", early), ("first_25pct", first25)):
        out["secondary"][tag] = {n: P.summarize(dct[n]) for n in dct}
    out["by_layer_window_mean"] = {n: [round(P.auroc(alll[n][:, l]), 3) for l in range(40)] for n in alll}
    out["window_length_alone"] = P.summarize(clen)
    capped = np.array([clen[P.su[i]] < clen[P.sa[i]] for i in range(P.n)])
    out["pairs_with_sibling_window_shorter"] = int(capped.sum())
    # sensitivity: exclude capped pairs
    keep = ~capped
    def auc_keep(score):
        return round(float(P.wins(score)[keep].mean()), 4)
    out["secondary"]["excluding_capped_pairs"] = {n: auc_keep(nw[n]) for n in nw}
    out["secondary"]["excluding_capped_pairs"]["window_length_alone"] = auc_keep(clen)
    Pu = P.subset(keep)
    def band(rn):
        rb = np.array([Pu.auroc(rn[:, j]) for j in range(100)])
        return [round(float(np.quantile(rb, 0.025)), 4), round(float(np.quantile(rb, 0.975)), 4)]
    out["exploratory_uncapped_pairs"] = {"n_pairs": Pu.n, "nguyen": Pu.summarize(nw["nguyen"]), "hua": Pu.summarize(nw["hua"]),
        "nguyen_random_band": band(rnd["nguyen"]), "hua_random_band": band(rnd["hua"]), "window_length_alone": Pu.summarize(clen)}
    # lens
    lj = d / "lens"
    if (lj / "index.jsonl").exists():
        tk = json.loads((lj / "tokens.json").read_text())
        cols = tk["columns"]; K = len(cols) - 1
        prim_c = [cols.index(" fictional"), cols.index(" simulation")]
        lidx = {json.loads(l)["run_id"]: json.loads(l) for l in open(lj / "index.jsonl")}
        lens_sc = {"jlens": np.zeros(N), "logit": np.zeros(N)}
        for s in sel:
            e = lidx[s["run_id"]]
            z = np.load(lj / e["file"])
            lp = z["lp"].astype(np.float32)  # [lens, layers, pos, K+1]
            per_pos = (lp[..., prim_c].mean(-1) - lp[..., K]).mean(1)  # [lens, pos]
            nwin = int(z["n_window"])
            for si, nm in enumerate(tk["lenses"]):
                ser[s["run_id"]][nm] = per_pos[si, :nwin]
            for k, c in enumerate(s["cuts"]):
                i = P.sid[(s["run_id"], k)]
                for si, nm in enumerate(tk["lenses"]):
                    lens_sc[nm][i] = per_pos[si, :c].mean()
        out["lens"] = {n: P.summarize(lens_sc[n]) for n in lens_sc}
        for n in lens_sc:
            out["lens"][n]["verdict"] = verdict(out["lens"][n])
        out["lens"]["jlens_minus_nguyen"] = P.diff(lens_sc["jlens"], nw["nguyen"])
        out["lens"]["jlens_minus_hua"] = P.diff(lens_sc["jlens"], nw["hua"])
        out["lens"]["jlens_minus_logit"] = P.diff(lens_sc["jlens"], lens_sc["logit"])
        out["lens"]["layers"] = tk["layers"]
        np.save(d / "lens_scores.npy", np.stack([lens_sc["jlens"], lens_sc["logit"]]))
        out["exploratory_uncapped_pairs"]["jlens"] = Pu.summarize(lens_sc["jlens"])
        out["exploratory_uncapped_pairs"]["logit"] = Pu.summarize(lens_sc["logit"])
    # ---- exploratory: length-matched pairs and length-residualised scores (not pre-registered)
    m_len = np.minimum(clen[P.sa], clen[P.su]).astype(int)
    def matched(key):
        xa = np.array([ser[P.samples[P.sa[i]][0]][key][:m_len[i]].mean(0) for i in range(P.n)])
        xu = np.array([ser[P.samples[P.su[i]][0]][key][:m_len[i]].mean(0) for i in range(P.n)])
        return xa, xu
    ex = {"n_pairs": P.n, "note": "exploratory, added after seeing that window length alone scores 0.66; not in the pre-registration"}
    for name, key, rk in (("nguyen", "nguyen", "rnd_n"), ("hua", "hua", "rnd_h")):
        xa, xu = matched(key)
        ex[name] = P.summ_w(P.wins_ab(xa, xu))
        ra, ru = matched(rk)
        rb = np.array([P.wins_ab(ra[:, j], ru[:, j]).mean() for j in range(100)])
        ex[name]["random_directions_band_2.5_97.5"] = [round(float(np.quantile(rb, 0.025)), 4), round(float(np.quantile(rb, 0.975)), 4)]
        ex[name]["random_directions_sd"] = round(float(rb.std()), 4)
    if "lens" in out:
        for nm in ("jlens", "logit"):
            xa, xu = matched(nm)
            ex[nm] = P.summ_w(P.wins_ab(xa, xu))
            ex["_matched_" + nm] = (xa, xu)
    out["exploratory_length_matched"] = {k: v for k, v in ex.items() if not k.startswith("_")}
    res_sc = {"nguyen": P.resid(nw["nguyen"], clen), "hua": P.resid(nw["hua"], clen)}
    if "lens" in out:
        res_sc["jlens"], res_sc["logit"] = P.resid(lens_sc["jlens"], clen), P.resid(lens_sc["logit"], clen)
    out["exploratory_length_residualised"] = {k: P.summarize(v) for k, v in res_sc.items()}
    for nm in ("nguyen", "hua"):
        rb = np.array([P.auroc(P.resid(rnd[nm][:, j], clen)) for j in range(100)])
        out["exploratory_length_residualised"][nm]["random_directions_band_2.5_97.5"] = [round(float(np.quantile(rb, 0.025)), 4), round(float(np.quantile(rb, 0.975)), 4)]
    np.savez(d / "pc_scores.npz", nguyen=nw["nguyen"], hua=nw["hua"], clen=clen, sa=P.sa, su=P.su, q=P.q)
    if a.text:
        out["text"] = text_and_ceiling(d, sel, P, nw, lens_sc if "lens" in out else None, sup, m_len, clen, res_sc)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_layer_window_mean",)}, indent=1)[:6000])


def text_and_ceiling(d, sel, P, nw, lens_sc, sup, m_len, clen, res_sc):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", revision="40c069824f4251a91eefaf281ebe4c544efd3e18")
    runs = {}
    want = {s["run_id"] for s in sel}
    for l in open(d / "runs.jsonl"):
        r = json.loads(l)
        if r["run_id"] in want:
            runs[r["run_id"]] = (len(r["prompt_token_ids"]), r["completion_token_ids"])
    texts, y, qid = [], [], []
    smap = {v: k for k, v in P.sid.items()}
    bygroup = {s["run_id"]: s for s in sel}
    for i in range(len(P.samples)):
        rid, k = P.samples[i]
        s = bygroup[rid]
        c = s["cuts"][k]
        texts.append(tok.decode(runs[rid][1][:c]))
        y.append(1 if s["group"] == "aware" else 0)
        qid.append(s["question_id"].split("|")[0])
    y, qid = np.array(y), np.array(qid)
    res = {}
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, qid))
    # (iii) task labels, questions held out
    oof = np.zeros(len(y))
    for tr, te in folds:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = v.fit_transform([texts[i] for i in tr]); Xte = v.transform([texts[i] for i in te])
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        oof[te] = m.decision_function(Xte)
    res["tfidf_trained_on_task_labels"] = P.summarize(oof)
    res["tfidf_task_trained_length_residualised_EXPLORATORY"] = P.summarize(P.resid(oof, clen))
    # exploratory length-matched: score both texts of a pair cut to the same number of tokens, classifier trained on the other folds
    wm = np.zeros(P.n)
    qpair = np.array([qid[P.sa[i]] for i in range(P.n)])
    for tr, te in folds:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = v.fit_transform([texts[i] for i in tr])
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        teq = set(qid[te])
        for i in range(P.n):
            if qpair[i] in teq:
                ta = tok.decode(runs[P.samples[P.sa[i]][0]][1][:m_len[i]]); tu = tok.decode(runs[P.samples[P.su[i]][0]][1][:m_len[i]])
                da, du = m.decision_function(v.transform([ta, tu]))
                wm[i] = 1.0 if da > du else (0.5 if da == du else 0.0)
    res["tfidf_task_trained_length_matched_EXPLORATORY"] = P.summ_w(wm)
    # (ii) trained on the benchmark-vs-user validation prompts (zero-shot for this task)
    val = json.load(open(d / "val_prompts.json"))
    vt = [s for s, _ in val["bench"]] + [s for s, _ in val["user"]]; vy = [1] * len(val["bench"]) + [0] * len(val["user"])
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(v.fit_transform(vt), vy)
    sc_val = m.decision_function(v.transform(texts))
    res["tfidf_trained_on_benchmark_vs_user_prompts"] = P.summarize(sc_val)
    res["jlens_minus_text_task_trained"] = P.diff(lens_sc["jlens"], oof) if lens_sc else None
    res["nguyen_minus_text_task_trained"] = P.diff(nw["nguyen"], oof)
    res["hua_minus_text_task_trained"] = P.diff(nw["hua"], oof)
    np.save(d / "text_oof.npy", oof)
    if (d / "text_emb.npy").exists():
        E = np.load(d / "text_emb.npy")
        oe = np.zeros(len(y))
        for tr, te in folds:
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(E[tr], y[tr])
            oe[te] = m.decision_function(E[te])
        res["embedding_classifier_trained_on_task_labels"] = P.summarize(oe)
        res["embedding_task_trained_length_residualised_EXPLORATORY"] = P.summarize(P.resid(oe, clen))
        res["jlens_minus_embedding"] = P.diff(lens_sc["jlens"], oe) if lens_sc else None
        res["nguyen_minus_embedding"] = P.diff(nw["nguyen"], oe)
        res["hua_minus_embedding"] = P.diff(nw["hua"], oe)
    # supervised ceiling on window-mean activations (questions held out); fixed C, no tuning
    ceil = {}
    X_all = np.stack([sup[i] for i in range(len(y))]).astype(np.float32)  # [N, 10, D]
    for li in range(X_all.shape[1]):
        o = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(X_all[tr, li])
            m = LogisticRegression(C=0.001, max_iter=500, class_weight="balanced").fit(sc.transform(X_all[tr, li]), y[tr])
            o[te] = m.decision_function(sc.transform(X_all[te, li]))
        ceil[str(li)] = P.summarize(o)["auroc"]
    res["supervised_ceiling_by_sup_layer_index"] = ceil
    return res


if __name__ == "__main__":
    main()
