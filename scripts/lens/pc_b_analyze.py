"""probe-control test (b) analysis (CPU). Pre-registered as PC-b (docs/preregistration.md).
  python scripts/lens/pc_b_analyze.py --out docs/experiments/probe-control/data/pc_test_b.json
Inputs: data/probe_control/b/{versions.jsonl,pairs.jsonl,scores.json,gen/runs.jsonl}, data/probe_control/{build/build.json,val_prompts.json}."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explore"))
from evalaware_models_analyze import TEST_ME  # noqa: E402

B = 2000
SCORES = ["nguyen", "hua", "jlens", "logit", "text", "n_tokens"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/probe_control"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); d = Path(a.dir)
    vers = {json.loads(l)["vid"]: json.loads(l) for l in open(d / "b/versions.jsonl")}
    pairs = [json.loads(l) for l in open(d / "b/pairs.jsonl")]
    sc = json.loads((d / "b/scores.json").read_text())
    rows = {r["vid"]: r for r in sc["rows"]}
    bj = json.loads((d / "build/build.json").read_text())
    Ln, Lh = bj["validation"]["nguyen"]["chosen_layer"], bj["validation"]["hua"]["chosen_layer"]
    # spontaneous belief
    cnt, tot, unfinished = defaultdict(int), defaultdict(int), 0
    for l in open(d / "b/gen/runs.jsonl"):
        r = json.loads(l)
        t = r["completion_text"].split("</think>")[0]
        tot[r["vid"]] += 1
        cnt[r["vid"]] += any(p.search(t) for p in TEST_ME)
        unfinished += "</think>" not in r["completion_text"]
    # text baseline: tfidf trained on benchmark-vs-user validation prompts (zero-shot for this task)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    val = json.load(open(d / "val_prompts.json"))
    vt = [s for s, _ in val["bench"]] + [s for s, _ in val["user"]]; vy = [1] * len(val["bench"]) + [0] * len(val["user"])
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(tv.fit_transform(vt), vy)
    vids = sorted(vers)
    txt = {v: "\n".join(m["content"] for m in vers[v]["messages"]) for v in vids}
    tscore = dict(zip(vids, clf.decision_function(tv.transform([txt[v] for v in vids]))))
    S = {v: {"nguyen": rows[v]["nguyen"][Ln], "hua": rows[v]["hua"][Lh], "jlens": rows[v]["jlens"], "logit": rows[v]["logit"],
             "text": float(tscore[v]), "n_tokens": rows[v]["n_tokens"]} for v in vids}
    P = {v: rows[v]["p_test"] for v in vids}
    # pair table
    cl = lambda it: ("q" + it[1:]) if it[0] in "mg" else it
    tab = []
    for p in pairs:
        e, pl = p["eval"], p["plain"]
        dp = P[e] - P[pl]; ds = cnt[e] - cnt[pl]
        tab.append({"pair": p["pair"], "cluster": cl(p["item"]), "e": e, "p": pl, "dp": dp, "ds": ds,
                    "matched": abs(dp) < 0.10 and abs(ds) <= 1, "matched_elicited": abs(dp) < 0.10})
    clusters = sorted({t["cluster"] for t in tab}); ci = {c: i for i, c in enumerate(clusters)}
    q = np.array([ci[t["cluster"]] for t in tab])
    rng = np.random.default_rng(0)
    boot = np.stack([np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)) for _ in range(B)]).astype(float)

    def wins(name):
        x = np.array([S[t["e"]][name] - S[t["p"]][name] for t in tab])
        return (x > 0) + 0.5 * (x == 0)

    def summ(w, mask):
        m = mask
        if m.sum() == 0:
            return None
        pq = np.bincount(q[m], weights=w[m], minlength=len(clusters)); nq = np.bincount(q[m], minlength=len(clusters)).astype(float)
        bs = (boot @ pq) / np.maximum(boot @ nq, 1e-9)
        return {"auroc": round(float(w[m].mean()), 4), "ci95": [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)], "n_pairs": int(m.sum())}

    pid = np.array([t["pair"] for t in tab])
    matched = np.array([t["matched"] for t in tab]); mel = np.array([t["matched_elicited"] for t in tab])
    W = {n: wins(n) for n in SCORES}
    out = {"layers": {"nguyen": Ln, "hua": Lh}, "n_versions": len(vids), "n_pairs": len(tab), "unfinished_runs": unfinished,
           "n_runs": int(sum(tot.values())), "n_clusters": len(clusters)}
    prim = np.isin(pid, ["P1", "P2", "P3", "P4"])
    out["n_matched_primary_pool"] = int((matched & prim).sum())
    out["underpowered"] = bool((matched & prim).sum() < 60)
    res = {}
    for n in SCORES:
        res[n] = {"primary_matched_P1-P4": summ(W[n], matched & prim), "all_pairs_P1-P4": summ(W[n], prim), "matched_elicited_only_P1-P4": summ(W[n], mel & prim)}
        for pp in ("P1", "P2", "P3", "P4", "P5"):
            res[n][f"{pp}_all"] = summ(W[n], pid == pp); res[n][f"{pp}_matched"] = summ(W[n], (pid == pp) & matched)
    out["pair_auroc"] = res
    out["pairs_by_set"] = {pp: {"n": int((pid == pp).sum()), "n_matched": int(((pid == pp) & matched).sum()),
                                "mean_dp_elicited": round(float(np.mean([t["dp"] for t in tab if t["pair"] == pp])), 3),
                                "mean_abs_dp": round(float(np.mean([abs(t["dp"]) for t in tab if t["pair"] == pp])), 3),
                                "mean_ds_spontaneous_of_4": round(float(np.mean([t["ds"] for t in tab if t["pair"] == pp])), 3),
                                "mean_token_diff": round(float(np.mean([S[t["e"]]["n_tokens"] - S[t["p"]]["n_tokens"] for t in tab if t["pair"] == pp])), 1)} for pp in ("P1", "P2", "P3", "P4", "P5")}
    kinds = defaultdict(list)
    for v in vids:
        kinds[vers[v]["kind"]].append(v)
    out["belief_by_kind"] = {k: {"n": len(vs), "elicited_p_test_mean": round(float(np.mean([P[v] for v in vs])), 3),
                                 "spontaneous_testing_me_rate": round(sum(cnt[v] for v in vs) / max(sum(tot[v] for v in vs), 1), 3)} for k, vs in kinds.items()}
    # regression across versions: z-scored score ~ eval-surface + ai-topic + elicited belief + set dummies; cluster bootstrap over underlying items
    evalset = {t["e"] for t in tab if t["pair"] in ("P1", "P2", "P3", "P4")}
    X0 = lambda v: [1.0, float(v in evalset), float(vers[v]["kind"] == "user_ai_topic"), P[v], float(v[0] == "m"), float(v[0] == "g")]
    clv = {v: cl(v.split("|")[0]) for v in vids}
    X = np.array([X0(v) for v in vids]); cv = np.array([ci[clv[v]] for v in vids])
    reg = {}
    for n in SCORES:
        y = np.array([S[v][n] for v in vids]); y = (y - y.mean()) / (y.std() + 1e-12)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        bs = []
        for b in range(500):
            idx = np.concatenate([np.where(cv == c)[0] for c in rng.integers(0, len(clusters), len(clusters))])
            bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        bs = np.array(bs)
        reg[n] = {nm: {"coef_sd": round(float(beta[k]), 3), "ci95": [round(float(np.quantile(bs[:, k], .025)), 3), round(float(np.quantile(bs[:, k], .975)), 3)]}
                  for k, nm in ((1, "eval_surface"), (2, "ai_topic_sentence"), (3, "elicited_belief"))}
    out["regression_z_score_on_surface_and_belief"] = reg
    out["scores_by_kind_mean"] = {k: {n: round(float(np.mean([S[v][n] for v in vs])), 3) for n in ("nguyen", "hua", "jlens", "logit", "text")} for k, vs in kinds.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_matched_primary_pool", "underpowered", "belief_by_kind", "pairs_by_set")}, indent=1))
    for n in SCORES:
        print(n, res[n]["primary_matched_P1-P4"], res[n]["all_pairs_P1-P4"], res[n]["P5_all"])


if __name__ == "__main__":
    main()
