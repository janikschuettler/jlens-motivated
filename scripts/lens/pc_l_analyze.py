"""probe-control PC-L (exploratory): length confound in test (b). CPU.
  python scripts/lens/pc_l_analyze.py --dir data/probe_control --out docs/experiments/probe-control/data/pc_test_b_length.json
(1) pair AUROC as in pc_b_analyze.py (same matching: elicited belief within 0.10 and spontaneous count within 1 of 4, same pairs);
(2) residualised: each score minus slope * log(token count), slope = pooled within-(set,kind) regression over the 640 versions;
(3) padded: eval/plain scores after the shorter version is padded to equal token count (pc_l_pad.py), same pairs.
Bootstrap over underlying items (clusters), 2,000 resamples, seed 0."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explore"))
from evalaware_models_analyze import TEST_ME  # noqa: E402
B = 2000
NAMES = ["nguyen", "hua", "jlens", "logit"]
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/probe_control"); ap.add_argument("--out", required=True)
a = ap.parse_args(); d = Path(a.dir)
vers = {json.loads(l)["vid"]: json.loads(l) for l in open(d / "b/versions.jsonl")}
pairs = [json.loads(l) for l in open(d / "b/pairs.jsonl")]
rows = {r["vid"]: r for r in json.loads((d / "b/scores.json").read_text())["rows"]}
bj = json.loads((d / "build/build.json").read_text()); Ln, Lh = bj["validation"]["nguyen"]["chosen_layer"], bj["validation"]["hua"]["chosen_layer"]
cnt = defaultdict(int)
for l in open(d / "b/gen/runs.jsonl"):
    r = json.loads(l); cnt[r["vid"]] += any(p.search(r["completion_text"].split("</think>")[0]) for p in TEST_ME)
P = {v: rows[v]["p_test"] for v in rows}
S0 = {v: {"nguyen": rows[v]["nguyen"][Ln], "hua": rows[v]["hua"][Lh], "jlens": rows[v]["jlens"], "logit": rows[v]["logit"]} for v in rows}
cl = lambda it: ("q" + it[1:]) if it[0] in "mg" else it
tab = [{"pair": p["pair"], "cluster": cl(p["item"]), "e": p["eval"], "p": p["plain"],
        "matched": abs(P[p["eval"]] - P[p["plain"]]) < 0.10 and abs(cnt[p["eval"]] - cnt[p["plain"]]) <= 1} for p in pairs]
clusters = sorted({t["cluster"] for t in tab}); ci = {c: i for i, c in enumerate(clusters)}; q = np.array([ci[t["cluster"]] for t in tab])
pid = np.array([t["pair"] for t in tab]); matched = np.array([t["matched"] for t in tab])
rng = np.random.default_rng(0)
boot = np.stack([np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)) for _ in range(B)]).astype(float)
def summ(w, m):
    if m.sum() == 0: return None
    pq = np.bincount(q[m], weights=w[m], minlength=len(clusters)); nq = np.bincount(q[m], minlength=len(clusters)).astype(float)
    bs = (boot @ pq) / np.maximum(boot @ nq, 1e-9)
    return {"auroc": round(float(w[m].mean()), 3), "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)], "n_pairs": int(m.sum())}
def wins_from(se, sp):
    x = np.array(se) - np.array(sp); return (x > 0) + 0.5 * (x == 0)
subsets = {"pooled_P1-P4": matched & np.isin(pid, ["P1", "P2", "P3", "P4"]), "P1": matched & (pid == "P1"), "P2": matched & (pid == "P2"),
           "P3": matched & (pid == "P3"), "P4": matched & (pid == "P4"), "P5_control": matched & (pid == "P5")}
def table(Se, Sp):
    return {n: {k: summ(wins_from([Se[i][n] for i in range(len(tab))], [Sp[i][n] for i in range(len(tab))]), m) for k, m in subsets.items()} for n in NAMES}
out = {"layers": {"nguyen": Ln, "hua": Lh}}
Se = [S0[t["e"]] for t in tab]; Sp = [S0[t["p"]] for t in tab]
out["original"] = table(Se, Sp)
ntok = {v: rows[v]["n_tokens"] for v in rows}
out["token_diff_eval_minus_plain"] = {k: {"mean": float(np.mean([ntok[t["e"]] - ntok[t["p"]] for t, pp in zip(tab, pid) if pp == k])),
                                         "share_eval_longer": float(np.mean([ntok[t["e"]] > ntok[t["p"]] for t, pp in zip(tab, pid) if pp == k]))} for k in ("P1", "P2", "P3", "P4", "P5")}
out["length_alone"] = {k: summ(wins_from([ntok[t["e"]] for t in tab], [ntok[t["p"]] for t in tab]), m) for k, m in subsets.items()}
# (2) residualised on log token count: slope pooled within (version kind) cells over all versions
beta = {}
for n in NAMES:
    cell = defaultdict(list)
    for v in rows: cell[vers[v]["kind"]].append(v)
    sxy = sxx = 0.0
    for vs in cell.values():
        L = np.log([ntok[v] for v in vs]); Sx = np.array([S0[v][n] for v in vs]); L -= L.mean(); Sx -= Sx.mean(); sxy += (L * Sx).sum(); sxx += (L * L).sum()
    beta[n] = sxy / sxx
R = {v: {n: S0[v][n] - beta[n] * np.log(ntok[v]) for n in NAMES} for v in rows}
out["residualised_on_log_tokens"] = table([R[t["e"]] for t in tab], [R[t["p"]] for t in tab]); out["residual_slopes"] = {k: float(v) for k, v in beta.items()}
# (2b) pair-level: regress score difference on log token ratio across ALL pairs of P1-P4 (change-type fixed effects), residual pair AUROC
dl = np.array([np.log(ntok[t["e"]] / ntok[t["p"]]) for t in tab])
res2 = {}
for n in NAMES:
    dy = np.array([S0[t["e"]][n] - S0[t["p"]][n] for t in tab]); keep = np.isin(pid, ["P1", "P2", "P3", "P4", "P5"])
    X = np.stack([dl] + [(pid == k).astype(float) for k in ("P1", "P2", "P3", "P4", "P5")], 1)
    bt = np.linalg.lstsq(X[keep], dy[keep], rcond=None)[0]
    r = dy - dl * bt[0]
    res2[n] = {"slope_per_log_ratio": float(bt[0]), **{k: summ((r > 0) + 0.5 * (r == 0), m) for k, m in subsets.items()}}
out["pair_difference_residualised_on_log_token_ratio"] = res2
# (3) padded
pp = Path(d / "b/pad_scores.json")
if pp.exists():
    pad = json.loads(pp.read_text()); out["pad_sanity"] = pad["sanity"]
    byk = {(r["pair"], r["item"]): r for r in pad["rows"]}
    ok = [(t["pair"], [p for p in pairs if p["eval"] == t["e"] and p["plain"] == t["p"] and p["pair"] == t["pair"]][0]["item"]) in byk for t in tab]
    if all(ok):
        Sep = []; Spp = []
        for t, p in zip(tab, pairs):
            r = byk[(p["pair"], p["item"])]
            Sep.append({n: (r["scores"]["eval"][n][Ln] if n == "nguyen" and isinstance(r["scores"]["eval"][n], list) else r["scores"]["eval"][n][Lh] if n == "hua" and isinstance(r["scores"]["eval"][n], list) else r["scores"]["eval"][n]) for n in NAMES})
            Spp.append({n: (r["scores"]["plain"][n][Ln] if n == "nguyen" and isinstance(r["scores"]["plain"][n], list) else r["scores"]["plain"][n][Lh] if n == "hua" and isinstance(r["scores"]["plain"][n], list) else r["scores"]["plain"][n]) for n in NAMES})
        out["padded_to_equal_tokens"] = table(Sep, Spp)
        out["padded_token_check"] = {"max_abs_diff_after_pad": int(max(abs(r["n_eval"] - r["n_plain"]) if r["padded"] is None else abs(r["n_padded"] - max(r["n_eval"], r["n_plain"])) for r in pad["rows"]))}
    else:
        out["padded_to_equal_tokens"] = "pairs missing"
Path(a.out).write_text(json.dumps(out, indent=1))
for k in ("original", "residualised_on_log_tokens", "padded_to_equal_tokens"):
    if k in out and isinstance(out[k], dict):
        print(k)
        for n in NAMES: print(" ", n, {s: (v["auroc"] if v else None) for s, v in out[k][n].items()})
print("length alone", {s: (v["auroc"] if v else None) for s, v in out["length_alone"].items()})
print("pair-resid", {n: {s: (v["auroc"] if isinstance(v, dict) and v else None) for s, v in res2[n].items() if s != "slope_per_log_ratio"} for n in NAMES})
