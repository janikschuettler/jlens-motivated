"""qwen8b-square-silent: pre-registered analysis Q-sq (G1-G3 mirror) (+ exploratory) on stored Gemma readouts (CPU; numpy).

uv run --project envs/judge python scripts/lens/gemma_analyze.py --lens data/explore/gemma27/lens_all \
    --runs data/explore/gemma27/hints/runs.jsonl data/explore/gemma27/subtle/runs.jsonl \
    --flags data/explore/gemma27/select_flags.json --out docs/experiments/gemma27-silent/data/lens_results.json

Score S (pre-registered): at pre_answer, log p(correct letter) - mean log p(unchosen wrong letters), averaged over the
workspace blocks; letter log-prob = logsumexp over forms. D = log p(correct) - log p(hinted letter). Items are the
resampling unit (2,000 bootstrap resamples, seed 0).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 2000


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_items(groups, stat, seed=0):
    """groups: dict name -> list of (item, value or tuple). stat(sample dict name -> list of values) -> float."""
    rng = np.random.default_rng(seed)
    per = {g: defaultdict(list) for g in groups}
    for g, lst in groups.items():
        for item, v in lst:
            per[g][item].append(v)
    keys = sorted(set().union(*[set(p) for p in per.values()]))
    est = stat({g: [v for k in per[g] for v in per[g][k]] for g in groups})
    bs = []
    for _ in range(B):
        samp = rng.choice(keys, size=len(keys))
        s = {g: [v for k in samp for v in per[g].get(k, [])] for g in groups}
        try:
            val = stat(s)
        except Exception:
            continue
        if np.isfinite(val):
            bs.append(val)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": round(float(est), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "n_runs": {g: len(groups[g]) for g in groups}, "n_items": {g: len(per[g]) for g in groups}}


def load(lens_dirs, runs_paths, flags):
    idx_all, out_all = None, []
    for ld in lens_dirs:
        i_, o_ = load_one(ld, runs_paths, flags)
        if idx_all is None:
            idx_all = i_
        else:
            idx_all["runs"] += i_["runs"]
            idx_all["flagged"] += i_["flagged"]
        out_all += o_
    return idx_all, out_all


def load_one(lens_dir, runs_paths, flags):
    idx = json.load(open(Path(lens_dir) / "index.json"))
    meta_runs = {}
    for p in runs_paths:
        for line in open(p):
            r = json.loads(line)
            meta_runs[r["run_id"]] = {k: r[k] for k in ("correct_letter", "hint_letter", "letters", "question_id")}
    out = []
    for i, rid in enumerate(idx["runs"]):
        z = np.load(Path(lens_dir) / f"{i:05d}.npz")
        meta = json.loads(str(z["meta"]))
        tl = z["tl"].astype(np.float32)  # [2, L, P, 8]
        L = tl.shape[1]
        let = np.logaddexp(tl[..., 0::2], tl[..., 1::2])  # [2, L, P, 4]
        r = meta_runs[rid]
        f = flags[rid]
        loc = meta["loc"]
        pos = meta["tl_pos"]
        pa = pos.index(loc["pre_answer"])
        letters = r["letters"]
        ci = letters.index(r["correct_letter"])
        hi = letters.index(r["hint_letter"]) if r["hint_letter"] else None
        ans = letters.index(loc["answer_letter"])
        others = [k for k in range(4) if k != ci and k != ans]
        d = {"run_id": rid, "item": r["question_id"], "group": f["group"], "arm": f["arm"], "kind": f["kind"], "tokens": f["tokens"]}
        # S (switch runs only: the answer is a wrong letter)
        if ans != ci:
            s_l = let[:, :, pa, ci] - let[:, :, pa, others].mean(-1)  # [2, L]
            d["S"] = s_l.mean(1)  # per lens
            d["S_block"] = s_l
            d["top3"] = (let[:, :, pa, ci] > let[:, :, pa, others].max(-1)).mean(1)  # share of blocks where correct is the highest
            # by distance before pre_answer
            prof = {}
            for k in (0, 1, 2, 4, 8, 16, 32, 48):
                if loc["pre_answer"] - k in pos:
                    q = pos.index(loc["pre_answer"] - k)
                    prof[k] = (let[:, :, q, ci] - let[:, :, q, others].mean(-1)).mean(1)
            d["S_dist"] = prof
        if hi is not None:
            d["D"] = (let[:, :, pa, ci] - let[:, :, pa, hi]).mean(1)
            d["D_block"] = let[:, :, pa, ci] - let[:, :, pa, hi]
        # option words: margin of an option's own letter over the other three at that option's end
        ow = {}
        for nm, k in (("correct", ci), ("hint", hi)):
            if k is None:
                continue
            q = pos.index(loc["opt_end"][letters[k]])
            own = let[:, :, q, k]
            oth = let[:, :, q, [j for j in range(4) if j != k]].mean(-1)
            ow[nm] = (own - oth).mean(1)
        d["opt"] = ow
        d["resid"] = z["resid"].astype(np.float32)  # [L, n_pos, d]
        out.append(d)
    return idx, out


def analyze(runs, names, idx, res):
    wording = runs
    sw = [d for d in wording if d["group"] in ("silent", "open")]
    sil = [d for d in sw if d["group"] == "silent"]
    opn = [d for d in sw if d["group"] == "open"]
    res["counts"] = {"wording_switch_read": len(sw), "silent": len(sil), "open": len(opn),
                     "items_with_silent": len({d["item"] for d in sil}), "items_with_open": len({d["item"] for d in opn})}

    # G1: mean over items of per-item mean S, silent switches. Use value tuples (item, S) so the item mean is kept.
    for li, nm in enumerate(names):
        rng = np.random.default_rng(0)
        pi = defaultdict(list)
        for d in sil:
            pi[d["item"]].append(d["S"][li])
        keys = sorted(pi)
        est = float(np.mean([np.mean(pi[k]) for k in keys])) if keys else float("nan")
        bs = []
        for _ in range(B):
            samp = rng.choice(keys, size=len(keys))
            bs.append(np.mean([np.mean(pi[k]) for k in samp]))
        pt = defaultdict(list)
        for d in sil:
            pt[d["item"]].append(float(d["top3"][li]))
        est_t = float(np.mean([np.mean(pt[k]) for k in keys])) if keys else float("nan")
        bt = [np.mean([np.mean(pt[k]) for k in rng.choice(keys, size=len(keys))]) for _ in range(B)]
        res[f"G1_{nm}"] = {"mean_S": round(est, 4), "lo": round(float(np.percentile(bs, 2.5)), 4),
                            "hi": round(float(np.percentile(bs, 97.5)), 4), "n_items": len(keys), "n_runs": len(sil),
                            "share_correct_highest_of_unchosen(blocks)": [round(est_t, 4), round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)],
                            "note": "per-item means averaged over items; chance for the share is 1/3"}
        # same for open switches
        po = defaultdict(list)
        for d in opn:
            po[d["item"]].append(d["S"][li])
        if po:
            ko = sorted(po)
            res[f"G1_open_{nm}"] = {"mean_S": round(float(np.mean([np.mean(po[k]) for k in ko])), 4), "n_items": len(ko), "n_runs": len(opn)}
    # G2: silent vs open AUROC of S, with length baseline
    def au_stat(key_fn):
        return lambda s: auroc([v for v in s["silent"]], [v for v in s["open"]])

    for li, nm in enumerate(names):
        g = {"silent": [(d["item"], d["S"][li]) for d in sil], "open": [(d["item"], d["S"][li]) for d in opn]}
        res[f"G2_{nm}"] = boot_items(g, au_stat(None)) if sil and opn else None
    g = {"silent": [(d["item"], d["tokens"]) for d in sil], "open": [(d["item"], d["tokens"]) for d in opn]}
    res["G2_length_baseline"] = boot_items(g, au_stat(None)) if sil and opn else None
    # J minus logit difference
    g = {"silent": [(d["item"], (d["S"][0], d["S"][1])) for d in sil], "open": [(d["item"], (d["S"][0], d["S"][1])) for d in opn]}
    if sil and opn:
        res["G2_J_minus_logit"] = boot_items(
            g, lambda s: auroc([v[0] for v in s["silent"]], [v[0] for v in s["open"]]) - auroc([v[1] for v in s["silent"]], [v[1] for v in s["open"]]))
    # tertiles by length
    if sil and opn:
        toks = np.array([d["tokens"] for d in sw])
        cuts = np.percentile(toks, [33.3, 66.7])
        ter = {}
        for t, (lo_, hi_) in enumerate([(-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9)]):
            sub = [d for d in sw if lo_ < d["tokens"] <= hi_]
            ps = [d["S"][0] for d in sub if d["group"] == "silent"]
            ns = [d["S"][0] for d in sub if d["group"] == "open"]
            ter[f"tertile{t + 1}"] = {"auroc_J": round(auroc(ps, ns), 3) if ps and ns else None, "n_silent": len(ps), "n_open": len(ns)}
        res["G2_length_tertiles_J"] = ter
    # G3: D, resisted vs switch (wording arms)
    res_runs = [d for d in wording if d["group"] == "resisted"]
    for li, nm in enumerate(names):
        g = {"resisted": [(d["item"], d["D"][li]) for d in res_runs], "switch": [(d["item"], d["D"][li]) for d in sw if "D" in d]}
        res[f"G3_{nm}"] = boot_items(g, lambda s: auroc(s["resisted"], s["switch"])) if res_runs and sw else None
    # exploratory (a): S by distance and by block for silent switches
    ex = {}
    for li, nm in enumerate(names):
        dist = {}
        for k in (0, 1, 2, 4, 8, 16, 32, 48):
            vals = defaultdict(list)
            for d in sil:
                if k in d.get("S_dist", {}):
                    vals[d["item"]].append(d["S_dist"][k][li])
            if vals:
                dist[k] = round(float(np.mean([np.mean(v) for v in vals.values()])), 4)
        blk = np.mean([d["S_block"][li] for d in sil], axis=0) if sil else []
        ex[nm] = {"S_by_distance_before_answer": dist, "S_by_block_silent": [round(float(x), 3) for x in blk],
                  "S_by_block_open": [round(float(x), 3) for x in (np.mean([d["S_block"][li] for d in opn], axis=0) if opn else [])]}
    res["explore_a"] = ex
    # (a2) the same G1 statistic over mid-depth blocks only (24-44) and late blocks only (46-54)
    for nm, li in (("J-lens", 0), ("logit lens", 1)):
        for lab, sl in (("blocks14_24", slice(0, 6)), ("blocks26_32", slice(6, 10))):
            pi = defaultdict(list)
            for d in sil:
                pi[d["item"]].append(float(d["S_block"][li][sl].mean()))
            ks = sorted(pi)
            if not ks:
                continue
            rng = np.random.default_rng(0)
            bs = [np.mean([np.mean(pi[k]) for k in rng.choice(ks, size=len(ks))]) for _ in range(B)]
            res.setdefault("explore_a2_G1_by_depth", {})[f"{nm}/{lab}"] = {
                "mean_S": round(float(np.mean([np.mean(pi[k]) for k in ks])), 4),
                "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4), "n_items": len(ks)}
    # (b) option words: correct-option letter margin minus hinted-option margin, silent switches vs resisted
    ob = {}
    for li, nm in enumerate(names):
        def mk(lst):
            return [(d["item"], d["opt"]["correct"][li] - d["opt"]["hint"][li]) for d in lst if "hint" in d["opt"]]
        g = {"silent": mk(sil), "resisted": mk(res_runs)}
        ob[nm] = boot_items(g, lambda s: auroc(s["silent"], s["resisted"])) if g["silent"] and g["resisted"] else None
    res["explore_b_option_words_silent_vs_resisted_AUROC"] = ob
    # (d) subtle arms: S per arm
    sub = defaultdict(list)
    for d in runs:
        if d["kind"] == "subtle" and d["group"] in ("silent", "open"):
            sub[(d["arm"], d["group"])].append(d)
    res["explore_d_subtle"] = {f"{k[0]}/{k[1]}": {"n": len(v), "items": len({d['item'] for d in v}),
                                                  "mean_S_J": round(float(np.mean([d["S"][0] for d in v])), 3),
                                                  "mean_S_logit": round(float(np.mean([d["S"][1] for d in v])), 3)} for k, v in sub.items()}
    # (c) probe: difference of means at pre_answer on block-output residuals, resisted vs switch, item-grouped folds
    pr = {}
    pool = [d for d in wording if d["group"] in ("resisted", "silent", "open")]
    items = sorted({d["item"] for d in pool})
    rng = np.random.default_rng(0)
    fold = {it: int(f) for it, f in zip(items, rng.permutation(len(items)) % 5)}
    for bi in (2, 4, 6, 8):  # blocks 18, 22, 26, 30
        scores = {}
        for f in range(5):
            tr = [d for d in pool if fold[d["item"]] != f and d["group"] in ("resisted", "silent", "open")]
            te = [d for d in pool if fold[d["item"]] == f]
            xr = np.array([d["resid"][bi, 0] for d in tr if d["group"] == "resisted"])
            xs = np.array([d["resid"][bi, 0] for d in tr if d["group"] != "resisted"])
            if len(xr) < 5 or len(xs) < 5:
                continue
            w = xr.mean(0) - xs.mean(0)
            mid = (xr.mean(0) + xs.mean(0)) / 2
            for d in te:
                scores[d["run_id"]] = float((d["resid"][bi, 0] - mid) @ w)
        byid = {d["run_id"]: d for d in pool}
        s_sil = [(byid[k]["item"], v) for k, v in scores.items() if byid[k]["group"] == "silent"]
        s_open = [(byid[k]["item"], v) for k, v in scores.items() if byid[k]["group"] == "open"]
        s_res = [(byid[k]["item"], v) for k, v in scores.items() if byid[k]["group"] == "resisted"]
        pr[f"block{idx['layers'][bi]}"] = {
            "resisted_vs_switch_AUROC": boot_items({"r": s_res, "s": s_sil + s_open}, lambda s: auroc(s["r"], s["s"])),
            "silent_vs_open_AUROC": boot_items({"s": s_sil, "o": s_open}, lambda s: auroc(s["s"], s["o"])) if s_sil and s_open else None,
            "silent_vs_resisted_AUROC": boot_items({"s": s_sil, "r": s_res}, lambda s: auroc(s["s"], s["r"])) if s_sil and s_res else None}
    res["explore_c_probe"] = pr
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    flags = json.load(open(a.flags))
    idx, all_runs = load(a.lens, a.runs, flags)
    names = ["J-lens", "logit lens"]
    final = {"n_read": len(all_runs), "flagged": len(idx["flagged"]), "layers": idx["layers"]}
    for label, sel in (("black_square_only", "subtle"), ("answer_hint_only", "wording")):
        runs = [d for d in all_runs if sel is None or d["kind"] == sel]
        if not runs:
            continue
        res = {}
        analyze(runs, names, idx, res)
        final[label] = res
    res = final
    Path(a.out).write_text(json.dumps(res, indent=1))
    for lab in ("pooled", "black_square_only", "answer_hint_only"):
        if lab in res:
            print(lab, json.dumps({k: v for k, v in res[lab].items() if k.startswith(("G1_J", "G1_log", "G2_J-lens", "G2_log", "G3", "counts"))}))


if __name__ == "__main__":
    main()
