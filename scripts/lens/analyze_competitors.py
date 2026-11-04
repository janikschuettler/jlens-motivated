"""J-lens against its competitors on every multiple-choice case we have read out.
EXPLORATORY. Overnight 2026-11-04, CPU only, on the per-run features written by
scripts/lens/competitors_extract.py (no new readouts, no GPU).

Fixed before computing any AUROC here (the scores were all defined earlier in the
project; the choice of one score per readout and position is made here):

  Positive group = the run that argues against its own knowledge or cheats (told to
  argue for B; hint follower; silent switch; kept the planted claim; grader-bug
  follower). AUROC = P(score of a positive run > score of a negative run); 0.5 is
  chance, 1 or 0 is perfect separation (0 = perfect in the other direction).
  95% CI: 2,000 bootstrap resamples of items (questions), runs pooled.

  Readouts, per position (last prompt token, mid reasoning, end of reasoning, answer
  position), from the extractor's features:
    J-lens / logit lens   D = lp(correct) - lp(pushed letter B), mean over the stored
                          workspace layers ("fixed"). Also the best single layer
                          (largest |AUROC - 0.5|; chosen after looking, optimistic)
                          and the margin M = lp(correct) - mean lp(the two uninvolved
                          letters), layer mean.
    model's own output    D from the model's own next-token distribution at the
                          answer position (black-box readout).
    linear probe          logistic regression on block outputs (standardised,
                          C = 0.05, balanced classes), trained on the group label
                          itself, 5-fold GroupKFold by item; every run is scored by
                          the fold that held out its item. Fixed layer = the middle
                          of the three stored layers; also best of the three
                          (optimistic). Mid reasoning only where the set stores
                          block outputs along the reasoning.
    keyword monitor       1 if the label file's mention list fires (hint,
                          instruction or grader words), else 0.
    length                completion length; reasoning length.
  Within-item version (items with both groups): each score minus its item mean.

Runs on octavia (needs scikit-learn):
    uv run --project envs/lens python scripts/lens/analyze_competitors.py \
        --features data/overnight/competitors --out data/overnight/competitors/auroc.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, SEED = 2000, 0
POS = ("last_prompt", "mid", "think_end", "pre_answer")
CASES = {  # case -> feature files (merged)
    "instr": ["instr.npz"],
    "wording": ["wording.npz"],
    "c3": ["c3.npz"],
    "silent": ["silent_a.npz", "silent_b.npz"],
    "grader_q14": ["grader_q14.npz"],
    "grader_gptoss": ["grader_gptoss.npz"],
    "grader_nt8": ["grader_nt8.npz"],
    "grader_nt14": ["grader_nt14.npz"],
}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = allv.argsort(kind="mergesort").argsort(kind="mergesort").astype(float) + 1
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Boot:
    """Item bootstrap shared by every score of a case (same resamples for all)."""

    def __init__(self, items: np.ndarray, group: np.ndarray, rng) -> None:
        self.items, self.group = items, group
        keys = sorted(set(items.tolist()))
        rows = defaultdict(list)
        for i, q in enumerate(items):
            rows[q].append(i)
        self.samples = []
        for _ in range(N_BOOT):
            pick = rng.choice(len(keys), size=len(keys))
            self.samples.append(np.concatenate([rows[keys[k]] for k in pick]))

    def __call__(self, score: np.ndarray, mask: np.ndarray | None = None) -> dict:
        ok = np.isfinite(score) if mask is None else (np.isfinite(score) & mask)
        g = self.group
        a = auroc(score[ok & (g == 1)], score[ok & (g == 0)])
        if not np.isfinite(a):
            return {"auroc": None}
        bs = []
        for s in self.samples:
            s = s[ok[s]]
            p, n = score[s][g[s] == 1], score[s][g[s] == 0]
            if len(p) and len(n):
                bs.append(auroc(p, n))
        return {"auroc": round(a, 4), "lo": round(float(np.percentile(bs, 2.5)), 4),
                "hi": round(float(np.percentile(bs, 97.5)), 4),
                "n_pos": int((ok & (g == 1)).sum()), "n_neg": int((ok & (g == 0)).sum()),
                "items_pos": len(set(self.items[ok & (g == 1)].tolist())),
                "items_neg": len(set(self.items[ok & (g == 0)].tolist()))}


def centre(score: np.ndarray, items: np.ndarray, both: np.ndarray) -> np.ndarray:
    out = np.full_like(score, np.nan, dtype=float)
    for q in set(items[both].tolist()):
        ii = np.where((items == q) & np.isfinite(score))[0]
        if len(ii):
            out[ii] = score[ii] - score[ii].mean()
    return out


def probe_scores(X: np.ndarray, y: np.ndarray, items: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    out = np.full(len(y), np.nan)
    ok = np.isfinite(X).all(1)
    idx = np.where(ok)[0]
    if len(set(y[idx].tolist())) < 2 or len(set(items[idx].tolist())) < 5:
        return out
    for tr, te in GroupKFold(5).split(idx, y[idx], items[idx]):
        tr, te = idx[tr], idx[te]
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.decision_function(sc.transform(X[te]))
    return out


def load_case(fdir: Path, files: list[str]) -> dict:
    parts = [np.load(fdir / f, allow_pickle=False) for f in files]
    metas = [json.loads(str(p["meta_json"])) for p in parts]
    d = {"meta": metas[0], "runs": [r for m in metas for r in m["runs"]]}
    for k in ("lens", "model", "resid", "cot_len", "completion_len"):
        d[k] = np.concatenate([p[k] for p in parts])
    d["cot_resid"] = parts[0]["cot_resid"] if len(parts) == 1 and parts[0]["cot_resid"].size else None
    d["counts"] = [m["counts"] for m in metas]
    return d


def analyse(name: str, d: dict, rng) -> dict:
    runs = d["runs"]
    items = np.array([r["item"] for r in runs])
    g = np.array([r["group"] for r in runs])
    both_items = set(items[g == 1].tolist()) & set(items[g == 0].tolist())
    both = np.array([q in both_items for q in items])
    boot = Boot(items, g, rng)
    layers = d["meta"]["lens_layers"]
    S: dict[str, np.ndarray] = {}
    lens = d["lens"]  # [runs, 4 pos, 2 score (D, M), 2 lens, L]
    for pi, pos in enumerate(POS):
        for li, ln in enumerate(("jlens", "logit")):
            S[f"{ln}|{pos}|D"] = np.nanmean(lens[:, pi, 0, li, :], -1)
            S[f"{ln}|{pos}|M"] = np.nanmean(lens[:, pi, 1, li, :], -1)
    S["model|pre_answer|D"] = d["model"][:, 2, 0]
    S["model|pre_answer|M"] = d["model"][:, 2, 1]
    S["monitor|text|flag"] = np.array([r["monitor"] for r in runs], float)
    S["length|completion|tokens"] = d["completion_len"].astype(float)
    S["length|reasoning|tokens"] = d["cot_len"].astype(float)
    pl = d["meta"]["probe_layers"]
    for ai, an in enumerate(d["meta"]["resid_anchors"]):
        for k, L in enumerate(pl):
            X = d["resid"][:, ai, k].astype(np.float32)
            S[f"probe|{an}|L{L}"] = probe_scores(X, g, items)
    if d["cot_resid"] is not None:
        for k, L in enumerate(d["meta"]["cot_layers"]):
            S[f"probe|mid|L{L}"] = probe_scores(d["cot_resid"][:, k].astype(np.float32), g, items)
    res = {"n_runs_pos": int((g == 1).sum()), "n_runs_neg": int((g == 0).sum()),
           "n_items_pos": len(set(items[g == 1].tolist())), "n_items_neg": len(set(items[g == 0].tolist())),
           "n_items_both": len(both_items), "lens_layers": layers, "probe_layers": pl,
           "cot_probe_layers": d["meta"]["cot_layers"], "counts": d["counts"],
           "arms": sorted({r["arm"] for r in runs}),
           "monitor_rate_pos": float(S["monitor|text|flag"][g == 1].mean()),
           "monitor_rate_neg": float(S["monitor|text|flag"][g == 0].mean()),
           "median_completion_pos": float(np.median(d["completion_len"][g == 1])),
           "median_completion_neg": float(np.median(d["completion_len"][g == 0])),
           "scores": {}, "within_item": {}, "per_layer": {}}
    for k, s in S.items():
        res["scores"][k] = boot(s)
        res["within_item"][k] = boot(centre(s, items, both), both)
    # per layer lens AUROCs (point estimates + CI), for the best-layer readout
    for pi, pos in enumerate(POS):
        for li, ln in enumerate(("jlens", "logit")):
            pl_res = {}
            for j, L in enumerate(layers):
                pl_res[str(L)] = boot(lens[:, pi, 0, li, j].astype(float))
            res["per_layer"][f"{ln}|{pos}|D"] = pl_res
    return res


def jdiff(name: str, d: dict, rng) -> dict:
    """Paired item bootstrap of AUROC(J-lens) - AUROC(logit lens) for the fixed lens
    score D at each position (same resamples for both lenses); also the difference of
    the separations |AUROC - 0.5| (direction-free)."""
    runs = d["runs"]
    items = np.array([r["item"] for r in runs])
    g = np.array([r["group"] for r in runs])
    boot = Boot(items, g, rng)
    out = {}
    for pi, pos in enumerate(POS):
        sj = np.nanmean(d["lens"][:, pi, 0, 0, :], -1)
        sl = np.nanmean(d["lens"][:, pi, 0, 1, :], -1)
        ok = np.isfinite(sj) & np.isfinite(sl)
        if not ok.any():
            continue
        aj = auroc(sj[ok & (g == 1)], sj[ok & (g == 0)])
        al = auroc(sl[ok & (g == 1)], sl[ok & (g == 0)])
        bs, bf = [], []
        for smp in boot.samples:
            smp = smp[ok[smp]]
            gg = g[smp]
            if (gg == 1).any() and (gg == 0).any():
                bj = auroc(sj[smp][gg == 1], sj[smp][gg == 0])
                bl = auroc(sl[smp][gg == 1], sl[smp][gg == 0])
                bs.append(bj - bl)
                bf.append(abs(bj - 0.5) - abs(bl - 0.5))
        out[pos] = {"jlens": round(aj, 4), "logit": round(al, 4),
                    "diff": round(aj - al, 4), "lo": round(float(np.percentile(bs, 2.5)), 4),
                    "hi": round(float(np.percentile(bs, 97.5)), 4),
                    "sep_diff": round(abs(aj - 0.5) - abs(al - 0.5), 4),
                    "sep_lo": round(float(np.percentile(bf, 2.5)), 4), "sep_hi": round(float(np.percentile(bf, 97.5)), 4)}
    return out


def run_jdiff(fdir: Path, name: str) -> dict:
    return jdiff(name, load_case(fdir, CASES[name]), np.random.default_rng(SEED))


def run_case(fdir: Path, name: str) -> dict:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    return analyse(name, load_case(fdir, CASES[name]), np.random.default_rng(SEED))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--jdiff-only", action="store_true", help="only the paired J-lens minus logit-lens AUROC differences")
    args = ap.parse_args()
    t0 = time.time()
    fdir = Path(args.features)
    out = {"exploratory": True, "n_boot": N_BOOT, "seed": SEED, "cases": {}}
    names = [n for n in args.cases.split(",") if all((fdir / f).exists() for f in CASES[n])]
    print("cases:", names, flush=True)
    from multiprocessing import Pool
    if args.jdiff_only:
        with Pool(len(names)) as pool:
            done = pool.starmap(run_jdiff, [(fdir, n) for n in names])
        out["jdiff"] = dict(zip(names, done))
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out["jdiff"]), flush=True)
        return
    with Pool(len(names)) as pool:
        done = pool.starmap(run_case, [(fdir, n) for n in names])
    for name, c in zip(names, done):
        out["cases"][name] = c
        print(f"{name}: {c['n_runs_pos']}/{c['n_runs_neg']} runs, {c['n_items_pos']}/{c['n_items_neg']} items", flush=True)
        for k, v in c["scores"].items():
            if v.get("auroc") is not None:
                print(f"   {k:32s} {v['auroc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]   within {c['within_item'][k].get('auroc')}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
