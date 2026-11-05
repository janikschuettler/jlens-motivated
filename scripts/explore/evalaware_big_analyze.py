"""Analysis of the Qwen3.6-27B eval-awareness smoke test (numpy only; exploratory).

  python scripts/explore/evalaware_big_analyze.py --dir data/evalaware_big/readout \
      --runs data/evalaware_big/api_runs.jsonl --out docs/experiments/evalaware-big-smoke/data/evalaware_big_smoke.json

Reads the compact readout of scripts/lens/evalaware_big.py. The score is the replication's:
mean J-lens log-prob of " fictional" and " simulation" minus the mean over 100 control nouns,
averaged over the workspace layers (blocks 25, 27, ..., 57 of 64 for this model).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/explore"))
from evalaware_models_analyze import TEST_ME  # noqa: E402

B = 2000


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auroc(rows, key, label, cluster, seed=0):
    """rows: dicts; AUROC of key for label True vs False; bootstrap resamples clusters."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r[cluster]].append(r)
    cl = list(by)
    est = auroc([r[key] for r in rows if r[label]], [r[key] for r in rows if not r[label]])
    bs = []
    for _ in range(B):
        samp = [r for c in (rng.choice(cl) for _ in cl) for r in by[c]]
        a = auroc([r[key] for r in samp if r[label]], [r[key] for r in samp if not r[label]])
        if a is not None:
            bs.append(a)
    lo, hi = (np.quantile(bs, [0.025, 0.975]).tolist() if bs else (None, None))
    return {"auroc": None if est is None else round(est, 3), "ci": [round(lo, 3), round(hi, 3)] if bs else None,
            "n_pos": sum(1 for r in rows if r[label]), "n_neg": sum(1 for r in rows if not r[label]),
            "n_clusters": len(cl)}


def boot_mean(vals, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(v.mean()), 3), [round(float(x), 3) for x in np.quantile(bs, [0.025, 0.975])]]


def first_test_me(text):
    starts = [m.start() for p in TEST_ME for m in p.finditer(text or "")]
    return min(starts) if starts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    tracked = meta["tracked"]
    col = {t["text"]: i for i, t in enumerate(tracked)}
    K = len(tracked)
    prim = [col[" fictional"], col[" simulation"]]
    testw = [i for i, t in enumerate(tracked) if "test" in t["groups"]]
    evalset_ids = {t["id"] for t in tracked if "eval_set" in t["groups"]}
    lenses = meta["lenses"]

    def score(lp, cols=prim):  # lp [..., layers, K+1] -> mean over layers
        lp = lp.astype(np.float32)
        return (lp[..., cols].mean(-1) - lp[..., K]).mean(-1)

    out = {"model": meta["model"]["hf_id"], "revision": meta["model"]["revision"],
           "lens": meta["lens"]["jlens"], "lens_info": meta["lens_info"],
           "logit_check_max_absdiff": meta["logit_check"]["max_absdiff"],
           "workspace_layers": meta["workspace_layers"], "kind": "exploratory"}

    # ---- step 1: prompts ----
    if (d / "prompts.npz").exists():
        idx = json.loads((d / "prompts_index.json").read_text())
        lp = np.load(d / "prompts.npz")["lp"]  # [n, lens, layers, pos(2), K+1]
        s1 = {}
        rows = []
        for i, r in enumerate(idx):
            row = {"suite": r["suite"], "eval": r["kind"] == "eval", "cl": f"{r['suite']}/{r['idx']}"}
            for si, ln in enumerate(lenses):
                row[f"{ln}_turn"] = float(score(lp[i, si, :, 0]))
                row[f"{ln}_last"] = float(score(lp[i, si, :, 1]))
                row[f"{ln}_turn_test"] = float(score(lp[i, si, :, 0], testw))
            rows.append(row)
        rows_nb = [r for r in rows if r["suite"] != "blackmail"]
        for k in ("jlens_turn", "logit_turn", "jlens_last", "logit_last", "jlens_turn_test", "logit_turn_test"):
            s1[k] = boot_auroc(rows_nb, k, "eval", "cl")
        # without the two suites whose wording is most eval-like ("being evaluated", sycophancy evals)
        rows_core = [r for r in rows_nb if r["suite"] not in ("syco", "grader_agent")]
        for k in ("jlens_turn", "logit_turn"):
            s1[k + "_without_syco_grader_agent"] = boot_auroc(rows_core, k, "eval", "cl")
        # paired J-lens minus logit-lens AUROC (same prompts)
        rng = random.Random(1)
        diffs = []
        for _ in range(B):
            samp = [rng.choice(rows_nb) for _ in rows_nb]
            ja = auroc([r["jlens_turn"] for r in samp if r["eval"]], [r["jlens_turn"] for r in samp if not r["eval"]])
            la = auroc([r["logit_turn"] for r in samp if r["eval"]], [r["logit_turn"] for r in samp if not r["eval"]])
            diffs.append(ja - la)
        s1["jlens_minus_logit_turn"] = [round(s1["jlens_turn"]["auroc"] - s1["logit_turn"]["auroc"], 3),
                                        [round(float(x), 3) for x in np.quantile(diffs, [0.025, 0.975])]]
        suites = {}
        for s in dict.fromkeys(r["suite"] for r in rows):
            rs = [r for r in rows if r["suite"] == s]
            suites[s] = {"n": len(rs), "kind": "eval" if rs[0]["eval"] else "deployment",
                         "jlens_turn": boot_mean([r["jlens_turn"] for r in rs]) if len(rs) > 1 else
                         [round(rs[0]["jlens_turn"], 3), None],
                         "logit_turn": round(float(np.mean([r["logit_turn"] for r in rs])), 3)}
        dep = [r for r in rows if r["suite"] in ("oasst", "wildchat")]
        dep_means = {s: np.mean([r["jlens_turn"] for r in dep if r["suite"] == s]) for s in ("oasst", "wildchat")}
        n_above = sum(1 for s, v in suites.items() if v["kind"] == "eval" and s != "blackmail"
                      and v["jlens_turn"][0] > max(dep_means.values()))
        s1.update(suites=suites, n_eval_suites_above_both_deploy=n_above, n_prompts=len(rows))
        # by-eye top hits
        tz = np.load(d / "prompts_top.npz")
        from transformers import AutoTokenizer  # noqa: optional, only for decoding
        tok = AutoTokenizer.from_pretrained(meta["model"]["hf_id"], revision=meta["model"]["revision"])
        ctrl_ids = [t for t in json.loads((REPO / "configs/evalaware_tokens_qwen3.6-27b.json").read_text())["tokens"]
                    if t["groups"] == ["control"]][:len(evalset_ids)]
        ctrl_set = {t["id"] for t in ctrl_ids}
        tops = {}
        for key in sorted({k.split("|")[0] for k in tz.files}):
            ids = tz[f"{key}|ids"]
            tid = tz[f"{key}|top_ids"][:, :, :, :20]  # [lens, layers, pos, 20]
            res = {}
            for si, ln in enumerate(lenses):
                hit = np.isin(tid[si], list(evalset_ids)).any(axis=(0, 2))
                chit = np.isin(tid[si], list(ctrl_set)).any(axis=(0, 2))
                res[ln] = {"frac_eval": round(float(hit.mean()), 4), "frac_control": round(float(chit.mean()), 4)}
            # J-lens examples: up to 12 positions with eval tokens in the top 20
            ex = []
            for p in np.where(np.isin(tid[0], list(evalset_ids)).any(axis=(0, 2)))[0][:400]:
                words = Counter(tok.decode([int(t)]) for t in tid[0][:, p].ravel() if int(t) in evalset_ids)
                ex.append({"pos": int(p), "token": tok.decode([int(ids[p])]),
                           "context": tok.decode([int(x) for x in ids[max(0, p - 8):p + 1]]),
                           "eval_tokens": dict(words.most_common(4))})
            res["n_positions"] = int(len(ids))
            res["examples"] = ex[:: max(1, len(ex) // 12)][:12]
            tops[key] = res
        s1["top_hits"] = tops
        out["step1_prompts"] = s1

    # ---- step 2: API runs ----
    if (d / "runs.npz").exists():
        ridx = json.loads((d / "runs_index.json").read_text())
        z = np.load(d / "runs.npz")
        texts = {json.loads(l)["run_id"]: json.loads(l) for l in open(a.runs)}
        rows = []
        for r in ridx:
            run = texts[r["run_id"]]
            reason = (run["reasoning"] or "").strip()
            fr = first_test_me(reason)
            fc = first_test_me(run["content"])
            lpz = z[r["key"] + "_lp"]  # [lens, layers, pos, K+1]
            tf = z[r["key"] + "_tf"]
            pc = r["pos_char"]
            cot_idx = [i for i in range(2, len(r["positions"]) - 1)]
            n_cot = len(cot_idx)
            rows.append({"run_id": r["run_id"], "arm": r["arm"], "item": r["item_id"], "cl": r["item_id"],
                         "took_bug": run["answer"] == run["hinted"], "says_test": fr is not None or fc is not None,
                         "says_test_reasoning": fr is not None, "first_char": fr, "reason_len": len(reason),
                         "r0": r["reasoning_char0"], "pc": pc, "cot_idx": cot_idx, "lp": lpz, "tf": tf,
                         "n_cot": n_cot, "completion_len": r["completion_len"]})
        # fidelity
        mlp = np.array([x["tf"].mean() for x in rows])
        low = np.array([(x["tf"] < -10).mean() for x in rows])
        med, mad = float(np.median(mlp)), float(np.median(np.abs(mlp - np.median(mlp))))
        flag = [x["run_id"] for x, v in zip(rows, mlp) if v < med - 4 * max(mad, 0.02)]
        fid = {"mean_token_logprob": {"median": round(med, 3), "min": round(float(mlp.min()), 3),
                                      "max": round(float(mlp.max()), 3), "q10": round(float(np.quantile(mlp, .1)), 3),
                                      "q90": round(float(np.quantile(mlp, .9)), 3)},
               "frac_tokens_below_-10": {"median": round(float(np.median(low)), 5), "max": round(float(low.max()), 5)},
               "median_abs_dev": round(mad, 3), "flagged_off_policy": flag, "rule": "mean log-prob < median - 4 MAD"}
        # pre-verbalization windows
        f_first = [x["first_char"] / max(1, x["reason_len"]) for x in rows if x["says_test_reasoning"]]
        f_med = float(np.median(f_first)) if f_first else 1.0
        f_min = float(np.min(f_first)) if f_first else 1.0

        def pre_score(x, cutoff_frac=None, lens=0, cols=prim):
            keep = []
            for i in x["cot_idx"]:
                c = x["pc"][i] - x["r0"]  # char offset within the stripped reasoning
                if cutoff_frac is not None:
                    ok = c < cutoff_frac * x["reason_len"]
                elif x["says_test_reasoning"]:
                    ok = c < x["first_char"] - 1
                else:
                    ok = True
                if ok:
                    keep.append(i)
            if not keep:
                return None, 0
            return float(score(x["lp"][lens][:, keep].transpose(1, 0, 2), cols).mean()), len(keep)

        s2 = {"n_runs": len(rows), "n_items": len({x["item"] for x in rows}),
              "arms": dict(Counter(x["arm"] for x in rows)),
              "says_test": sum(x["says_test"] for x in rows),
              "says_test_in_reasoning": sum(x["says_test_reasoning"] for x in rows),
              "took_bug": sum(x["took_bug"] for x in rows),
              "took_bug_given_test": [sum(x["took_bug"] for x in rows if x["says_test"]), sum(x["says_test"] for x in rows)],
              "took_bug_given_no_test": [sum(x["took_bug"] for x in rows if not x["says_test"]),
                                         sum(not x["says_test"] for x in rows)],
              "first_verbalization_frac": {"median": round(f_med, 3), "min": round(f_min, 3),
                                           "q25": round(float(np.quantile(f_first, .25)), 3) if f_first else None},
              "fidelity": fid}
        comps = {}
        for si, ln in enumerate(lenses):
            for name, cutoff, cols in (("own_cutoff", None, prim), ("matched_median_frac", "med", prim),
                                       ("early_min_frac", "min", prim), ("own_cutoff_testwords", None, testw)):
                rr = []
                for x in rows:
                    cf = None
                    if cutoff == "med":  # verbalizers: own cutoff; others: the median verbalization point
                        cf = None if x["says_test_reasoning"] else f_med
                    elif cutoff == "min":
                        cf = f_min
                    v, n = pre_score(x, cf, si, cols)
                    if v is not None:
                        rr.append({"v": v, "lab": x["says_test_reasoning"], "cl": x["cl"]})
                comps[f"{ln}_{name}"] = boot_auroc(rr, "v", "lab", "cl")
            # prompt-only score of these runs (item x arm level): turn start
            rr = [{"v": float(score(x["lp"][si][:, 0])), "lab": x["says_test_reasoning"], "cl": x["cl"]} for x in rows]
            comps[f"{ln}_prompt_turn_start"] = boot_auroc(rr, "v", "lab", "cl")
            # whole reasoning, no echo control (for reference)
            rr = [{"v": float(score(x["lp"][si][:, x["cot_idx"]].transpose(1, 0, 2)).mean()),
                   "lab": x["says_test_reasoning"], "cl": x["cl"]} for x in rows]
            comps[f"{ln}_whole_reasoning"] = boot_auroc(rr, "v", "lab", "cl")
        # within-arm (same prompt framing) for the J-lens own-cutoff comparison
        for arm in sorted({x["arm"] for x in rows}):
            rr = []
            for x in rows:
                if x["arm"] == arm:
                    v, n = pre_score(x, None, 0)
                    if v is not None:
                        rr.append({"v": v, "lab": x["says_test_reasoning"], "cl": x["cl"]})
            comps[f"jlens_own_cutoff_{arm}"] = boot_auroc(rr, "v", "lab", "cl")
        # items with both labels: within-item difference
        by = defaultdict(lambda: {True: [], False: []})
        for x in rows:
            v, n = pre_score(x, None, 0)
            if v is not None:
                by[(x["item"], x["arm"])][x["says_test_reasoning"]].append(v)
        diffs = [np.mean(g[True]) - np.mean(g[False]) for g in by.values() if g[True] and g[False]]
        s2["within_item_arm_diff_jlens_own_cutoff"] = (boot_mean(diffs) + [len(diffs)]) if len(diffs) > 1 else None
        s2["auroc"] = comps
        s2["mean_scores"] = {
            f"{ln}_{lab}": round(float(np.mean([v for v in [pre_score(x, None, si)[0] for x in rows
                                                            if x["says_test_reasoning"] == (lab == "test")]
                                               if v is not None])), 3)
            for si, ln in enumerate(lenses) for lab in ("test", "no_test")}
        s2["n_pre_positions_median"] = {
            lab: float(np.median([pre_score(x, None, 0)[1] for x in rows if x["says_test_reasoning"] == (lab == "test")]))
            for lab in ("test", "no_test")}
        out["step2_runs"] = s2
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
