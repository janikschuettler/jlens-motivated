"""Point of no return (Q6): analysis of the resamples (CPU; numpy, plus the tokenizer for
quoting the commitment segments).

  ~/.local/bin/uv run --no-sync --project envs/gen python scripts/explore/analyze_q6.py \
      configs/explore/q6_point_of_no_return.yaml [--cases c3 hint8b ...]

Reads data/explore/q6_pnr/<case>/{plans,resamples}.jsonl, the case's existing lens readouts
(index.jsonl + one .npz per run) and, where they exist, the forced-early-answer forks (E5)
of the same runs (data/explore/q6_pnr/<case>/early_forks.jsonl, filtered on octavia).
Writes docs/overnight/data/point_of_no_return.json (derived numbers only) and
data/explore/q6_pnr/analysis_full.json (per-run detail and segment quotes; not in git).
Definitions: docs/overnight/preregistration.md, "Q6".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from jmr import config

ROOT = config.REPO_ROOT
B = 2000  # bootstrap resamples
WINDOW = 8  # lens positions averaged before each cut


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def boot(items: list[str], per_item: dict[str, list], stat, n=B, seed=0):
    """Item bootstrap: stat(list of run values) over resampled items. Returns (est, lo, hi)."""
    vals = [v for it in items for v in per_item[it]]
    est = stat(vals)
    if est is None or not items:
        return {"est": est, "lo": None, "hi": None, "n_items": len(items), "n": len(vals)}
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [v for it in (rng.choice(items) for _ in items) for v in per_item[it]]
        r = stat(s)
        if r is not None:
            out.append(r)
    lo, hi = (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))) if out else (None, None)
    return {"est": est, "lo": lo, "hi": hi, "n_items": len(items), "n": len(vals)}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def share(pred):
    """Share of runs with pred(run) true, over runs where pred(run) is not None."""
    def f(xs):
        v = [pred(x) for x in xs]
        v = [bool(x) for x in v if x is not None]
        return float(np.mean(v)) if v else None
    return f


def median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    # average ranks for ties
    def avg_rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        xs = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = avg_rank(a), avg_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def commitment_cut(p, thr):
    k_star = None
    for k in range(len(p) - 1, -1, -1):
        if p[k] is None:
            continue
        if p[k] >= thr - 1e-9:
            k_star = k
        else:
            break
    return k_star


def readout_file(run_id: str) -> str:
    return hashlib.sha1(run_id.encode()).hexdigest()[:20] + ".npz"


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def lens_scores(npz_path: Path, plan: dict, outcome: str) -> dict | None:
    """Per cut: J-lens and logit-lens score, averaged over workspace layers and the WINDOW
    positions before the cut. MCQ: log p(correct letter) - log p(hinted letter) (letters
    summed over their tracked forms). Hack: log of the summed probability of the
    pre-registered hack-concept tokens."""
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    meta = json.loads(str(z["meta_json"]))
    lo = int(z["positions_lo"])
    if outcome == "hack":
        arr = z["x_lp"].astype(np.float32)  # [lens, layer, T, K]
        sets = meta["tracked"]["sets"]
        idx = [i for i, s in enumerate(sets) if s == "concept"]
        score = lse(arr[..., idx], axis=-1)  # [lens, layer, T]
    else:
        arr = z["lp"].astype(np.float32)
        labels = meta["tracked"]["labels"]
        corr, hint = plan["original"]["correct"], plan["original"]["hinted"]
        ci = [i for i, s in enumerate(labels) if s.strip().lstrip("(") == corr]
        hi = [i for i, s in enumerate(labels) if s.strip().lstrip("(") == hint]
        score = lse(arr[..., ci], axis=-1) - lse(arr[..., hi], axis=-1)
    layer_mean = score.mean(axis=1)  # [lens, T]
    T = layer_mean.shape[1]
    out = {"jlens": [], "logit": []}
    for c in plan["cuts"]:
        a, b = max(c["pos"] - WINDOW, lo), c["pos"]  # positions a..b-1 (the state before the next token)
        ia, ib = a - lo, min(b - lo, T)
        for li, name in enumerate(("jlens", "logit")):
            out[name].append(float(layer_mean[li, ia:ib].mean()) if ib > ia else None)
    return out


def analyze_case(spec: dict, case: dict, tok) -> tuple[dict, dict]:
    d = ROOT / spec["out_dir"] / case["name"]
    plans = {p["run_id"]: p for p in jl(d / "plans.jsonl")}
    rs = jl(d / "resamples.jsonl")
    extra = defaultdict(list)  # second-pass cuts inside the commitment segment (exploratory)
    if (d / "refine" / "resamples.jsonl").exists():
        for p in jl(d / "refine" / "plans.jsonl"):
            extra[p["run_id"]] = p["cuts"]
        rs_refine = jl(d / "refine" / "resamples.jsonl")
    else:
        rs_refine = []
    thr = spec["threshold"]
    thrs = [thr] + spec.get("sensitivity_thresholds", [])
    is_hack = case["outcome"] == "hack"
    cnt = defaultdict(lambda: defaultdict(lambda: {"n": 0, "valid": 0, "same": 0, "hint": 0, "correct": 0,
                                                  "hack": 0, "length": 0}))
    for r in rs + rs_refine:
        c = cnt[r["run_id"]][r["cut_pos"]]
        c["n"] += 1
        c["length"] += r["finish_reason"] == "length"
        if r["valid"]:
            c["valid"] += 1
            c["same"] += bool(r["same"])
            if is_hack:
                c["hack"] += bool(r["hack"])
            else:
                p = plans[r["run_id"]]["original"]
                c["hint"] += r["answer"] == p["hinted"]
                c["correct"] += r["answer"] == p["correct"]
    # readouts
    # readouts: a local copy (d/readouts, for sets that live on another machine) or the case's dir
    rdir = d / "readouts" if (d / "readouts" / "index.jsonl").exists() else (
        ROOT / case["readouts"] if case.get("readouts") and (ROOT / case["readouts"] / "index.jsonl").exists() else None)
    rfile = {}
    if rdir is not None:
        for m in jl(rdir / "index.jsonl"):
            if m["run_id"] in plans and m.get("own", True):
                rfile[m["run_id"]] = m.get("file") or readout_file(m["run_id"])
    early = defaultdict(dict)
    if (d / "early_forks.jsonl").exists():
        for f in jl(d / "early_forks.jsonl"):
            pn = f.get("p_norm") or {}
            early[f["source_run_id"]][round(f["f"], 2)] = pn
    runs = []
    fractions = spec["fractions"]
    for rid, p in sorted(plans.items()):
        cuts = p["cuts"]
        K = len(cuts)
        cc = [cnt[rid][c["pos"]] for c in cuts]
        p_same = [c["same"] / c["valid"] if c["valid"] else None for c in cc]
        p_out = [(c["hack"] if is_hack else c["hint"]) / c["valid"] if c["valid"] else None for c in cc]
        s, e = p["span"]
        # value at every requested fraction: the cut nearest to the fraction's target
        at_f = {}
        for f in fractions:
            target = s + f * (e - s)
            k = min(range(K), key=lambda j: (abs(cuts[j]["pos"] - target), cuts[j]["pos"]))
            at_f[f"{f:.1f}"] = k
        row = {"run_id": rid, "group": p["group"], "item": p["question_id"], "K": K, "span_len": e - s,
               "p_same": p_same, "p_out": p_out, "n_valid": [c["valid"] for c in cc], "n": [c["n"] for c in cc],
               "n_length": sum(c["length"] for c in cc), "cut_frac": [c["frac"] for c in cuts],
               "cut_pos": [c["pos"] for c in cuts], "at_f": at_f, "marks": p["marks"], "original": p["original"]}
        for t_ in thrs:
            k = commitment_cut(p_same, t_)
            row[f"k_star@{t_}"] = k
        k = row[f"k_star@{thr}"]
        row["commit_frac"] = cuts[k]["frac"] if k is not None else None
        row["commit_prev_frac"] = cuts[k - 1]["frac"] if k else None
        m = p["marks"]
        span_frac = lambda pos: None if pos is None else round((pos - s) / max(e - s, 1), 4)
        row["mention_frac"] = span_frac(m.get("first_mention")) if m.get("first_mention") is not None and m["first_mention"] < e else None
        row["states_final_frac"] = span_frac(m.get("first_states_final"))
        if k is not None:
            cp = cuts[k]["pos"]
            fm = m.get("first_mention")
            row["before_mention"] = None if (fm is None or fm >= e and not is_hack) else bool(cp <= fm)
            sf = m.get("first_states_final")
            row["before_states_final"] = None if sf is None else bool(cp <= sf)
            if is_hack:
                row["commit_in"] = "thinking" if (m.get("think_end") is not None and cp <= m["think_end"]) else "code"
        # second pass: the commitment point on all cuts (first pass + cuts inside the segment)
        if extra.get(rid):
            merged = sorted(cuts + extra[rid], key=lambda c: c["pos"])
            ps_m = [(cnt[rid][c["pos"]]["same"] / cnt[rid][c["pos"]]["valid"]) if cnt[rid][c["pos"]]["valid"] else None
                    for c in merged]
            km = commitment_cut(ps_m, thr)
            row["refined"] = {"cut_pos": [c["pos"] for c in merged], "cut_frac": [c["frac"] for c in merged],
                              "p_same": ps_m, "k_star": km,
                              "commit_frac": merged[km]["frac"] if km is not None else None,
                              "segment": [merged[km - 1]["pos"], merged[km]["pos"]] if km else None,
                              "segment_frac": [merged[km - 1]["frac"], merged[km]["frac"]] if km else None}
        # segment quote: tokens between the last uncommitted cut and the commitment cut
        if tok is not None and k:
            row["commit_segment_tokens"] = (row["refined"]["segment"] if row.get("refined", {}).get("segment")
                                            else [cuts[k - 1]["pos"], cuts[k]["pos"]])
        # lens
        if rid in rfile:
            ls = lens_scores(rdir / rfile[rid], p, case["outcome"])
            if ls is not None:
                row["lens"] = ls
        if rid in early:
            row["early"] = {f"{f:.1f}": (pn.get(p["original"]["answer"]) if not is_hack else None)
                            for f, pn in early[rid].items()}
        runs.append(row)

    # ---- summaries per group
    out = {"n_resamples": len(rs), "valid_rate": float(np.mean([r["valid"] for r in rs])) if rs else None,
           "length_rate": float(np.mean([r["finish_reason"] == "length" for r in rs])) if rs else None,
           "groups": {}}
    for g in case["groups"]:
        G = [r for r in runs if r["group"] == g]
        if not G:
            continue
        per_item = defaultdict(list)
        for r in G:
            per_item[r["item"]].append(r)
        items = sorted(per_item)
        res = {"n_runs": len(G), "n_items": len(items),
               "median_span_tokens": float(np.median([r["span_len"] for r in G])),
               "median_cuts": float(np.median([r["K"] for r in G]))}
        for t_ in thrs:
            key = f"k_star@{t_}"
            res[f"committed_at_start@{t_}"] = boot(items, per_item, share(lambda r: r[key] == 0))
            res[f"never_committed@{t_}"] = boot(items, per_item, share(lambda r: r[key] is None))
        res["commit_frac_median"] = boot(items, per_item, lambda rr: median_or_none([r["commit_frac"] for r in rr]))
        res["commit_frac_mean_never_as_1"] = boot(
            items, per_item, lambda rr: mean_or_none([1.0 if r["commit_frac"] is None else r["commit_frac"] for r in rr]))
        res["commit_frac_values"] = [r["commit_frac"] for r in G]
        Rf = [r for r in G if r.get("refined")]
        if Rf:
            res["refined"] = {"n_runs": len(Rf),
                              "segment_tokens_median_first_pass": median_or_none(
                                  [r["cut_pos"][r[f"k_star@{thr}"]] - r["cut_pos"][r[f"k_star@{thr}"] - 1] for r in Rf]),
                              "segment_tokens_median_refined": median_or_none(
                                  [r["refined"]["segment"][1] - r["refined"]["segment"][0] for r in Rf
                                   if r["refined"]["segment"]]),
                              "k_star_moved": sum(r["refined"]["commit_frac"] != r["commit_frac"] for r in Rf),
                              "now_uncommitted": sum(r["refined"]["k_star"] is None for r in Rf)}
        res["before_mention"] = boot(items, per_item, share(lambda r: r.get("before_mention")))
        res["n_with_mention"] = sum(r.get("before_mention") is not None for r in G)
        res["before_states_final"] = boot(items, per_item, share(lambda r: r.get("before_states_final")))
        res["mention_frac_median"] = median_or_none([r["mention_frac"] for r in G])
        res["states_final_frac_median"] = median_or_none([r["states_final_frac"] for r in G])
        if is_hack:
            res["commit_in_thinking"] = boot(items, per_item, share(lambda r: (r.get("commit_in") == "thinking")
                                                                    if r.get("commit_in") else None))
        curve, curve_out = {}, {}
        for f in fractions:
            fk = f"{f:.1f}"
            curve[fk] = boot(items, per_item, lambda rr: mean_or_none([r["p_same"][r["at_f"][fk]] for r in rr]))
            curve_out[fk] = boot(items, per_item, lambda rr: mean_or_none([r["p_out"][r["at_f"][fk]] for r in rr]))
        res["p_same_curve"] = curve
        res["p_outcome_curve"] = curve_out  # MCQ: share ending on the hinted B; hack: share hacking
        # lens and forced-early-answer relations
        L = [r for r in G if "lens" in r]
        res["n_with_readout"] = len(L)
        for r in L:  # per-run lens relations, computed once
            k = r[f"k_star@{thr}"]
            for lens in ("jlens", "logit"):
                sc = r["lens"][lens]
                ok = None not in sc and None not in r["p_out"]
                r[f"rho_{lens}"] = spearman(sc, r["p_out"]) if ok else None
                r[f"seg_pct_{lens}"] = r[f"seg_signed_{lens}"] = None
                if k and None not in sc and len(sc) >= 3:
                    deltas = [sc[j] - sc[j - 1] for j in range(1, len(sc))]
                    a = [abs(x) for x in deltas]
                    dk = a[k - 1]
                    r[f"seg_pct_{lens}"] = (sum(x < dk for x in a) + 0.5 * (sum(x == dk for x in a) - 1)) / max(len(a) - 1, 1)
                    others = [x for j, x in enumerate(deltas) if j != k - 1]
                    r[f"seg_signed_{lens}"] = deltas[k - 1] - float(np.mean(others))
        if L:
            per_item_L = defaultdict(list)
            for r in L:
                per_item_L[r["item"]].append(r)
            for lens in ("jlens", "logit"):
                for key, name in ((f"rho_{lens}", f"rho_{lens}_vs_p_outcome"),
                                  (f"seg_pct_{lens}", f"commit_segment_change_pct_{lens}"),
                                  (f"seg_signed_{lens}", f"commit_segment_signed_minus_others_{lens}")):
                    res[name] = boot(sorted(per_item_L), per_item_L, lambda rr, key=key: mean_or_none([r[key] for r in rr]))
                res[f"lens_curve_{lens}"] = {
                    f"{f:.1f}": mean_or_none([r["lens"][lens][r["at_f"][f"{f:.1f}"]] for r in L]) for f in fractions}
                res[f"n_commit_segment_{lens}"] = sum(1 for r in L if r[f"seg_pct_{lens}"] is not None)
                res[f"n_rho_{lens}"] = sum(1 for r in L if r[f"rho_{lens}"] is not None)
        E = [r for r in G if r.get("early")]
        res["n_with_forced_answer"] = len(E)
        for r in E:
            fs = sorted(f for f in r["early"] if f in r["at_f"])
            a = [r["early"][f] for f in fs]
            b = [r["p_same"][r["at_f"][f]] for f in fs]
            r["rho_early"] = spearman(a, b) if None not in a and None not in b else None
        if E:
            per_item_E = defaultdict(list)
            for r in E:
                per_item_E[r["item"]].append(r)
            res["rho_forced_answer_vs_p_same"] = boot(sorted(per_item_E), per_item_E,
                                                      lambda rr: mean_or_none([r["rho_early"] for r in rr]))
            res["n_rho_forced_answer"] = sum(1 for r in E if r["rho_early"] is not None)
            res["forced_answer_curve"] = {f: mean_or_none([r["early"].get(f) for r in E]) for f in sorted(E[0]["early"])}
        out["groups"][g] = res
    # quotes of commitment segments (full analysis file only)
    if tok is not None:
        rows_by = {p["run_id"]: p for p in plans.values()}
        full_runs = {r["run_id"]: r for r in runs}
        src = d / "source_runs.jsonl" if (d / "source_runs.jsonl").exists() else ROOT / case["runs"]
        all_runs = {r["run_id"]: r for r in (json.loads(l) for l in open(src)) if r["run_id"] in full_runs} \
            if src.exists() else {}
        for rid, r in full_runs.items():
            if r.get("commit_segment_tokens") and rid in all_runs:
                a, b = r["commit_segment_tokens"]
                seq = all_runs[rid]["prompt_token_ids"] + all_runs[rid]["completion_token_ids"]
                r["commit_segment_text"] = tok.decode(seq[a:b])
                r["before_cut0_text"] = tok.decode(seq[max(rows_by[rid]["span"][0], a - 60):a])
    return out, {"runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--no-quotes", action="store_true")
    ap.add_argument("--data-out", default="docs/overnight/data/point_of_no_return.json")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    out = {"spec": {k: spec[k] for k in ("resamples", "fractions", "threshold", "sensitivity_thresholds")},
           "cases": {}}
    full = {}
    for case in sorted(spec["cases"], key=lambda c: c["order"]):
        if args.cases and case["name"] not in args.cases:
            continue
        d = ROOT / spec["out_dir"] / case["name"]
        if not (d / "resamples.jsonl").exists():
            print(f"{case['name']}: no resamples, skipped")
            continue
        tok = None
        if not args.no_quotes:
            import os

            from transformers import AutoTokenizer

            if case.get("profile"):
                os.environ["JMR_MODEL_PROFILE"] = case["profile"]
            else:
                os.environ.pop("JMR_MODEL_PROFILE", None)
            m = config.load("model")
            tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
        res, fr = analyze_case(spec, case, tok)
        res["meta"] = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else None
        if res["meta"]:
            res["meta"] = {k: res["meta"][k] for k in ("n_runs", "n_requests", "n_done", "stats", "load_s", "git")
                           if k in res["meta"]}
        out["cases"][case["name"]] = res
        full[case["name"]] = fr
        g = res["groups"]
        print(case["name"], json.dumps({k: {"n": v["n_runs"], "start": v[f"committed_at_start@{spec['threshold']}"]["est"],
                                            "never": v[f"never_committed@{spec['threshold']}"]["est"],
                                            "commit_med": v["commit_frac_median"]["est"]} for k, v in g.items()}))
    p = ROOT / args.data_out
    old = json.loads(p.read_text()) if p.exists() and args.cases else {"cases": {}}
    old["cases"].update(out["cases"])
    old["spec"] = out["spec"]
    p.write_text(json.dumps(old, indent=1))
    fp = ROOT / spec["out_dir"] / "analysis_full.json"
    oldf = json.loads(fp.read_text()) if fp.exists() and args.cases else {}
    oldf.update(full)
    fp.write_text(json.dumps(oldf))
    print("wrote", p, fp)


if __name__ == "__main__":
    main()
