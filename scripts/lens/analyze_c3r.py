"""C3-R (docs/preregistration.md "C3-R"): fresh replication of the planted-claim J-lens positive; primary test =
within-question AUROC of the J-lens S (pairs of kept-B runs from the same question, question bootstrap, within-question
permutation). Reuses the R-C3 loader/score (analyze_c3_repeat.load) and the F2 difficulty analysis (firming_f2.py).

uv run --project envs/lens python scripts/lens/analyze_c3r.py --batches 1 2 3 4 --out data/explore/c3r/c3r.json
Writes <out> (full, with per-run scores: stays in data/) and <out stem>_summary.json (derived numbers, trackable).
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_c3_repeat as A  # noqa: E402
from analyze_main import auroc_boot  # noqa: E402

R = Path(__file__).resolve().parents[2]
D = R / "data"
NB, NP = 2000, 10000


def U(p, n):
    if not len(p) or not len(n):
        return 0.0, 0
    d = np.asarray(p)[:, None] - np.asarray(n)[None, :]
    return float((d > 0).sum() + 0.5 * (d == 0).sum()), d.size


def within(rows, key):
    by = defaultdict(lambda: ([], []))
    for r in rows:
        v = key(r)
        if v is None or not np.isfinite(v):
            continue
        by[r["item"]][0 if r["pos"] else 1].append(v)
    u = n = 0
    for p, q in by.values():
        if p and q:
            uu, nn = U(p, q); u += uu; n += nn
    return u / n if n else float("nan")


def ci(v):
    v = [x for x in v if np.isfinite(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, required=True)
    ap.add_argument("--items", default="data/explore/c3r/items_with_B.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(R / a.items)]
    known = {d["id"]: d["acc"]["A"] for d in items}
    old = {json.loads(l)["id"] for l in open(D / "explore/known_items.jsonl")}
    runs, counts, ctrl, labels = [], {}, {}, []
    for b in a.batches:
        A.SOURCES[f"c3r{b}"] = (f"readouts/c3r{b}_keptB", f"explore/w2/prefill_c3r{b}/labels.jsonl",
                                f"explore/w2/forks_reelicit_c3r{b}/forks.jsonl")
        r, c = A.load(f"c3r{b}", known)
        for x in r:
            x["source"] = "new"
        runs += r; counts[f"batch{b}"] = c
        labels += A.jsonl(D / f"explore/w2/prefill_c3r{b}/labels.jsonl")
        for f in A.jsonl(D / f"explore/w2/forks_reelicit_c3r{b}/forks.jsonl"):
            if f.get("kind") == "control":
                ctrl.setdefault(f["question_id"], f)
    assert not ({r["item"] for r in runs} & old), "run on an old item"
    rng = np.random.default_rng(0)
    nq = len({r["item"] for r in runs})
    pos = [r for r in runs if r["pos"]]; neg = [r for r in runs if not r["pos"]]
    # F2-style inputs
    f2_in = {"known": [{"id": d["id"], "n": d["n"], "n_correct": d["n_correct"]} for d in items],
             "control": [{"batch": "", "question_id": q, "argmax": c["argmax"], "correct_letter": c["correct_letter"],
                          "p_norm": c.get("p_norm"), "p_letters": c.get("p_letters")} for q, c in ctrl.items()],
             "labels": [{"question_id": l["question_id"], "outcome": l["outcome"]} for l in labels]}
    out_p = Path(a.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    (out_p.parent / "f2_in.json").write_text(json.dumps(f2_in))
    (out_p.parent / "c3r_runs.json").write_text(json.dumps({"runs": [{k: r[k] for k in ("run_id", "source", "item", "S", "cot_len", "reel", "pos")} for r in runs]}))
    subprocess.run([sys.executable, str(R / "scripts/lens/firming_f2.py"), "--inp", str(out_p.parent / "f2_in.json"),
                    "--runs", str(out_p.parent / "c3r_runs.json"), "--out", str(out_p.parent / "c3r_f2.json")],
                   check=True, stdout=subprocess.DEVNULL)
    f2 = json.load(open(out_p.parent / "c3r_f2.json"))
    dw = f2["d_within"]
    # within-question for length (text baseline) + pooled with R-C3
    both = sorted({r["item"] for r in runs if r["pos"]} & {r["item"] for r in neg})
    bq = {q: [r for r in runs if r["item"] == q] for q in both}
    lenv = []
    for _ in range(NB):
        pick = rng.choice(len(both), len(both))
        rr = [dict(x, item=i) for i, j in enumerate(pick) for x in bq[both[j]]]
        lenv.append(within(rr, lambda r: float(r["cot_len"]) if r["cot_len"] is not None else None))
    within_len = within(runs, lambda r: float(r["cot_len"]) if r["cot_len"] is not None else None)
    per_q = []
    for q in both:
        p = [r["S"]["jlens"] for r in bq[q] if r["pos"]]; n = [r["S"]["jlens"] for r in bq[q] if not r["pos"]]
        u, m = U(p, n); per_q.append(u / m)
    from math import comb
    k = int(sum(x > 0.5 for x in per_q)); nn_ = int(sum(x != 0.5 for x in per_q))
    sign_p = sum(comb(nn_, i) for i in range(k, nn_ + 1)) / 2 ** nn_ if nn_ else None
    # run-level (R-C3 form)
    runlevel = A.test(runs, rng)
    # pooled with R-C3 original-repeat runs (exploratory)
    prev = json.load(open(R / "docs/experiments/replicate-leads/data/c3_lead_repeat.json"))["runs"]
    pool = runs + prev
    bothp = sorted({r["item"] for r in pool if r["pos"]} & {r["item"] for r in pool if not r["pos"]})
    bqp = {q: [r for r in pool if r["item"] == q] for q in bothp}
    pj = []
    for _ in range(500):
        pick = rng.choice(len(bothp), len(bothp))
        rr = [dict(x, item=i) for i, j in enumerate(pick) for x in bqp[bothp[j]]]
        pj.append(within(rr, lambda r: r["S"]["jlens"]))
    pooled = {"questions_both": len(bothp), "jlens_within": within(pool, lambda r: r["S"]["jlens"]), "ci95_500boot": ci(pj),
              "logit_within": within(pool, lambda r: r["S"]["logit"])}
    nboth = dw["n_questions_both"]
    j = dw["jlens"]
    verdict = ("underpowered (< 30 questions with both outcomes)" if nboth < 30
               else ("pass" if (j["ci95"][0] > 0.5 and j["perm_p_onesided"] < 0.05) else "fail"))
    summ = {
        "prereg": "docs/preregistration.md C3-R", "batches": a.batches, "layers": A.LAYERS,
        "n_questions_with_kept_B": nq, "n_kept_B": len(runs), "n_pos": len(pos), "n_neg": len(neg),
        "n_pos_questions": len({r["item"] for r in pos}), "n_neg_questions": len({r["item"] for r in neg}),
        "n_questions_both_outcomes": nboth, "n_runs_in_both": dw["n_runs"],
        "PRIMARY_within_question_jlens": j, "VERDICT": verdict,
        "within_question_logit": dw["logit"], "jlens_minus_logit_within": dw["jlens_minus_logit"],
        "within_question_length": {"auroc": within_len, "ci95": ci(lenv)},
        "per_question_mean_auroc_jlens": float(np.mean(per_q)) if per_q else None,
        "questions_expected_direction": k, "questions_not_tied": nn_, "sign_test_p_onesided": sign_p,
        "run_level": runlevel,
        "difficulty_a_alone": f2["a_alone"], "difficulty_b_logistic": f2["b_logistic_S_coef_logodds_per_SD"],
        "difficulty_c_matched": f2["c_matched"], "control_wrong_share": f2["control_wrong_share"],
        "miss20_dist": f2["miss20_dist"], "pooled_with_RC3_exploratory": pooled, "counts": counts,
    }
    out_p.write_text(json.dumps(summ, indent=1))
    (out_p.parent / (out_p.stem + "_summary.json")).write_text(json.dumps({k: v for k, v in summ.items()}, indent=1))
    print(json.dumps({k: summ[k] for k in ("n_kept_B", "n_questions_both_outcomes", "PRIMARY_within_question_jlens", "VERDICT",
                                            "within_question_logit", "jlens_minus_logit_within", "within_question_length")}, indent=1))


if __name__ == "__main__":
    main()
