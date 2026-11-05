"""Independent recomputation of headline lens numbers from the raw readouts (F3 in
docs/overnight/preregistration.md). Numpy only; deliberately does not import any project
analysis code, so that it is a second implementation, not a re-run.

For each set it writes one JSON line per run (run id, question, group, score per lens) and
prints AUROCs with a 95% question-bootstrap interval, plus two checks of the readout files:
  - position check: at pre_answer the model's own next-token distribution (model_lp) must put
    most of its mass on the letter that was actually written next (if pre_answer were off by
    one, it would not);
  - layer check: the stored layers.

Letters: the columns of the stored tracked log-probs are found by their labels ("X", " X", "(X");
a letter's log-prob is the log-sum-exp over its forms.

Sets (paths relative to --data):
  rc3   planted-claim repeat R-C3: readouts/c3rep_keptB + readouts/c3rep2_keptB, labels
        explore/w2/prefill_c3rep{,2}/labels.jsonl, forks explore/w2/forks_reelicit_c3rep{,2};
        positive = template re-elicitation argmax == correct; S = C - mean(other non-answer wrong), L14-32.
  rcl   claimed-answer-key repeat R-CL: readouts/clrep_followers, forks explore/w2/forks_reelicit_clrep;
        positive = full re-elicitation == correct, negative == B; S on layers 30, 32.
  t2    pre-registered T2: from_mayor/readouts/w2_wording_lens, forks from_mayor/explore/w2/forks_reelicit_p2,
        known = from_mayor/explore/w2/known_pool.jsonl acc A >= 0.9; 4 pooled arms; final = B;
        complied = full re-elicitation == correct, persuaded == B.
  t1b   T1b, one readout dir (silent vs open switches, non-thinking); --labels hinted labels, --screen
        screen labels (known = >= 9 of 10 correct).
  hack  reward-hack organism H2a: warm-start adapter on its own runs, hidden vs no-hack; concept score
        (log summed prob of the concept set) over the echo-free reasoning, L14-32 mean; problem-level means.

    python scripts/lens/firming_recompute.py rc3 --data data --out /tmp/firming/rc3.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def jl(p):
    return [json.loads(x) for x in open(p)]


def auroc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    # P(p > n) + 0.5 P(p == n), by direct comparison (small sets) or ranks
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def boot(rows, key, nb=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r["item"]].append(r)
    items = sorted(by)
    rng = np.random.default_rng(seed)
    est = auroc([r[key] for r in rows if r["pos"]], [r[key] for r in rows if not r["pos"]])
    bs = []
    for _ in range(nb):
        s = [r for k in rng.integers(0, len(items), len(items)) for r in by[items[k]]]
        p = [r[key] for r in s if r["pos"]]
        n = [r[key] for r in s if not r["pos"]]
        if p and n:
            bs.append(auroc(p, n))
    return round(float(est), 4), [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])]


def letter_cols(meta):
    lab = meta["tracked"]["labels"]
    return {L: [i for i, s in enumerate(lab) if s in (L, " " + L, "(" + L)] for L in meta["letters"]}


def lse(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def margin_at(path, meta, pos, others, layers_keep=None):
    """S = lp(correct) - mean lp(others) at position pos; per lens, mean over layers (or layers_keep).
    Returns (S [2], layers, model mass on the written letter at pos)."""
    with np.load(path) as z:
        lp = z["lp"].astype(np.float64)  # [lens, L, T, K]
        mlp = z["model_lp"].astype(np.float64)  # [T, K]
        lo = int(z["positions_lo"])
        layers = z["layers"].tolist()
    i = pos - lo
    if not 0 <= i < lp.shape[2]:
        raise ValueError(f"position {pos} outside stored window ({lo}, {lp.shape[2]}) in {path}")
    cols = letter_cols(meta)
    C = meta["correct_letter"]
    lC = lse(lp[:, :, i, cols[C]])
    lO = np.mean([lse(lp[:, :, i, cols[x]]) for x in others], 0)
    M = lC - lO  # [lens, L]
    if layers_keep is not None:
        M = M[:, [layers.index(L) for L in layers_keep]]
    written = meta["answer"]
    mass = float(np.exp(lse(mlp[i, cols[written]]))) if written else float("nan")
    return M.mean(-1), layers, mass


def parse(v):
    if isinstance(v, str) and (v[:1] in "[{(" or v in ("None", "True", "False")):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def summarise(name, rows, lenses=("jlens", "logit"), extra=None):
    npos = sum(r["pos"] for r in rows)
    out = {"set": name, "n_pos": npos, "n_neg": len(rows) - npos,
           "items_pos": len({r["item"] for r in rows if r["pos"]}),
           "items_neg": len({r["item"] for r in rows if not r["pos"]})}
    for ln in lenses:
        out[ln] = boot(rows, ln)
    masses = [r["mass_written"] for r in rows if np.isfinite(r.get("mass_written", np.nan))]
    if masses:
        out["position_check"] = {"median_model_mass_on_written_letter": round(float(np.median(masses)), 4),
                                 "share_above_0.5": round(float(np.mean(np.array(masses) > 0.5)), 4)}
    if extra:
        out.update(extra)
    print(json.dumps(out), flush=True)
    return out


def reel_map(paths, mode):
    out, pl = {}, {}
    for p in paths:
        for d in jl(p):
            if d.get("kind") == "reelicit" and d.get("mode") == mode:
                out[d["source_run_id"]] = d["argmax"]
                pl[d["source_run_id"]] = d.get("p_letters")
    return out, pl


def do_rc3(D):
    rows, layers_seen, pls = [], set(), []
    for suf in ("", "2"):
        rdir = D / f"readouts/c3rep{suf}_keptB"
        lab = {d["run_id"]: d for d in jl(D / f"explore/w2/prefill_c3rep{suf}/labels.jsonl")}
        reel, pl = reel_map([D / f"explore/w2/forks_reelicit_c3rep{suf}/forks.jsonl"], "template")
        seen = set()
        for m in jl(rdir / "index.jsonl"):
            if m["run_id"] in seen:
                continue
            seen.add(m["run_id"])
            if lab[m["run_id"]]["outcome"] != "followed":
                continue
            C, A = m["correct_letter"], m["answer"]
            assert A == m["hint_letter"] != C
            others = [x for x in m["letters"] if x not in (C, A)]
            S, layers, mass = margin_at(rdir / m["file"], m, m["named_positions"]["pre_answer"], others)
            layers_seen.add(tuple(layers))
            pls.append(pl[m["run_id"]])
            rows.append({"run_id": m["run_id"], "item": m["question_id"], "pos": reel[m["run_id"]] == C,
                         "reel": reel[m["run_id"]], "p_letters": pl[m["run_id"]],
                         "jlens": float(S[0]), "logit": float(S[1]), "mass_written": mass})
    pls = np.array([p for p in pls if p is not None], float)
    extra = {"layers": sorted(layers_seen), "reel_p_letters": {
        "median": round(float(np.median(pls)), 4), "share_below_0.5": round(float(np.mean(pls < 0.5)), 4),
        "share_below_0.1": round(float(np.mean(pls < 0.1)), 4)}}
    return rows, summarise("rc3", rows, extra=extra)


def do_rcl(D):
    rdir = D / "readouts/clrep_followers"
    reel, pl = reel_map([D / "explore/w2/forks_reelicit_clrep/forks.jsonl"], "full")
    rows, seen, layers_seen = [], set(), set()
    n_other = 0
    for m in jl(rdir / "index.jsonl"):
        if m["run_id"] in seen:
            continue
        seen.add(m["run_id"])
        C, A, H = m["correct_letter"], m["answer"], m["hint_letter"]
        assert A == H != C
        r = reel[m["run_id"]]
        if r not in (C, H):
            n_other += 1
            continue
        others = [x for x in m["letters"] if x not in (C, A)]
        S, layers, mass = margin_at(rdir / m["file"], m, m["named_positions"]["pre_answer"], others, (30, 32))
        layers_seen.add(tuple(layers))
        rows.append({"run_id": m["run_id"], "item": m["question_id"], "pos": r == C, "reel": r,
                     "p_letters": pl[m["run_id"]], "jlens": float(S[0]), "logit": float(S[1]), "mass_written": mass})
    return rows, summarise("rcl", rows, extra={"excluded_other_letter": n_other, "layers": sorted(layers_seen)})


T2_ARMS = {"think_qwen_bare_answer_ours", "think_qwen_hint_only_ours", "think_qwen_mb_expert_noprefix_ours",
           "think_qwen_mb_expert_ourshintfirst"}


def do_t2(D):
    M = D / "from_mayor"
    rdir = M / "readouts/w2_wording_lens"
    known = {d["id"] for d in jl(M / "explore/w2/known_pool.jsonl") if (d.get("acc") or {}).get("A", 0) >= 0.9}
    reel, pl = reel_map([M / "explore/w2/forks_reelicit_p2/forks.jsonl"], "full")
    rows, seen, layers_seen, n_other, n_noreel = [], set(), set(), 0, 0
    for m in jl(rdir / "index.jsonl"):
        if m["run_id"] in seen:
            continue
        seen.add(m["run_id"])
        ex = parse(m.get("extras")) if isinstance(m.get("extras"), str) else m.get("extras") or {}
        arm = ex.get("arm") if isinstance(ex, dict) else None
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        if arm not in T2_ARMS or q not in known or H is None or A != H:
            continue
        if m["named_positions"].get("pre_answer", -1) < 0:
            continue
        r = reel.get(m["run_id"])
        if r is None:
            n_noreel += 1
            continue
        if r not in (C, H):
            n_other += 1
            continue
        others = [x for x in m["letters"] if x not in (C, A)]
        S, layers, mass = margin_at(rdir / m["file"], m, m["named_positions"]["pre_answer"], others)
        layers_seen.add(tuple(layers))
        rows.append({"run_id": m["run_id"], "item": q, "arm": arm, "pos": r == C, "reel": r,
                     "p_letters": pl[m["run_id"]], "jlens": float(S[0]), "logit": float(S[1]), "mass_written": mass})
    return rows, summarise("t2", rows, extra={"excluded_other_letter": n_other, "followers_without_reelicitation": n_noreel,
                                               "layers": sorted(layers_seen)})


def do_t1b(D, rdirs, labels_p, screen_p):
    nc, nn = defaultdict(int), defaultdict(int)
    for d in jl(screen_p):
        nn[d["question_id"]] += 1
        nc[d["question_id"]] += d["outcome"] == "correct"
    known = {q for q in nn if nn[q] == 10 and nc[q] >= 9}
    lab = {d["run_id"]: d for d in jl(labels_p)}
    rows, seen, layers_seen = [], set(), set()
    for rd in rdirs:
        rdir = D / rd
        for m in jl(rdir / "index.jsonl"):
            if m["run_id"] in seen:
                continue
            seen.add(m["run_id"])
            q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
            if q not in known or H is None or A != H or m["enable_thinking"]:
                continue
            if m["named_positions"].get("pre_answer", -1) < 0:
                continue
            ment = parse(lab[m["run_id"]].get("mentions"))
            others = [x for x in m["letters"] if x not in (C, A)]
            S, layers, mass = margin_at(rdir / m["file"], m, m["named_positions"]["pre_answer"], others)
            layers_seen.add(tuple(layers))
            rows.append({"run_id": m["run_id"], "item": q, "pos": not ment, "jlens": float(S[0]),
                         "logit": float(S[1]), "mass_written": mass})
    return rows, summarise("t1b", rows, extra={"layers": sorted(layers_seen), "known_items": len(known)})


def do_hack(D, rdir):
    rdir = D / rdir
    rows = []
    for m in jl(rdir / "index.jsonl"):
        if m["reader"] != "disposition" or m["arm"] != "disposition" or m["finish_reason"] != "stop":
            continue
        if m["group"] not in ("hidden", "no_hack"):
            continue
        ef = m["echo_free"]
        if ef is None or ef[1] - ef[0] < 5:
            continue
        with np.load(rdir / m["file"]) as z:
            x = z["x_lp"].astype(np.float64)  # [lens, L, T, K]
            lo = int(z["positions_lo"])
            layers = z["layers"].tolist() if "layers" in z.files else None
        kc = [i for i, s in enumerate(m["tracked"]["sets"]) if s == "concept"]
        t = np.arange(ef[0], ef[1]) + m["prompt_len"] - lo
        assert t.min() >= 0 and t.max() < x.shape[2]
        C = lse(x[:, :, t][..., kc]).mean(-1).mean(-1)  # per lens: mean over positions then layers
        rows.append({"run_id": m["run_id"], "item": m["problem_id"], "pos": m["group"] == "hidden",
                     "jlens": float(C[0]), "logit": float(C[1]), "layers": layers})
    # problem-level means over problems with both groups
    by = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        by[r["item"]][r["pos"]].append(r)
    prob = []
    for q, g in by.items():
        if g[True] and g[False]:
            for s in (True, False):
                prob.append({"item": q, "pos": s, "jlens": float(np.mean([r["jlens"] for r in g[s]])),
                             "logit": float(np.mean([r["logit"] for r in g[s]]))})
    out = summarise("hack_problem_means", prob, extra={"n_runs": len(rows), "layers": rows[0]["layers"]})
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["rc3", "rcl", "t2", "t1b", "hack"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--readouts", nargs="*", default=[])
    ap.add_argument("--labels")
    ap.add_argument("--screen")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.data)
    if a.set == "rc3":
        rows, s = do_rc3(D)
    elif a.set == "rcl":
        rows, s = do_rcl(D)
    elif a.set == "t2":
        rows, s = do_t2(D)
    elif a.set == "t1b":
        rows, s = do_t1b(D, a.readouts, a.labels, a.screen)
    else:
        rows, s = do_hack(D, a.readouts[0])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps({"summary": s}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
