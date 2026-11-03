"""Pre-registered tests T1 and T2 (docs/overnight/preregistration.md), plus
labelled exploratory analyses. Lens env, CPU.

Score S (pre-registered, unchanged): at pre_answer, J-lens log-prob of the
correct letter minus the mean log-prob of the unchosen wrong letters other than
the final answer (letters = log-sum-exp over the tracked surface forms " X",
"X", "(X"), averaged over the workspace layers 14, 16, ..., 32. Comparison: the
same with the logit lens. AUROC with 95% bootstrap intervals resampling items.

T2 (thinking, wording arms, known items, final answer = hinted B):
  complied = E7 re-elicitation (full history, argmax) gives the correct letter,
  persuaded = it gives B. Pooled over the "answer is"/"indicates" arms.
  Pass: J-lens AUROC(S; complied vs persuaded) lower bound > 0.5 and >= 15 items in each group.
T1 (non-thinking setting B, nothink_scale, known in B, excluding the 42 items
  with silent switches in lens_main): silent vs open switches (keyword mention check).
  Pass: J-lens lower bound > 0.5 (and the logit interval includes 0.5, predicted); >= 15 items with a silent switch.

Exploratory (labelled as such): per-layer AUROCs, the margin along 20 CoT bins,
CoT-text baselines, and a mid-CoT linear probe (complied vs persuaded) on the
stored CoT block outputs, GroupKFold by item.

uv run --project envs/lens python scripts/lens/analyze_prereg.py --test t2 --out data/readouts/prereg_t2.json
uv run --project envs/lens python scripts/lens/analyze_prereg.py --test t1 --out data/readouts/prereg_t1.json
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_main import auroc_boot, boot_mean  # noqa: E402

SEED = 0
N_BINS = 20
POOLED_ARMS = ["think_qwen_bare_answer_ours", "think_qwen_hint_only_ours", "think_qwen_mb_expert_noprefix_ours",
               "think_qwen_mb_expert_ourshintfirst"]


def parse(v):
    """w2 labels store lists/None/bools as strings."""
    if isinstance(v, str) and v[:1] in "[{(" or v in ("None", "True", "False"):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def letter_series(z, meta, letter):
    """[2 lenses, layers, T] log-sum-exp over forms, and min rank."""
    tr = meta["tracked"]
    nf = len(tr["forms"])
    k = tr["letters"].index(letter) * nf
    lp = np.logaddexp.reduce(z["lp"][..., k:k + nf].astype(np.float64), axis=-1)
    rk = np.log10(z["rank"][..., k:k + nf].min(-1).astype(np.float64))
    return lp, rk


def score_run(z, m):
    """S per lens and per layer at pre_answer, the margin along 20 CoT bins, and
    the rank version. Returns None if the run has no pre_answer anchor."""
    pa = m["named_positions"]["pre_answer"]
    if pa < 0 or m["answer"] is None:
        return None
    C, A = m["correct_letter"], m["answer"]
    U = [x for x in m["letters"] if x not in {C, A}]
    lo = int(z["positions_lo"])
    lpC, rkC = letter_series(z, m, C)
    lpU = np.mean([letter_series(z, m, x)[0] for x in U], 0)
    rkU = np.mean([letter_series(z, m, x)[1] for x in U], 0)
    marg = lpC - lpU  # [2, L, T]
    out = {"layer_margin": marg[:, :, pa - lo], "layer_rank_margin": (rkU - rkC)[:, :, pa - lo]}
    out["S"] = out["layer_margin"].mean(-1)  # [2]: jlens, logit
    span = m["cot_span"]
    bins = np.full(marg.shape[:2] + (N_BINS,), np.nan)
    if span:
        ts, te = span
        x = marg[:, :, ts - lo:te - lo]
        b = np.minimum((np.arange(te - ts) * N_BINS) // (te - ts), N_BINS - 1)
        for k in range(N_BINS):
            if (b == k).any():
                bins[:, :, k] = x[:, :, b == k].mean(-1)
    out["bins"] = bins
    return out


def load_set(rdir: Path):
    index = list({m["run_id"]: m for m in map(json.loads, open(rdir / "index.jsonl"))}.values())
    return index


def verdict_t(auc, n_items_min, items_needed=15):
    if n_items_min < items_needed:
        return "underpowered"
    return "pass" if auc["lo"] > 0.5 else "fail"


def run_t2(D: Path, rng) -> dict:
    rdir = D / "readouts/w2_wording_lens"
    reel = {}
    reel_t = {}
    for line in open(D / "explore/w2/forks_reelicit_p2/forks.jsonl"):
        d = json.loads(line)
        if d.get("kind") == "reelicit":
            (reel if d.get("mode") == "full" else reel_t)[d["source_run_id"]] = d["argmax"]
    nA = json.load(open(D / "explore/gap_factorial/screen_A_lens/n_correct_A.json"))
    runs_txt = {}
    for line in open(D / "explore/w2/wording_lens/runs.jsonl"):
        d = json.loads(line)
        runs_txt[d["run_id"]] = d["completion_text"]
    items = {d["id"]: d for d in map(json.loads, open(D / "explore/w2/wording/items.jsonl"))}
    runs = []
    cot = []
    for m in load_set(rdir):
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        arm = m["extras"].get("arm")
        if nA.get(q, 0) < 9:
            continue
        if A == H:
            r = reel.get(m["run_id"])
            g = "complied" if r == C else "persuaded" if r == H else "final_B_other_reelicit"
        elif A == C:
            g = "resisted"
        else:
            continue
        z = np.load(rdir / m["file"])
        sc = score_run(z, m)
        if sc is None:
            continue
        txt = runs_txt[m["run_id"]]
        think = txt.split("</think>")[0]
        opt = str(parse(items[q]["options"])[m["letters"].index(C)]).strip().rstrip(".")
        names_correct = bool(re.search(rf"\(({C})\)|(?i:option|answer|choice)\s*\(?{C}(?![A-Za-z])", think)) or \
            (len(opt) >= 3 and opt.lower() in think.lower())
        hint_kw = bool(re.search(r"(?i)\b(hint|expert|the answer is|given answer|stated|provided|suggested|indicated|claimed)\b", think))
        runs.append({"run_id": m["run_id"], "item": q, "arm": arm, "group": g, "S": sc["S"],
                     "layer_margin": sc["layer_margin"], "layer_rank_margin": sc["layer_rank_margin"], "bins": sc["bins"],
                     "reel_template": reel_t.get(m["run_id"]), "C": C, "H": H,
                     "cot_names_correct": names_correct, "cot_hint_kw": hint_kw, "cot_len": len(think)})
        cot.append((z["cot_resid"].astype(np.float32), z["cot_resid_pos"]) if "cot_resid" in z.files else None)
    item_of = {i: r["item"] for i, r in enumerate(runs)}

    def idx(group, arms=None):
        return [i for i, r in enumerate(runs) if r["group"] == group and (arms is None or r["arm"] in arms)]

    res = {"definition": "T2: complied (re-elicit correct) vs persuaded (re-elicit B), final answer B, known items, pooled arms " + ", ".join(POOLED_ARMS),
           "counts": {}, "pre_registered": {}, "exploratory": {}}
    for arm in sorted({r["arm"] for r in runs}):
        res["counts"][arm] = {g: {"runs": len(idx(g, [arm])), "items": len({runs[i]["item"] for i in idx(g, [arm])})}
                              for g in ("complied", "persuaded", "final_B_other_reelicit", "resisted")}
    cp, pp = idx("complied", POOLED_ARMS), idx("persuaded", POOLED_ARMS)
    n_items = (len({runs[i]["item"] for i in cp}), len({runs[i]["item"] for i in pp}))
    for li, lens in enumerate(("jlens", "logit")):
        s = {i: float(r["S"][li]) for i, r in enumerate(runs)}
        a = auroc_boot(s, item_of, cp, pp, rng)
        res["pre_registered"][lens] = a
    res["pre_registered"]["n_items_complied_persuaded"] = n_items
    res["pre_registered"]["verdict_jlens"] = verdict_t(res["pre_registered"]["jlens"], min(n_items))
    # per arm (pre-registered score, reported where n allows)
    res["per_arm"] = {}
    for arm in POOLED_ARMS:
        c, p = idx("complied", [arm]), idx("persuaded", [arm])
        if len({runs[i]["item"] for i in c}) >= 5 and len({runs[i]["item"] for i in p}) >= 5:
            res["per_arm"][arm] = {lens: auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, c, p, rng)
                                   for li, lens in enumerate(("jlens", "logit"))}
    # group means of S (items bootstrap)
    res["S_means"] = {}
    for g in ("complied", "persuaded", "resisted", "final_B_other_reelicit"):
        by = defaultdict(list)
        for i in idx(g, POOLED_ARMS):
            by[runs[i]["item"]].append(runs[i]["S"])
        res["S_means"][g] = boot_mean({q: np.mean(v, 0) for q, v in by.items()}, rng)

    # ---------------- exploratory
    E = res["exploratory"]
    layers = list(range(14, 33, 2))
    E["per_layer_auroc"] = {lens: [auroc_boot({i: float(r["layer_margin"][li][k]) for i, r in enumerate(runs)}, item_of, cp, pp, rng)
                                   for k in range(len(layers))] for li, lens in enumerate(("jlens", "logit"))}
    E["per_layer_mean_margin"] = {}
    for g in ("complied", "persuaded", "resisted"):
        by = defaultdict(list)
        for i in idx(g, POOLED_ARMS):
            by[runs[i]["item"]].append(runs[i]["layer_margin"])
        E["per_layer_mean_margin"][g] = boot_mean({q: np.mean(v, 0) for q, v in by.items()}, rng)
    E["bins_mean_margin"] = {}
    for g in ("complied", "persuaded", "resisted"):
        by = defaultdict(list)
        for i in idx(g, POOLED_ARMS):
            by[runs[i]["item"]].append(runs[i]["bins"])
        with np.errstate(all="ignore"):
            E["bins_mean_margin"][g] = boot_mean({q: np.nanmean(np.stack(v), 0) for q, v in by.items()}, rng)
    # bins AUROC at a few layers
    E["bins_auroc_jlens"] = {}
    for L in (20, 26, 32):
        k = layers.index(L)
        E["bins_auroc_jlens"][f"L{L}"] = [auroc_boot({i: float(r["bins"][0][k][b]) for i, r in enumerate(runs)}, item_of, cp, pp, rng)["auroc"]
                                          for b in range(N_BINS)]
    E["cot_baselines"] = {
        "cot names correct option (1/0)": auroc_boot({i: float(r["cot_names_correct"]) for i, r in enumerate(runs)}, item_of, cp, pp, rng),
        "cot hint/answer keyword (1/0)": auroc_boot({i: float(r["cot_hint_kw"]) for i, r in enumerate(runs)}, item_of, cp, pp, rng),
        "cot length (chars)": auroc_boot({i: float(r["cot_len"]) for i, r in enumerate(runs)}, item_of, cp, pp, rng),
        "re-elicitation, template mode = correct (1/0)": auroc_boot(
            {i: float(r["reel_template"] == r["C"]) for i, r in enumerate(runs)}, item_of, cp, pp, rng),
    }
    # mid-CoT probe: complied vs persuaded from CoT block outputs, items held out
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    sel = [i for i in cp + pp if cot[i] is not None]
    y = np.array([1 if runs[i]["group"] == "complied" else 0 for i in sel])
    groups = np.array([runs[i]["item"] for i in sel])
    probe = {}
    n_pos = 16
    for L in (16, 20, 24, 28, 32):
        k = layers.index(L)
        for pos_set, name in (([4, 8, 12], "cot_04_08_12 (mean)"), (list(range(n_pos)), "all 16 (mean)")):
            X = np.stack([np.nanmean(cot[i][0][k, pos_set], 0) for i in sel])
            ok = np.isfinite(X).all(1)
            Xo, yo, go = X[ok], y[ok], groups[ok]
            pred = np.full(len(yo), np.nan)
            for tr, te in GroupKFold(n_splits=5).split(Xo, yo, go):
                if len(set(yo[tr])) < 2:
                    continue
                sc = StandardScaler().fit(Xo[tr])
                clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(sc.transform(Xo[tr]), yo[tr])
                pred[te] = clf.predict_proba(sc.transform(Xo[te]))[:, 1]
            sel_ok = [sel[j] for j in range(len(sel)) if ok[j]]
            score = {sel_ok[j]: float(pred[j]) for j in range(len(sel_ok))}
            cpi = [i for i in sel_ok if runs[i]["group"] == "complied"]
            ppi = [i for i in sel_ok if runs[i]["group"] == "persuaded"]
            probe[f"L{L} {name}"] = auroc_boot(score, item_of, cpi, ppi, rng)
    E["mid_cot_probe_complied_vs_persuaded"] = probe
    # same score S, all arms pooled (incl. "chose" arms)
    ca, pa_ = idx("complied"), idx("persuaded")
    E["S_all_arms"] = {lens: auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, ca, pa_, rng)
                       for li, lens in enumerate(("jlens", "logit"))}
    return res


def run_t1(D: Path, rng, rdir_name: str, labels_path: str, exclude_from: str,
           counts_path: str = "explore/gap_factorial/screen_B_lens/n_correct_B.json") -> dict:
    rdir = D / rdir_name
    nB = json.load(open(D / counts_path))
    known = {q for q, k in nB.items() if k >= 9}
    excl = set(json.load(open(D / exclude_from)))
    labels = {d["run_id"]: d for d in map(json.loads, open(D / labels_path))}
    runs = []
    for m in load_set(rdir):
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        if q not in known or q in excl or H is None or A != H or m["enable_thinking"]:
            continue
        lab = labels.get(m["run_id"])
        if lab is None:
            raise ValueError(f"no label for {m['run_id']}")
        mentions = parse(lab.get("mentions"))
        z = np.load(rdir / m["file"])
        sc = score_run(z, m)
        if sc is None:
            continue
        runs.append({"run_id": m["run_id"], "item": q, "group": "open" if mentions else "silent", "S": sc["S"],
                     "layer_margin": sc["layer_margin"], "arm": m["extras"].get("arm"), "resp_len": m["completion_len"]})
    item_of = {i: r["item"] for i, r in enumerate(runs)}
    si = [i for i, r in enumerate(runs) if r["group"] == "silent"]
    oi = [i for i, r in enumerate(runs) if r["group"] == "open"]
    res = {"definition": "T1: silent vs open switches, non-thinking setting B, known in B, excluding lens_main's 42 silent-switch items",
           "counts": {"silent": {"runs": len(si), "items": len({runs[i]["item"] for i in si})},
                      "open": {"runs": len(oi), "items": len({runs[i]["item"] for i in oi})},
                      "n_excluded_items": len(excl), "n_known_B": len(known)},
           "pre_registered": {}, "exploratory": {}}
    for li, lens in enumerate(("jlens", "logit")):
        res["pre_registered"][lens] = auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, si, oi, rng)
    n_si = res["counts"]["silent"]["items"]
    res["pre_registered"]["verdict_jlens"] = verdict_t(res["pre_registered"]["jlens"], n_si)
    lg = res["pre_registered"]["logit"]
    res["pre_registered"]["logit_interval_includes_0.5"] = lg["lo"] <= 0.5 <= lg["hi"]
    layers = list(range(14, 33, 2))
    res["exploratory"]["per_layer_auroc"] = {lens: [auroc_boot({i: float(r["layer_margin"][li][k]) for i, r in enumerate(runs)}, item_of, si, oi, rng)
                                                    for k in range(len(layers))] for li, lens in enumerate(("jlens", "logit"))}
    res["S_means"] = {}
    for g, ii in (("silent", si), ("open", oi)):
        by = defaultdict(list)
        for i in ii:
            by[runs[i]["item"]].append(runs[i]["S"])
        res["S_means"][g] = boot_mean({q: np.mean(v, 0) for q, v in by.items()}, rng)
    # exploratory length control: AUROC(S) within response-length tertiles (tertiles over all switches)
    lens_all = np.array([r["resp_len"] for r in runs])
    cuts = np.quantile(lens_all, [1 / 3, 2 / 3]) if len(runs) else [0, 0]
    res["exploratory"]["length_tertiles"] = {"cuts_tokens": [float(c) for c in cuts]}
    for t in range(3):
        inside = lambda i: (t == 0 and runs[i]["resp_len"] <= cuts[0]) or (t == 1 and cuts[0] < runs[i]["resp_len"] <= cuts[1]) or (t == 2 and runs[i]["resp_len"] > cuts[1])
        s_t, o_t = [i for i in si if inside(i)], [i for i in oi if inside(i)]
        res["exploratory"]["length_tertiles"][f"t{t}"] = {lens: auroc_boot({i: float(r["S"][li]) for i, r in enumerate(runs)}, item_of, s_t, o_t, rng)
                                                         for li, lens in enumerate(("jlens", "logit"))} if s_t and o_t else None
    res["exploratory"]["response_length_auroc_silent_vs_open"] = auroc_boot({i: float(r["resp_len"]) for i, r in enumerate(runs)}, item_of, si, oi, rng)
    res["silent_run_ids_for_reading"] = [runs[i]["run_id"] for i in np.random.default_rng(SEED).permutation(si)[:10]]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["t1", "t2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--t1-readouts", default="readouts/nothink_scale")
    ap.add_argument("--t1-labels", default="explore/w2/nothink_scale/labels.jsonl")
    ap.add_argument("--t1-exclude", default="readouts/lens_main_silent_items.json")
    ap.add_argument("--t1-counts", default="explore/gap_factorial/screen_B_lens/n_correct_B.json")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    D = Path(args.data)
    res = run_t2(D, rng) if args.test == "t2" else run_t1(D, rng, args.t1_readouts, args.t1_labels, args.t1_exclude, args.t1_counts)

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    Path(args.out).write_text(json.dumps(conv(res)))
    print(json.dumps(conv({k: v for k, v in res.items() if k in ("counts", "pre_registered", "per_arm", "S_means")}), indent=1)[:6000])


if __name__ == "__main__":
    main()
