"""Main lens analysis, overnight 2026-11-03 (lens env, CPU): is the correct
letter still represented when Qwen3-8B (thinking) outputs a wrong option on an
item it knows, and does that differ between instructed (K->B), persuaded
switches and honest errors? Exploratory; derived numbers only.

Inputs (all on arena9-mayor): batch readouts (scripts/lens/readout_runs.py) of
  w2/instructed, w2/grader, gap_factorial/factorial_think2, gap_factorial/screen_A_lens
plus their runs.jsonl / items.jsonl, the E7 re-elicitation forks and the
setting-A screen counts (screen_A_lens/n_correct_A.json).

Groups (known = >= 9/10 correct in screen setting A; all groups on known items
except the honest-error ones):
  kb_instr        I1/I3, final = B (instructed to argue for B)
  i2_knows        I2, "Argued: B", final "Answer:" = correct (reports A while arguing B)
  i2_says_b       I2, final = B
  persuaded_claim V2/V2', final = B and full-history re-elicitation = B
  deferred_claim  V2/V2', final = B, re-elicitation = correct
  persuaded_mb    factorial think_qwen_mb_expert_ours, final = B
  followed_ours   factorial think_qwen_ours_expert_ours, final = B
  resisted_claim  V2/V2', final = correct
  resisted_mb     factorial think_qwen_mb_expert_ours, final = correct
  unhinted        screen A (2 samples per known item) + w2 unhinted, final = correct
  honest_error    screen A items with 2-8/10 correct, final = a wrong letter
  unstable_correct  the same items, final = correct

Measures (form "any": log-sum-exp over " X", "X", "(X"; min rank), per lens x layer:
  margin(T) = lp(T) - mean lp(U), and the same with log10 rank (mean over U
  minus T; positive = T ranks better), where U = the letters that are neither
  correct, nor the final answer, nor the shown hint (2 letters in every group
  here; for unhinted runs U excludes the item's hinted letter).
  T = correct letter ("corr"); T = the salient unchosen letter ("sal"): the
  correct letter when the final answer is wrong, the shown hint when the run
  resisted, the item's hinted letter in unhinted runs. Salience control:
  corr margin in followed runs vs sal margin in resisted runs.
Positions: last_prompt, think_end, pre_answer, 20 bins over the CoT (all
  positions; echo-free positions only), echo-free mean and the last echo-free
  position. Echo-free = CoT positions t <= m - 2, m = first CoT token of a
  mention of the correct letter (regex) or of the correct option's text.

Probe: multinomial logistic regression on the block outputs at last_prompt,
  think_end and pre_answer (layers PROBE_LAYERS), label = correct letter,
  trained on unhinted + resisted runs only (there the correct letter is also
  the final answer: an answer probe on its training data), 5-fold GroupKFold
  by item; every run is scored by the fold that held out its item.

Fixed in advance (before looking at any result): headline score = mean over
  layers 20-28 of the corr margin (log-prob) at think_end and the echo-free
  mean, for each lens; probe headline = P(correct) at think_end, layer 24.

uv run --project envs/lens python scripts/lens/analyze_main.py --out data/readouts/lens_main.json
"""

import argparse
import bisect
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from jmr import config

N_BINS, N_BOOT, SEED = 20, 2000, 0
LENS_NAMES = ("jlens", "logit")
PROBE_LAYERS = [8, 12, 16, 20, 24, 28, 32, 35]
PROBE_C = 0.05
HEAD_LAYERS = [20, 22, 24, 26, 28]
SETS = {  # readout dir -> run-set dir (under data/)
    "readouts/w2_instructed": "explore/w2/instructed",
    "readouts/w2_grader": "explore/w2/grader",
    "readouts/gap_factorial_factorial_think2": "explore/gap_factorial/factorial_think2",
    "readouts/gap_factorial_screen_A_lens": "explore/gap_factorial/screen_A_lens",
}
FORKS = "explore/w2/forks_reelicit_p1/forks.jsonl"
SCREEN_COUNTS = "explore/gap_factorial/screen_A_lens/n_correct_A.json"
SCREEN_ITEMS = "explore/gap_factorial/screen/items.jsonl"


def letter_mention_res(L: str) -> list[re.Pattern]:
    e = re.escape(L)
    return [re.compile(p.encode()) for p in (
        rf"\({e}\)", rf"(?<![A-Za-z]){e}\)", rf"(?i:option|choice|answer|letter)\s*\(?{e}(?![A-Za-z])",
        rf"\*\*{e}(?![A-Za-z])", rf"\b{e}:\s")]


def first_mention(text: bytes, lo: int, hi: int, letter: str, option: str) -> int | None:
    best = None
    for pat in letter_mention_res(letter):
        m = pat.search(text, lo, hi)
        if m and (best is None or m.start() < best):
            best = m.start()
    opt = option.strip().rstrip(".").strip()
    if len(opt) >= 3:
        m = re.compile(re.escape(opt.encode()), re.IGNORECASE).search(text, lo, hi)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def boot_mean(vals_by_item: dict, rng) -> dict:
    keys = sorted(vals_by_item)
    if not keys:
        return {"n_items": 0}
    x = np.stack([vals_by_item[k] for k in keys]).astype(np.float64)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    with np.errstate(all="ignore"):
        bs = np.nanmean(x[idx], axis=1)
        return {"n_items": len(keys), "mean": np.nanmean(x, 0).round(4).tolist(),
                "lo": np.nanpercentile(bs, 2.5, axis=0).round(4).tolist(),
                "hi": np.nanpercentile(bs, 97.5, axis=0).round(4).tolist()}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_boot(score: dict, items: dict, g_pos: list, g_neg: list, rng) -> dict:
    """AUROC of score (run index -> value) for g_pos vs g_neg runs, 95% CI by
    resampling items."""
    pi = defaultdict(list)
    ni = defaultdict(list)
    for i in g_pos:
        pi[items[i]].append(score[i])
    for i in g_neg:
        ni[items[i]].append(score[i])
    keys = sorted(set(pi) | set(ni))
    if not pi or not ni:
        return {"auroc": None}
    a = auroc(np.array([v for k in pi for v in pi[k]]), np.array([v for k in ni for v in ni[k]]))
    bs = []
    for _ in range(N_BOOT // 2):
        samp = rng.choice(keys, size=len(keys))
        p = np.array([v for k in samp for v in pi.get(k, [])])
        n = np.array([v for k in samp for v in ni.get(k, [])])
        if len(p) and len(n):
            bs.append(auroc(p, n))
    return {"auroc": round(a, 4), "lo": round(float(np.nanpercentile(bs, 2.5)), 4),
            "hi": round(float(np.nanpercentile(bs, 97.5)), 4), "n_pos": len(g_pos), "n_neg": len(g_neg),
            "n_items_pos": len(pi), "n_items_neg": len(ni)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    D = Path(args.data)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    from transformers import AutoTokenizer
    from jmr.tokens import token_bytes
    mcfg = config.load("model")
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])

    n_correct_A = json.load(open(D / SCREEN_COUNTS))
    known = {q for q, k in n_correct_A.items() if k >= 9}
    unstable = {q for q, k in n_correct_A.items() if 2 <= k <= 8}
    items = {}
    for p in [D / SCREEN_ITEMS] + [D / s / "items.jsonl" for s in SETS.values()]:
        if p.exists():
            for line in open(p):
                d = json.loads(line)
                if d["id"] in items and items[d["id"]]["correct"] != d["correct"]:
                    raise ValueError(f"items disagree on the correct letter for {d['id']}")
                items.setdefault(d["id"], d)
    reelicit = {}
    for line in open(D / FORKS):
        d = json.loads(line)
        if d.get("kind") == "reelicit" and d.get("mode") == "full":
            reelicit[d["source_run_id"]] = d["argmax"]

    # ---------------------------------------------------------- per-run pass
    runs = []  # dicts with metadata + metrics
    feats = {a: [] for a in ("last_prompt", "think_end", "pre_answer")}
    skipped = defaultdict(int)
    for rdir, sdir in SETS.items():
        if not (D / rdir / "index.jsonl").exists():
            print(f"WARNING: no readouts in {rdir}; skipped", flush=True)
            continue
        index = [json.loads(line) for line in open(D / rdir / "index.jsonl")]
        index = list({m["run_id"]: m for m in index}.values())
        comp = {}
        for line in open(D / sdir / "runs.jsonl"):
            d = json.loads(line)
            comp[d["run_id"]] = d["prompt_token_ids"] + d["completion_token_ids"]
        for m in index:
            q, C, A = m["question_id"], m["correct_letter"], m["answer"]
            H = m["hint_letter"]
            arm = m["extras"].get("arm", m["condition"])
            if A is None or m["named_positions"]["think_end"] < 0 or m["named_positions"]["pre_answer"] < 0:
                skipped["no answer / no </think>"] += 1
                continue
            # group
            g = None
            if rdir.endswith("screen_A_lens"):
                if q in known and A == C:
                    g = "unhinted"
                elif q in unstable:
                    g = "unstable_correct" if A == C else "honest_error"
            elif q in known:
                if arm == "unhinted":
                    g = "unhinted" if A == C else None
                elif arm in ("I1_argue", "I3_conceal"):
                    g = "kb_instr" if A == H else None
                elif arm == "I2_debate":
                    g = "i2_knows" if A == C else "i2_says_b" if A == H else None
                elif arm in ("V2_info", "V2p_chen"):
                    if A == H:
                        r = reelicit.get(m["run_id"])
                        g = "persuaded_claim" if r == H else "deferred_claim" if r == C else None
                    elif A == C:
                        g = "resisted_claim"
                elif arm == "think_qwen_mb_expert_ours":
                    g = "persuaded_mb" if A == H else "resisted_mb" if A == C else None
                elif arm == "think_qwen_ours_expert_ours":
                    g = "followed_ours" if A == H else "resisted_ours" if A == C else None
            if g is None:
                skipped["not in a group"] += 1
                continue
            item_h = items.get(q, {}).get("hinted")
            shown_h = H
            letters = m["letters"]
            sal = C if A != C else (shown_h if shown_h else item_h)
            # Excluded from U: correct, answer, and the salient letter (shown hint,
            # or for unhinted correct runs the item's hinted letter).
            U = [x for x in letters if x not in {C, A, sal}]
            if len(U) != 2:
                skipped[f"|U| = {len(U)}"] += 1
                continue

            z = np.load(D / rdir / m["file"])
            lo = int(z["positions_lo"])
            tr = m["tracked"]
            nf = len(tr["forms"])
            lp_all, rank_all = z["lp"].astype(np.float64), z["rank"].astype(np.float64)  # [2, L, T, 12]

            def letter_lp(Lx):
                k = tr["letters"].index(Lx) * nf
                return np.logaddexp.reduce(lp_all[..., k:k + nf], axis=-1)  # [2, L, T]

            def letter_lr(Lx):
                k = tr["letters"].index(Lx) * nf
                return np.log10(rank_all[..., k:k + nf].min(-1))

            lpU = np.mean([letter_lp(x) for x in U], 0)
            lrU = np.mean([letter_lr(x) for x in U], 0)
            series = {"corr/lp": letter_lp(C) - lpU, "corr/lr": lrU - letter_lr(C),
                      "sal/lp": letter_lp(sal) - lpU, "sal/lr": lrU - letter_lr(sal),
                      "corr_raw/lp": letter_lp(C)}
            npos = m["named_positions"]
            ts, te = m["cot_span"]
            # echo control
            ids = comp[m["run_id"]]
            pieces = token_bytes(tok, ids)
            starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
            text = b"".join(pieces)
            opt = items[q]["options"][letters.index(C)] if q in items else ""
            fm = first_mention(text, int(starts[ts]), int(starts[te]), C, opt)
            mpos = None if fm is None else bisect.bisect_right(starts.tolist(), fm) - 1
            ef_end = te if mpos is None else max(ts, mpos - 1)  # echo-free positions [ts, ef_end)
            pos = np.arange(ts, te)
            b = np.minimum(((pos - ts) * N_BINS) // (te - ts), N_BINS - 1)
            ef = pos < ef_end
            met = {}
            for key, s in series.items():
                for anc in ("last_prompt", "think_end", "pre_answer"):
                    met[f"{key}/{anc}"] = s[:, :, npos[anc] - lo]  # [2, L]
                x = s[:, :, ts - lo:te - lo]
                for tag, mask in (("all", np.ones_like(ef)), ("ef", ef)):
                    bins = np.full(x.shape[:2] + (N_BINS,), np.nan)
                    for k in range(N_BINS):
                        sel = (b == k) & mask
                        if sel.any():
                            bins[:, :, k] = x[:, :, sel].mean(-1)
                    met[f"{key}/bins_{tag}"] = bins
                met[f"{key}/ef_mean"] = x[:, :, ef].mean(-1) if ef.any() else np.full(x.shape[:2], np.nan)
                met[f"{key}/ef_last"] = x[:, :, ef_end - 1 - ts] if ef.any() else np.full(x.shape[:2], np.nan)
            # probe features
            names = m["resid_anchor_names"]
            res = z["resid"]  # [36, 3, d]
            for a in feats:
                feats[a].append(res[PROBE_LAYERS, names.index(a)].astype(np.float32))
            runs.append({"run_id": m["run_id"], "item": q, "group": g, "arm": arm, "C": C, "A": A, "H": shown_h,
                         "letters": letters, "cot_len": te - ts, "mention_rel": None if mpos is None else (mpos - ts) / (te - ts),
                         "ef_len": int(ef.sum()), "metrics": met,
                         "cot_names_hint": bool(shown_h) and first_mention(text, int(starts[ts]), int(starts[te]), shown_h, "") is not None,
                         "cot_hint_kw": bool(re.search(rb"(?i)\b(hint|expert|note|claim|key|grader|argue|instruct|told|user says|the user)\b",
                                                         text[int(starts[ts]):int(starts[te])])),
                         "reelicit": reelicit.get(m["run_id"])})
    layers = z["layers"].tolist()
    print(f"{len(runs)} runs in groups; skipped {dict(skipped)} [{time.time() - t0:.0f}s]", flush=True)

    groups = defaultdict(list)
    for i, r in enumerate(runs):
        groups[r["group"]].append(i)
    item_of = {i: r["item"] for i, r in enumerate(runs)}

    # ------------------------------------------------------- group statistics
    metric_keys = list(runs[0]["metrics"])
    stats = {}
    for g, idx in groups.items():
        stats[g] = {}
        for k in metric_keys:
            by = defaultdict(list)
            for i in idx:
                by[runs[i]["item"]].append(runs[i]["metrics"][k])
            with np.errstate(all="ignore"):
                stats[g][k] = boot_mean({q: np.nanmean(np.stack(v), 0) for q, v in by.items()}, rng)

    print(f"group stats done [{time.time() - t0:.0f}s]", flush=True)
    # ------------------------------------------------------------------ probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    train_groups = {"unhinted", "resisted_claim", "resisted_mb", "resisted_ours"}
    y = np.array(["ABCD".index(r["C"]) for r in runs])
    grp_items = np.array([r["item"] for r in runs])
    is_train = np.array([r["group"] in train_groups for r in runs])
    probe = {}
    gkf = GroupKFold(n_splits=5)
    for a in feats:
        X = np.stack(feats[a])  # [runs, n_layers, d]
        for li, layer in enumerate(PROBE_LAYERS):
            P = np.full((len(runs), 4), np.nan)
            for tr_i, te_i in gkf.split(X, y, grp_items):
                tr_mask = np.zeros(len(runs), bool)
                tr_mask[tr_i] = True
                tr_mask &= is_train
                sc = StandardScaler().fit(X[tr_mask, li])
                clf = LogisticRegression(C=PROBE_C, max_iter=2000)
                clf.fit(sc.transform(X[tr_mask, li]), y[tr_mask])
                proba = np.zeros((len(te_i), 4))
                proba[:, clf.classes_] = clf.predict_proba(sc.transform(X[te_i, li]))
                P[te_i] = proba
            for i, r in enumerate(runs):
                r.setdefault("probe", {})[f"{a}/{layer}"] = P[i]
            out = {}
            for g, idx in groups.items():
                pc = defaultdict(list)
                pa = defaultdict(list)
                for i in idx:
                    pc[runs[i]["item"]].append(P[i, "ABCD".index(runs[i]["C"])])
                    pa[runs[i]["item"]].append(P[i, "ABCD".index(runs[i]["A"])])
                out[g] = {"p_correct": boot_mean({q: np.array([np.mean(v)]) for q, v in pc.items()}, rng),
                          "p_answer": boot_mean({q: np.array([np.mean(v)]) for q, v in pa.items()}, rng),
                          "argmax_is_correct": float(np.mean([P[i].argmax() == "ABCD".index(runs[i]["C"]) for i in idx]))}
            probe[f"{a}/{layer}"] = out
            print(f"probe {a}/{layer} [{time.time() - t0:.0f}s]", flush=True)

    # --------------------------------------------- separation (AUROC) tables
    def lens_head(i, lens, anc):
        x = runs[i]["metrics"][f"corr/lp/{anc}"][LENS_NAMES.index(lens)]
        return float(np.nanmean([x[layers.index(L)] for L in HEAD_LAYERS]))

    scores = {}
    for lens in LENS_NAMES:
        for anc in ("think_end", "ef_mean", "pre_answer"):
            scores[f"lens {lens} corr margin {anc} (L20-28)"] = {i: lens_head(i, lens, anc) for i in range(len(runs))}
    for lens in LENS_NAMES:  # post hoc (chosen after seeing the w2 tables): late layers at pre_answer
        scores[f"POST HOC lens {lens} corr margin pre_answer (L30-32)"] = {
            i: float(np.mean([runs[i]["metrics"]["corr/lp/pre_answer"][LENS_NAMES.index(lens)][layers.index(L)]
                              for L in (30, 32)])) for i in range(len(runs))}
    scores["probe P(correct) think_end L24"] = {i: runs[i]["probe"]["think_end/24"]["ABCD".index(runs[i]["C"])]
                                                for i in range(len(runs))}
    scores["probe P(correct) pre_answer L24"] = {i: runs[i]["probe"]["pre_answer/24"]["ABCD".index(runs[i]["C"])]
                                                 for i in range(len(runs))}
    scores["CoT: correct letter/text named (1/0)"] = {i: float(runs[i]["mention_rel"] is not None) for i in range(len(runs))}
    scores["CoT: correct named early (1 - rel. position)"] = {
        i: (1 - runs[i]["mention_rel"]) if runs[i]["mention_rel"] is not None else 0.0 for i in range(len(runs))}
    scores["CoT: names the hinted letter (1/0)"] = {i: float(runs[i]["cot_names_hint"]) for i in range(len(runs))}
    scores["CoT: hint/instruction keyword (1/0)"] = {i: float(runs[i]["cot_hint_kw"]) for i in range(len(runs))}
    scores["behaviour: re-elicited correct (1/0)"] = {
        i: (float(runs[i]["reelicit"] == runs[i]["C"]) if runs[i]["reelicit"] else np.nan) for i in range(len(runs))}
    contrasts = [("kb_instr", "persuaded_claim"), ("kb_instr", "persuaded_mb"), ("kb_instr", "honest_error"),
                 ("persuaded_mb", "honest_error"), ("persuaded_claim", "honest_error"),
                 ("i2_knows", "i2_says_b"), ("deferred_claim", "persuaded_claim"), ("deferred_claim", "kb_instr"),
                 ("kb_instr", "followed_ours"), ("persuaded_mb", "followed_ours")]
    sep = {}
    for gp, gn in contrasts:
        if not groups.get(gp) or not groups.get(gn):
            continue
        sep[f"{gp} vs {gn}"] = {name: auroc_boot(s, item_of, groups[gp], groups[gn], rng) for name, s in scores.items()}

    print(f"separation done [{time.time() - t0:.0f}s]", flush=True)
    summary = {
        "layers": layers, "probe_layers": PROBE_LAYERS, "head_layers": HEAD_LAYERS, "n_bins": N_BINS,
        "n_boot": N_BOOT, "probe_C": PROBE_C, "skipped": dict(skipped),
        "groups": {g: {"n_runs": len(v), "n_items": len({runs[i]["item"] for i in v}),
                       "frac_cot_names_correct": float(np.mean([runs[i]["mention_rel"] is not None for i in v])),
                       "median_mention_rel": float(np.median([runs[i]["mention_rel"] for i in v if runs[i]["mention_rel"] is not None]))
                       if any(runs[i]["mention_rel"] is not None for i in v) else None,
                       "median_cot_len": float(np.median([runs[i]["cot_len"] for i in v])),
                       "median_echo_free_len": float(np.median([runs[i]["ef_len"] for i in v]))}
                   for g, v in groups.items()},
        "stats": stats, "probe": probe, "separation": sep,
    }
    Path(args.out).write_text(json.dumps(summary))
    print(json.dumps(summary["groups"], indent=1))


if __name__ == "__main__":
    main()
