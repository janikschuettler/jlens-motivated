"""Non-thinking section of the main lens analysis (overnight 2026-11-03; lens
env, CPU). Silent vs open switches vs resisted vs unhinted, on items known in
setting B (>= 9/10 correct in the gap screen's nothink_mb_unhinted_mb).

Runs: gap_factorial non-thinking cells (factorial_nothink: M&B-hint and
ours-hint cells, 4 each, plus 3 unhinted cells) and screen setting B (2
samples per known item, the unhinted cell matching nothink_mb_mb_expert_mb).
Switch = final answer = the shown hint B. Silent = the gap agent's keyword
check found no hint mention in the completion (labels.jsonl "mentions" empty);
open = it found one. The reasoning span is the response up to the final
answer (no think block). Measures and controls as in analyze_main.py (form
"any"; margins against the unchosen wrong letters; echo-free = response
positions before the first naming of the correct letter or its option text).

Score fixed in advance (orchestrator's definition, before looking): correct-letter
log-prob minus the mean of the other wrong letters (neither correct nor the hint)
at pre_answer, averaged over the workspace layers 14-32, per lens; plus per layer.

Probe: residuals at last_prompt and pre_answer only (no think block). Trained
on unhinted + resisted runs (label = correct letter = their answer), 5-fold
GroupKFold by item. last_prompt is identical for all samples of one prompt.

uv run --project envs/lens python scripts/lens/analyze_nothink.py --out data/readouts/lens_nothink.json
"""

import argparse
import bisect
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_main import LENS_NAMES, N_BINS, PROBE_C, PROBE_LAYERS, SEED, auroc_boot, boot_mean, first_mention  # noqa: E402

from jmr import config  # noqa: E402

SETS = {
    "readouts/gap_factorial_factorial_nothink": "explore/gap_factorial/factorial_nothink",
    "readouts/gap_factorial_screen_B_lens": "explore/gap_factorial/screen_B_lens",
}
LABELS = ["explore/gap_factorial/factorial_nothink/labels.jsonl", "explore/gap_factorial/screen/labels.jsonl"]
COUNTS_B = "explore/gap_factorial/screen_B_lens/n_correct_B.json"
ITEMS = "explore/gap_factorial/factorial_nothink/items.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--examples", type=int, default=6, help="silent switches to print for reading")
    args = ap.parse_args()
    D = Path(args.data)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    from transformers import AutoTokenizer
    from jmr.tokens import token_bytes
    mcfg = config.load("model")
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])

    known = {q for q, k in json.load(open(D / COUNTS_B)).items() if k >= 9}
    items = {json.loads(line)["id"]: json.loads(line) for line in open(D / ITEMS)}
    labels = {}
    for p in LABELS:
        for line in open(D / p):
            d = json.loads(line)
            labels[d["run_id"]] = d

    runs, feats, skipped = [], {"last_prompt": [], "pre_answer": []}, defaultdict(int)
    texts = {}
    for rdir, sdir in SETS.items():
        if not (D / rdir / "index.jsonl").exists():
            print(f"WARNING: no readouts in {rdir}; skipped", flush=True)
            continue
        index = list({m["run_id"]: m for m in map(json.loads, open(D / rdir / "index.jsonl"))}.values())
        comp = {}
        for line in open(D / sdir / "runs.jsonl"):
            d = json.loads(line)
            comp[d["run_id"]] = (d["prompt_token_ids"] + d["completion_token_ids"], d["completion_text"])
        for m in index:
            q, C, A = m["question_id"], m["correct_letter"], m["answer"]
            arm = m["extras"].get("arm", m["condition"])
            if m["enable_thinking"]:
                skipped["thinking run"] += 1
                continue
            if A is None or m["named_positions"]["pre_answer"] < 0 or m["cot_span"] is None:
                skipped["no answer"] += 1
                continue
            if q not in known:
                skipped["not known in setting B"] += 1
                continue
            lab = labels.get(m["run_id"], {})
            unh = "unhinted" in arm
            H = None if unh else m["hint_letter"]
            fam = None if unh else ("mb" if "_mb_expert_" in arm else "ours" if "_ours_expert_" in arm else None)
            g = None
            if unh:
                g = "unhinted" if A == C else None
            elif A == H:
                g = f"{'open' if lab.get('mentions') else 'silent'}_switch_{fam}"
            elif A == C:
                g = f"resisted_{fam}"
            if g is None:
                skipped["not in a group"] += 1
                continue
            item_h = items[q]["hinted"]
            sal = C if A != C else (H if H else item_h)
            U = [x for x in m["letters"] if x not in {C, A, sal}]
            if len(U) != 2:
                skipped[f"|U| = {len(U)}"] += 1
                continue
            z = np.load(D / rdir / m["file"])
            lo = int(z["positions_lo"])
            tr = m["tracked"]
            nf = len(tr["forms"])
            lp_all, rank_all = z["lp"].astype(np.float64), z["rank"].astype(np.float64)

            def letter_lp(Lx):
                k = tr["letters"].index(Lx) * nf
                return np.logaddexp.reduce(lp_all[..., k:k + nf], axis=-1)

            def letter_lr(Lx):
                k = tr["letters"].index(Lx) * nf
                return np.log10(rank_all[..., k:k + nf].min(-1))

            lpU = np.mean([letter_lp(x) for x in U], 0)
            lrU = np.mean([letter_lr(x) for x in U], 0)
            series = {"corr/lp": letter_lp(C) - lpU, "corr/lr": lrU - letter_lr(C),
                      "sal/lp": letter_lp(sal) - lpU, "sal/lr": lrU - letter_lr(sal)}
            npos = m["named_positions"]
            ts, te = m["cot_span"]  # response [P, pre_answer + 1)
            ids, ctext = comp[m["run_id"]]
            pieces = token_bytes(tok, ids)
            starts = np.cumsum([0] + [len(p) for p in pieces[:-1]]).tolist()
            text = b"".join(pieces)
            opt = items[q]["options"][m["letters"].index(C)]
            fm = first_mention(text, starts[ts], starts[te - 1], C, opt)
            mpos = None if fm is None else bisect.bisect_right(starts, fm) - 1
            ef_end = te if mpos is None else max(ts, mpos - 1)
            pos = np.arange(ts, te)
            b = np.minimum(((pos - ts) * N_BINS) // (te - ts), N_BINS - 1)
            ef = pos < ef_end
            met = {}
            for key, s in series.items():
                for anc in ("last_prompt", "pre_answer"):
                    met[f"{key}/{anc}"] = s[:, :, npos[anc] - lo]
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
            names = m["resid_anchor_names"]
            for a in feats:
                feats[a].append(z["resid"][PROBE_LAYERS, names.index(a)].astype(np.float32))
            if g.startswith("silent"):
                texts[m["run_id"]] = ctext
            runs.append({"run_id": m["run_id"], "item": q, "group": g, "arm": arm, "C": C, "A": A, "H": H,
                         "metrics": met, "mention_rel": None if mpos is None else (mpos - ts) / (te - ts),
                         "ef_len": int(ef.sum()), "resp_len": te - ts, "kw_mention": bool(lab.get("mentions"))})
    layers = z["layers"].tolist()
    print(f"{len(runs)} runs in groups; skipped {dict(skipped)} [{time.time() - t0:.0f}s]", flush=True)
    groups = defaultdict(list)
    for i, r in enumerate(runs):
        groups[r["group"]].append(i)
    for g in list(groups):  # pooled over the two hint wordings
        for kind in ("silent_switch", "open_switch", "resisted"):
            if g.startswith(kind):
                groups[f"{kind}_all"] += groups[g]
    item_of = {i: r["item"] for i, r in enumerate(runs)}

    stats = {}
    for g, idx in groups.items():
        stats[g] = {}
        for k in runs[0]["metrics"]:
            by = defaultdict(list)
            for i in idx:
                by[runs[i]["item"]].append(runs[i]["metrics"][k])
            with np.errstate(all="ignore"):
                stats[g][k] = boot_mean({q: np.nanmean(np.stack(v), 0) for q, v in by.items()}, rng)
    print(f"stats [{time.time() - t0:.0f}s]", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    y = np.array(["ABCD".index(r["C"]) for r in runs])
    gi = np.array([r["item"] for r in runs])
    is_train = np.array([r["group"] in ("unhinted",) or r["group"].startswith("resisted") for r in runs])
    probe = {}
    for a in feats:
        X = np.stack(feats[a])
        for li, layer in enumerate(PROBE_LAYERS):
            P = np.full((len(runs), 4), np.nan)
            for tr_i, te_i in GroupKFold(n_splits=5).split(X, y, gi):
                msk = np.zeros(len(runs), bool)
                msk[tr_i] = True
                msk &= is_train
                sc = StandardScaler().fit(X[msk, li])
                clf = LogisticRegression(C=PROBE_C, max_iter=2000).fit(sc.transform(X[msk, li]), y[msk])
                pr = np.zeros((len(te_i), 4))
                pr[:, clf.classes_] = clf.predict_proba(sc.transform(X[te_i, li]))
                P[te_i] = pr
            for i, r in enumerate(runs):
                r.setdefault("probe", {})[f"{a}/{layer}"] = P[i]
            out = {}
            for g, idx in groups.items():
                pc, pa = defaultdict(list), defaultdict(list)
                for i in idx:
                    pc[runs[i]["item"]].append(P[i, "ABCD".index(runs[i]["C"])])
                    pa[runs[i]["item"]].append(P[i, "ABCD".index(runs[i]["A"])])
                out[g] = {"p_correct": boot_mean({q: np.array([np.mean(v)]) for q, v in pc.items()}, rng),
                          "p_answer": boot_mean({q: np.array([np.mean(v)]) for q, v in pa.items()}, rng)}
            probe[f"{a}/{layer}"] = out
    print(f"probe [{time.time() - t0:.0f}s]", flush=True)

    scores = {}
    for lens in LENS_NAMES:
        li_ = LENS_NAMES.index(lens)
        scores[f"PRE-DEFINED lens {lens} corr margin pre_answer (mean L14-32)"] = {
            i: float(np.mean(runs[i]["metrics"]["corr/lp/pre_answer"][li_])) for i in range(len(runs))}
        for L in layers:
            scores[f"lens {lens} corr margin pre_answer L{L}"] = {
                i: float(runs[i]["metrics"]["corr/lp/pre_answer"][li_][layers.index(L)]) for i in range(len(runs))}
        scores[f"lens {lens} corr margin echo-free mean (mean L14-32)"] = {
            i: float(np.nanmean(runs[i]["metrics"]["corr/lp/ef_mean"][li_])) for i in range(len(runs))}
    for a in feats:
        scores[f"probe P(correct) {a} L24"] = {i: runs[i]["probe"][f"{a}/24"]["ABCD".index(runs[i]["C"])]
                                               for i in range(len(runs))}
    scores["CoT: hint keyword mention (1/0)"] = {i: float(runs[i]["kw_mention"]) for i in range(len(runs))}
    scores["CoT: correct letter/text named (1/0)"] = {i: float(runs[i]["mention_rel"] is not None) for i in range(len(runs))}
    scores["CoT: correct named early (1 - rel. position)"] = {
        i: (1 - runs[i]["mention_rel"]) if runs[i]["mention_rel"] is not None else 0.0 for i in range(len(runs))}
    sep = {}
    for fam in ("mb", "ours", "all"):
        for gp, gn in ((f"silent_switch_{fam}", f"resisted_{fam}"), (f"silent_switch_{fam}", f"open_switch_{fam}"),
                       (f"open_switch_{fam}", f"resisted_{fam}")):
            if groups.get(gp) and groups.get(gn):
                sep[f"{gp} vs {gn}"] = {n: auroc_boot(s, item_of, groups[gp], groups[gn], rng) for n, s in scores.items()}
    print(f"separation [{time.time() - t0:.0f}s]", flush=True)

    ex = [runs[i] for i in groups.get("silent_switch_mb", [])]
    rng.shuffle(ex)
    examples = []
    for r in ex[: args.examples]:
        t = texts[r["run_id"]]
        examples.append({"run_id": r["run_id"], "correct": r["C"], "hint": r["H"], "answer": r["A"],
                         "chars": len(t), "text": t})
    Path(args.out.replace(".json", "_silent_examples.json")).write_text(json.dumps(examples, indent=1))
    summary = {"layers": layers, "probe_layers": PROBE_LAYERS, "n_bins": N_BINS, "skipped": dict(skipped),
               "n_known_B": len(known),
               "groups": {g: {"n_runs": len(v), "n_items": len({runs[i]["item"] for i in v}),
                              "frac_names_correct": float(np.mean([runs[i]["mention_rel"] is not None for i in v])),
                              "median_resp_len": float(np.median([runs[i]["resp_len"] for i in v])),
                              "median_echo_free_len": float(np.median([runs[i]["ef_len"] for i in v]))}
                          for g, v in groups.items()},
               "stats": stats, "probe": probe, "separation": sep}
    Path(args.out).write_text(json.dumps(summary))
    print(json.dumps(summary["groups"], indent=1))


if __name__ == "__main__":
    main()
