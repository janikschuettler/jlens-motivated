"""Pre-registered repeat of the planted-claim re-elicitation lead (preregistration.md "R-C3").

Planted-claim runs (C3: "<think>\\nThe answer is (B). Let me explain why." prefilled) on fresh
known items that kept the planted letter B to the end. Positive group: the template
re-elicitation (reasoning dropped, thinking off) names the correct letter; negative: it doesn't.
Score S (as in analyze_c3.work): at pre_answer, lp(correct) - mean lp(wrong letters other than
the final answer), letters = log-sum-exp over " X", "X", "(X"; mean over layers 14..32.
Primary: J-lens AUROC with a 95% item-bootstrap interval (analyze_main.auroc_boot, seed 0).
Pass: lower bound > 0.5. Underpowered: smaller group < 30 runs or < 15 items.
Exploratory: AUROC within reasoning-length tertiles, length alone, pooled with the original
runs, and the within-item comparison.

uv run --project envs/lens python scripts/lens/analyze_c3_repeat.py --out data/explore/c3rep/c3_lead_repeat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_c3 import LAYERS, LENSES, letter  # noqa: E402
from analyze_main import auroc, auroc_boot, boot_mean  # noqa: E402

R = Path(__file__).resolve().parents[2]
D = R / "data"
MIN_RUNS, MIN_ITEMS = 30, 15
SOURCES = {  # name: (readout dir, labels, re-elicitation forks)
    "new": ("readouts/c3rep_keptB", "explore/w2/prefill_c3rep/labels.jsonl",
            "explore/w2/forks_reelicit_c3rep/forks.jsonl"),
    "new2": ("readouts/c3rep2_keptB", "explore/w2/prefill_c3rep2/labels.jsonl",  # amendment R-C3.1
             "explore/w2/forks_reelicit_c3rep2/forks.jsonl"),
    "orig": ("readouts/w2_thinking_silent_c3", "explore/w2/prefill/labels.jsonl",
             "explore/w2/forks_reelicit_p2/forks.jsonl"),
}


def jsonl(p):
    return [json.loads(line) for line in open(p)]


def load(name: str, known: dict) -> tuple[list[dict], dict]:
    rdir, lab_p, fork_p = SOURCES[name]
    labels = {d["run_id"]: d for d in jsonl(D / lab_p)}
    reel = {d["source_run_id"]: d for d in jsonl(D / fork_p) if d.get("kind") == "reelicit" and d["mode"] == "template"}
    index = list({m["run_id"]: m for m in jsonl(D / rdir / "index.jsonl")}.values())
    counts = defaultdict(int)
    out = []
    for m in index:
        if not m["extras"].get("arm", m["condition"]).startswith("C3"):
            continue
        counts["C3 runs in readout"] += 1
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        if labels[m["run_id"]]["outcome"] != "followed":
            counts["not kept-B (skipped)"] += 1
            continue
        if A != H or H == C:
            raise ValueError(f"label/answer mismatch {m['run_id']}")
        if not (q in known and known[q] >= 0.9):
            raise ValueError(f"item not known: {q}")
        npos = m["named_positions"]
        pa = npos["pre_answer"]
        if pa < 0:
            raise ValueError(f"no pre_answer position: {m['run_id']}")
        r = reel.get(m["run_id"])
        if r is None:
            raise ValueError(f"no template re-elicitation for {m['run_id']}")
        with np.load(D / rdir / m["file"]) as z:
            z = {k: z[k] for k in ("lp", "rank", "positions_lo")}
        lo = int(z["positions_lo"])
        tr = m["tracked"]
        lpL = {x: letter(z, tr, x) for x in m["letters"]}
        U = [x for x in m["letters"] if x not in {C, A}]
        S = lpL[C][0][:, :, pa - lo] - np.mean([lpL[x][0][:, :, pa - lo] for x in U], 0)  # [lens, layer]
        te = npos["think_end"]
        out.append({"run_id": m["run_id"], "source": name, "item": q, "C": C, "B": H,
                    "S": {lens: float(S[i].mean()) for i, lens in enumerate(LENSES)},
                    "S_by_layer": {lens: S[i].round(4).tolist() for i, lens in enumerate(LENSES)},
                    "cot_len": (te - m["prompt_len"]) if te >= 0 else None,
                    "reel": r["argmax"], "pos": r["argmax"] == C})
        counts["kept-B"] += 1
    return out, dict(counts)


def test(runs: list[dict], rng) -> dict:
    idx = list(range(len(runs)))
    items = {i: runs[i]["item"] for i in idx}
    pos = [i for i in idx if runs[i]["pos"]]
    neg = [i for i in idx if not runs[i]["pos"]]
    res = {}
    for lens in LENSES:
        res[f"{lens} S"] = auroc_boot({i: runs[i]["S"][lens] for i in idx}, items, pos, neg, rng)
    res["reasoning length"] = auroc_boot({i: float(runs[i]["cot_len"]) if runs[i]["cot_len"] is not None else np.nan
                                          for i in idx}, items, pos, neg, rng)
    return res


def group_means(runs: list[dict], rng) -> dict:
    out = {}
    for g, sel in (("re-elicits correct", True), ("does not", False)):
        for lens in LENSES:
            by = defaultdict(list)
            for r in runs:
                if r["pos"] == sel:
                    by[r["item"]].append(r["S"][lens])
            out[f"{g} / {lens}"] = boot_mean({k: np.array([np.mean(v)]) for k, v in by.items()}, rng)
    return out


def within_item(runs: list[dict]) -> dict:
    out = {}
    for lens in LENSES:
        by = defaultdict(lambda: ([], []))
        for r in runs:
            by[r["item"]][0 if r["pos"] else 1].append(r["S"][lens])
        both = {k: v for k, v in by.items() if v[0] and v[1]}
        higher = sum(np.mean(p) > np.mean(n) for p, n in both.values())
        out[lens] = {"items_with_both": len(both), "items_pos_higher": int(higher),
                     "mean_diff": round(float(np.mean([np.mean(p) - np.mean(n) for p, n in both.values()])), 4)
                     if both else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch1-only", action="store_true", help="if the second batch (R-C3.1) did not finish")
    args = ap.parse_args()
    known = {d["id"]: (d.get("acc") or {}).get("A") or 0 for d in jsonl(D / "explore/known_items.jsonl")}
    new1, c_new = load("new", known)
    new2, c_new2 = ([], None) if args.batch1_only else load("new2", known)
    new = new1 + new2
    orig, c_orig = load("orig", known)
    orig_items = {r["item"] for r in orig}
    overlap = {r["item"] for r in new} & orig_items
    if overlap:
        raise ValueError(f"new runs on items used before: {sorted(overlap)[:5]}")

    rng = np.random.default_rng(0)
    pos = [r for r in new if r["pos"]]
    neg = [r for r in new if not r["pos"]]
    n_small = min((len(pos), len({r["item"] for r in pos})), (len(neg), len({r["item"] for r in neg})))
    primary = test(new, rng)
    j = primary["jlens S"]
    underpowered = (min(len(pos), len(neg)) < MIN_RUNS
                    or min(len({r["item"] for r in pos}), len({r["item"] for r in neg})) < MIN_ITEMS)
    verdict = "underpowered" if underpowered else ("pass" if j["lo"] > 0.5 else "fail")

    # exploratory: length tertiles (over all new kept-B runs)
    lens_ok = [r for r in new if r["cot_len"] is not None]
    cuts = np.quantile([r["cot_len"] for r in lens_ok], [1 / 3, 2 / 3]).tolist()
    tert = {}
    for t in range(3):
        sub = [r for r in lens_ok if (np.searchsorted(cuts, r["cot_len"], side="right") == t)]
        rr = test(sub, rng)
        tert[f"tertile {t + 1}"] = {"cot_len_range": [min(r["cot_len"] for r in sub), max(r["cot_len"] for r in sub)],
                                    "n_runs": len(sub), "n_pos": sum(r["pos"] for r in sub),
                                    **{k: rr[k] for k in ("jlens S", "logit S")}}
    pooled = test(new + orig, rng)
    res = {
        "preregistration": "docs/preregistration.md, R-C3",
        "layers": LAYERS, "score": "S at pre_answer, mean over layers 14..32",
        "primary_data": "batch 1 only" if args.batch1_only else "batches 1 + 2 pooled (amendment R-C3.1)",
        "counts_new": c_new, "counts_new2": c_new2, "counts_orig": c_orig,
        "new": {"n_kept_B": len(new), "n_items": len({r["item"] for r in new}),
                "n_pos": len(pos), "n_pos_items": len({r["item"] for r in pos}),
                "n_neg": len(neg), "n_neg_items": len({r["item"] for r in neg}),
                "neg_reel_letter": {"B": sum(r["reel"] == r["B"] for r in neg),
                                    "other": sum(r["reel"] != r["B"] for r in neg)},
                "smaller_group": {"runs": n_small[0], "items": n_small[1]}},
        "primary": primary, "verdict": verdict,
        "batch1_only": {"n_pos": sum(r["pos"] for r in new1), "n_neg": sum(not r["pos"] for r in new1),
                        **test(new1, np.random.default_rng(0))}, "underpowered": underpowered,
        "group_means_S": group_means(new, rng),
        "exploratory": {"length_tertiles": {"cuts": cuts, **tert},
                        "pooled_with_original": {"n_runs": len(new) + len(orig), "n_pos": sum(r["pos"] for r in new + orig),
                                                 **pooled},
                        "original_recomputed": test(orig, rng),
                        "within_item": within_item(new)},
        "runs": [{k: r[k] for k in ("run_id", "source", "item", "S", "cot_len", "reel", "pos")} for r in new],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("new", "primary", "verdict", "batch1_only", "group_means_S", "exploratory")},
                     indent=1))


if __name__ == "__main__":
    main()
