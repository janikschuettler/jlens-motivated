"""Pre-registered repeat of the claimed-answer-key "deferred vs persuaded" lead (preregistration.md "R-CL").

V2 (grader_info) and V2' (chen_unauthorized) runs on fresh known items whose final answer is B.
Positive "deferred": the full-history re-elicitation (CoT kept, thinking off) names the correct
letter; negative "persuaded": it names B; other letters excluded (as in analyze_main).
Score (= analyze_main "POST HOC lens jlens corr margin pre_answer (L30-32)"): at pre_answer,
lp(correct) - mean lp(the two wrong letters other than B), letters = log-sum-exp over the tracked
forms, mean over stored layers 30 and 32. Primary: J-lens AUROC, 95% item-bootstrap interval
(analyze_main.auroc_boot, seed 0). Pass: lower bound > 0.5. Underpowered: smaller group < 15 runs.
Exploratory: per arm, reasoning length alone, within-item, template-mode grouping.

uv run --project envs/lens python scripts/lens/analyze_cl_repeat.py --out data/explore/clrep/claim_lead_repeat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_c3 import LENSES, letter  # noqa: E402
from analyze_main import auroc_boot, boot_mean  # noqa: E402

R = Path(__file__).resolve().parents[2]
D = R / "data"
MIN_RUNS = 15
SCORE_LAYERS = (30, 32)
ARMS = ("V2_info", "V2p_chen")
RDIR, LABELS, FORKS = ("readouts/clrep_followers", "explore/w2/grader_clrep/labels.jsonl",
                       "explore/w2/forks_reelicit_clrep/forks.jsonl")


def jsonl(p):
    return [json.loads(line) for line in open(p)]


def load(known: dict, orig_items: set) -> tuple[list[dict], dict]:
    labels = {d["run_id"]: d for d in jsonl(D / LABELS)}
    reel = defaultdict(dict)
    for d in jsonl(D / FORKS):
        if d.get("kind") == "reelicit":
            reel[d["mode"]][d["source_run_id"]] = d["argmax"]
    index = list({m["run_id"]: m for m in jsonl(D / RDIR / "index.jsonl")}.values())
    counts = defaultdict(int)
    out = []
    for m in index:
        arm = m["extras"].get("arm", m["condition"])
        if arm not in ARMS:
            raise ValueError(f"unexpected arm {arm}")
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        lab = labels[m["run_id"]]
        if A != H or H == C or lab["answer"] != H or lab["finish_reason"] != "stop":
            raise ValueError(f"not a follower: {m['run_id']}")
        if not (q in known and known[q] >= 0.9) or q in orig_items:
            raise ValueError(f"item not fresh/known: {q}")
        pa = m["named_positions"]["pre_answer"]
        if pa < 0:
            raise ValueError(f"no pre_answer position: {m['run_id']}")
        rf, rt = reel["full"].get(m["run_id"]), reel["template"].get(m["run_id"])
        if rf is None:
            raise ValueError(f"no full re-elicitation for {m['run_id']}")
        with np.load(D / RDIR / m["file"]) as z:
            layers = z["layers"].tolist()
            z = {k: z[k] for k in ("lp", "rank", "positions_lo")}
        lo = int(z["positions_lo"])
        tr = m["tracked"]
        lpL = {x: letter(z, tr, x) for x in m["letters"]}
        U = [x for x in m["letters"] if x not in {C, A}]
        if len(U) != 2:
            raise ValueError(f"|U| = {len(U)}: {m['run_id']}")
        M = lpL[C][0][:, :, pa - lo] - np.mean([lpL[x][0][:, :, pa - lo] for x in U], 0)  # [lens, layer]
        li = [layers.index(L) for L in SCORE_LAYERS]
        te = m["named_positions"]["think_end"]
        counts["followers"] += 1
        counts[f"full re-elicit = {'correct' if rf == C else 'B' if rf == H else 'other'}"] += 1
        if rf not in (C, H):
            continue
        out.append({"run_id": m["run_id"], "arm": arm, "item": q, "C": C, "B": H,
                    "S": {lens: float(M[i, li].mean()) for i, lens in enumerate(LENSES)},
                    "S_by_layer": {lens: dict(zip(layers, M[i].round(4).tolist())) for i, lens in enumerate(LENSES)},
                    "cot_len": (te - m["prompt_len"]) if te >= 0 else None,
                    "reel_full": rf, "reel_template": rt, "pos": rf == C})
    return out, dict(counts)


def test(runs: list[dict], rng, key="pos") -> dict:
    idx = list(range(len(runs)))
    items = {i: runs[i]["item"] for i in idx}
    pos = [i for i in idx if runs[i][key] is True]
    neg = [i for i in idx if runs[i][key] is False]
    res = {f"{lens} S": auroc_boot({i: runs[i]["S"][lens] for i in idx}, items, pos, neg, rng) for lens in LENSES}
    res["reasoning length"] = auroc_boot({i: float(runs[i]["cot_len"]) if runs[i]["cot_len"] is not None else np.nan
                                          for i in idx}, items, pos, neg, rng)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    known = {d["id"]: (d.get("acc") or {}).get("A") or 0 for d in jsonl(D / "explore/known_items.jsonl")}
    orig_items = {d["id"] for d in jsonl(D / "explore/known_items.jsonl") if ((d.get("acc") or {}).get("A") or 0) >= 0.9}
    fresh = {d["id"] for d in jsonl(D / "explore/c3rep_items.jsonl")}
    orig_items -= fresh  # the 40 wave2_part1 items
    runs, counts = load(known, orig_items)
    rng = np.random.default_rng(0)
    pos = [r for r in runs if r["pos"]]
    neg = [r for r in runs if not r["pos"]]
    primary = test(runs, rng)
    j = primary["jlens S"]
    underpowered = min(len(pos), len(neg)) < MIN_RUNS
    verdict = "underpowered" if underpowered else ("pass" if j["lo"] > 0.5 else "fail")

    gm = {}
    for g, sel in (("deferred", True), ("persuaded", False)):
        for lens in LENSES:
            by = defaultdict(list)
            for r in runs:
                if r["pos"] == sel:
                    by[r["item"]].append(r["S"][lens])
            gm[f"{g} / {lens}"] = boot_mean({k: np.array([np.mean(v)]) for k, v in by.items()}, rng)
    per_arm = {a: test([r for r in runs if r["arm"] == a], rng) for a in ARMS}
    within = {}
    for lens in LENSES:
        by = defaultdict(lambda: ([], []))
        for r in runs:
            by[r["item"]][0 if r["pos"] else 1].append(r["S"][lens])
        both = {k: v for k, v in by.items() if v[0] and v[1]}
        within[lens] = {"items_with_both": len(both),
                        "items_deferred_higher": int(sum(np.mean(p) > np.mean(n) for p, n in both.values())),
                        "mean_diff": round(float(np.mean([np.mean(p) - np.mean(n) for p, n in both.values()])), 4)
                        if both else None}
    for r in runs:
        r["pos_template"] = True if r["reel_template"] == r["C"] else False if r["reel_template"] == r["B"] else None
    res = {
        "preregistration": "docs/preregistration.md, R-CL",
        "score": "corr margin at pre_answer (correct minus mean of the two other wrong letters), mean of layers 30, 32",
        "counts": counts,
        "n": {"runs": len(runs), "items": len({r["item"] for r in runs}),
              "deferred": len(pos), "deferred_items": len({r["item"] for r in pos}),
              "persuaded": len(neg), "persuaded_items": len({r["item"] for r in neg})},
        "primary": primary, "verdict": verdict, "underpowered": underpowered,
        "group_means_S": gm,
        "exploratory": {"per_arm": per_arm, "within_item": within,
                        "template_grouping": test(runs, rng, key="pos_template")},
        "runs": [{k: r[k] for k in ("run_id", "arm", "item", "S", "S_by_layer", "cot_len", "reel_full",
                                    "reel_template", "pos")} for r in runs],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("counts", "n", "primary", "verdict", "group_means_S", "exploratory")}, indent=1))


if __name__ == "__main__":
    main()
