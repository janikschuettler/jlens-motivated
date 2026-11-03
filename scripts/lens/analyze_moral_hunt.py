"""Moral hunt, quick exploratory lens table (lens env, CPU; overnight, beyond MMLU-Redux; no claims).

Octavia's readout of w2_moral_nudge (copied read-only), arm mb_expert, thinking, known items
(octavia's moral_known.jsonl, acc >= 0.9), wrong-letter hint. Groups: followed (final = hinted
letter) vs resisted (final = correct).

- Redux moral items (4 options): score S as pre-registered (docs/overnight/preregistration.md):
  at pre_answer, log-prob of the correct letter minus the mean log-prob of the unchosen wrong
  letters other than the final answer, mean over layers 14..32; J-lens and logit lens.
  AUROC(S; followed vs resisted), items resampled.
- MoralChoice (2 options): S is undefined for followed runs (no wrong letter other than the
  answer). Descriptive only: log-prob of the unchosen letter at pre_answer (correct letter in
  followed runs, hinted letter in resisted runs), layer mean, per lens.

uv run --project envs/lens python scripts/lens/analyze_moral_hunt.py --rdir data/explore/moral_hunt/readouts_octavia/w2_moral_nudge
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_main import auroc_boot, boot_mean  # noqa: E402
from analyze_prereg import letter_series, load_set, score_run  # noqa: E402

from jmr import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--pool", default="data/explore/moral_hunt/octavia/moral_known.jsonl")
    ap.add_argument("--out", default="data/explore/moral_hunt/lens_table.json")
    args = ap.parse_args()
    R = config.REPO_ROOT
    rdir = R / args.rdir
    pool = {}
    for line in open(R / args.pool):
        r = json.loads(line)
        pool[r["id"]] = r["acc"]["A"]
    rng = np.random.default_rng(0)
    runs = []
    for m in load_set(rdir):
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        if (m.get("extras") or {}).get("arm") != "mb_expert" or not m["enable_thinking"]:
            continue
        if (pool.get(q) or 0) < 0.9 or H in (None, C) or A not in (C, H):
            continue
        z = np.load(rdir / m["file"])
        pa = m["named_positions"]["pre_answer"]
        if pa < 0:
            continue
        lo = int(z["positions_lo"])
        group = "followed" if A == H else "resisted"
        unchosen = C if A == H else H
        lp_un = letter_series(z, m, unchosen)[0][:, :, pa - lo].mean(-1)  # [2]
        S, L, sal = None, None, None
        if len(m["letters"]) > 2:
            sc = score_run(z, m)
            S, L = sc["S"], sc["layer_margin"]  # [2], [2, 10]
            if group == "resisted":  # salience control as in lens_main: hint letter vs the other wrong letters
                lpH = letter_series(z, m, H)[0][:, :, pa - lo]
                lpO = np.mean([letter_series(z, m, x)[0][:, :, pa - lo] for x in m["letters"] if x not in {C, H}], 0)
                sal = lpH - lpO
        runs.append({"item": q, "mc": q.startswith("moralchoice/"), "group": group, "S": S, "L": L, "sal": sal,
                     "lp_unchosen": lp_un})
    res = {"note": "exploratory, beyond MMLU-Redux, no claims", "counts": {}, "redux_moral_S": {}, "moralchoice_unchosen_lp": {}}
    for mc, tag in ((False, "redux_moral"), (True, "moralchoice")):
        sel = [r for r in runs if r["mc"] == mc]
        res["counts"][tag] = {g: {"runs": sum(r["group"] == g for r in sel), "items": len({r["item"] for r in sel if r["group"] == g})}
                              for g in ("followed", "resisted")}
    red = [r for r in runs if not r["mc"]]
    items = {i: r["item"] for i, r in enumerate(red)}
    f = [i for i, r in enumerate(red) if r["group"] == "followed"]
    s = [i for i, r in enumerate(red) if r["group"] == "resisted"]
    for li, lens in enumerate(("jlens", "logit")):
        res["redux_moral_S"][lens] = {"auroc_followed_vs_resisted": auroc_boot({i: float(r["S"][li]) for i, r in enumerate(red)}, items, f, s, rng)}
        for g, ii in (("followed", f), ("resisted", s)):
            by = defaultdict(list)
            for i in ii:
                by[red[i]["item"]].append(red[i]["S"][li])
            res["redux_moral_S"][lens][f"S_mean_{g}"] = boot_mean({q: np.mean(v) for q, v in by.items()}, rng)
    res["redux_moral_layers_30_32_post_hoc"] = {}
    for li, lens in enumerate(("jlens", "logit")):
        out = {}
        for k, layer in ((8, 30), (9, 32)):
            by = defaultdict(list)
            for i in f:
                by[red[i]["item"]].append(red[i]["L"][li][k])
            out[f"followed_correct_margin_L{layer}"] = boot_mean({q: np.mean(v) for q, v in by.items()}, rng)
            by = defaultdict(list)
            for i in s:
                by[red[i]["item"]].append(red[i]["sal"][li][k])
            out[f"resisted_hint_margin_L{layer}"] = boot_mean({q: np.mean(v) for q, v in by.items()}, rng)
        res["redux_moral_layers_30_32_post_hoc"][lens] = out
    mcs = [r for r in runs if r["mc"]]
    for li, lens in enumerate(("jlens", "logit")):
        res["moralchoice_unchosen_lp"][lens] = {}
        for g in ("followed", "resisted"):
            by = defaultdict(list)
            for r in mcs:
                if r["group"] == g:
                    by[r["item"]].append(r["lp_unchosen"][li])
            res["moralchoice_unchosen_lp"][lens][g] = boot_mean({q: np.mean(v) for q, v in by.items()}, rng)
    (R / args.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
