"""Lens analysis for gpt-oss-20b in the models x grader-bug study (numpy, CPU; any env).

  python scripts/lens/analyze_mg_gptoss.py --readouts data/readouts/mg_gptoss \
      --runs data/explore/models_grader/gptoss_arms/readout_runs.jsonl \
      --concepts configs/concept_tokens_grader_gptoss.json --out data/readouts/mg_gptoss_analysis.json

Same measures as scripts/lens/analyze_models_grader.py (fixed in its docstring and in
docs/experiments/models-grader/lens_plan.md), with the concept-token file as an argument and
cross-arm contrasts (same action, different permission):
  L_allowed/followed vs A_V0/followed and vs A_sys/followed (AUROC, item bootstrap).
Within-arm: followed vs resisted in A_V0 and A_sys. Per measure and lens, the score is the mean
over the workspace layers. Derived numbers only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_models_grader import LENSES, boot_auroc, boot_mean, lse  # noqa: E402

PAIRS = [("L_allowed/followed", "A_V0/followed"), ("L_allowed/followed", "A_sys/followed"),
         ("A_V0/followed", "A_V0/resisted"), ("A_sys/followed", "A_sys/resisted"),
         ("A_V0/resisted", "A_V1_null/resisted"),
         ("A_V0/followed:named_only", "A_V0/followed:open"), ("A_sys/followed:named_only", "A_sys/followed:open"),
         ("A_V0/followed:named_only", "A_V0/resisted")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subgroups", default=None,
                    help="follower_kw.jsonl (scripts/explore/mg_gptoss_followers.py): followers are also "
                         "counted under '<arm>/followed:<category>'")
    a = ap.parse_args()
    sub = {}
    if a.subgroups:
        for line in open(a.subgroups):
            x = json.loads(line)
            sub[x["run_id"]] = x["category"]
    rng = random.Random(0)
    R = Path(a.readouts)
    arm_of = {}
    for line in open(a.runs):
        r = json.loads(line)
        arm_of[r["run_id"]] = r.get("arm") or r["run_id"].split("/")[1]
    gid = {t["id"]: t["group"] for t in json.loads(Path(a.concepts).read_text())["tokens"]}
    groups_c = sorted(set(gid.values()))
    rows = defaultdict(lambda: defaultdict(list))  # "arm/group" -> measure -> [(item, [2, L])]
    layers, n_skip = None, defaultdict(int)
    for line in open(R / "index.jsonl"):
        m = json.loads(line)
        arm = arm_of.get(m["run_id"])
        C, B, Ans = m["correct_letter"], m["hint_letter"], m["answer"]
        if arm is None or B is None or Ans is None:
            n_skip["no arm / letter / answer"] += 1
            continue
        group = "followed" if Ans == B else "resisted" if Ans == C else None
        if group is None:
            n_skip["other wrong"] += 1
            continue
        keys = [f"{arm}/{group}"]
        if group == "followed" and m["run_id"] in sub:
            keys.append(f"{arm}/followed:{sub[m['run_id']]}")
        z = np.load(R / m["file"])
        layers = z["layers"].tolist()
        tr = m["tracked"]
        nf = len(tr["forms"])
        lp = z["lp"].astype(np.float64)
        lo = int(z["positions_lo"])

        def letter(Lx):
            k = tr["letters"].index(Lx) * nf
            return lse(lp[..., k:k + nf])

        U = [x for x in m["letters"] if x not in (C, B)]
        lpU = np.mean([letter(x) for x in U], 0)
        corr, sal = letter(C) - lpU, letter(B) - lpU
        npos, q = m["named_positions"], m["question_id"]
        for anc in ("pre_answer", "think_end", "last_prompt"):
            p = npos.get(anc, -1)
            if p is None or p < 0:
                continue
            for key in keys:
                rows[key][f"corr@{anc}"].append((q, corr[:, :, p - lo]))
                rows[key][f"sal@{anc}"].append((q, sal[:, :, p - lo]))
        if "x_ids" in z.files:
            cidx = np.where(z["x_opt"] == -1)[0]
            xlp = z["x_lp"].astype(np.float64)
            span = m["cot_span"]
            for g in groups_c:
                k = [int(i) for i in cidx if gid[int(z["x_ids"][i])] == g]
                s = lse(xlp[..., k])
                for anc in ("pre_answer", "think_end", "last_prompt"):
                    p = npos.get(anc, -1)
                    if p is not None and p >= 0:
                        for key in keys:
                            rows[key][f"concept_{g}@{anc}"].append((q, s[:, :, p - lo]))
                if span and span[1] > span[0]:
                    for key in keys:
                        rows[key][f"concept_{g}@cot_mean"].append((q, s[:, :, span[0] - lo:span[1] - lo].mean(-1)))
    out = {"layers": layers, "lenses": list(LENSES), "skipped": dict(n_skip), "groups": {}, "contrasts": {}}
    for key, meas in rows.items():
        og = out["groups"][key] = {"n": len(meas.get("corr@pre_answer", []))}
        for name, lst in meas.items():
            by = defaultdict(list)
            for q, v in lst:
                by[q].append(v)
            og[name] = boot_mean(by, rng)
    for g1, g2 in PAIRS:
        if g1 not in rows or g2 not in rows:
            continue
        oc = out["contrasts"][f"{g1} vs {g2}"] = {}
        for name in rows[g1]:
            if not (name.startswith("corr@") or name.startswith("concept_")):
                continue
            for li, lens in enumerate(LENSES):
                pb, nb = defaultdict(list), defaultdict(list)
                for q, v in rows[g1][name]:
                    pb[q].append(float(np.mean(v[li])))
                for q, v in rows[g2].get(name, []):
                    nb[q].append(float(np.mean(v[li])))
                if nb:
                    oc[f"{name}/{lens}"] = boot_auroc(pb, nb, rng)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for key, og in sorted(out["groups"].items()):
        s = og.get("corr@pre_answer")
        sl = og.get("sal@pre_answer")
        f = lambda d: "-" if not d else f"J {np.mean(d['mean'][0]):+.2f} logit {np.mean(d['mean'][1]):+.2f}"
        print(f"{key:24s} n={og['n']:4d}  corr@pre_answer {f(s)}   sal@pre_answer {f(sl)}")
    for c, d in out["contrasts"].items():
        for k, v in d.items():
            if v["auroc"] == v["auroc"] and ("corr@pre_answer" in k or "@last_prompt" in k or "cot_mean" in k):
                print(f"  {c} | {k}: {v['auroc']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] ({v['n_pos']} vs {v['n_neg']})")


if __name__ == "__main__":
    main()
