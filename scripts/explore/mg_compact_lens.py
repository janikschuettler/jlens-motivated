"""Compact derived lens numbers for docs/overnight/data/models_grader.json (models x grader-bug).

  python scripts/explore/mg_compact_lens.py RAW.json OUT.json --model ... --readout ... [--checks JSON]
RAW = scripts/lens/analyze_models_grader.py output. Keeps group means (S = mean over workspace layers,
late = last three) with per-layer means/CIs, the difference bootstraps and all pooled AUROCs.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--readout", required=True)
    ap.add_argument("--checks", default="{}")
    a = ap.parse_args()
    d = json.load(open(a.raw))
    P = d["pooled"]
    out = {"model": a.model, "layers": d["layers"], "readout": a.readout, "checks": json.loads(a.checks),
           "groups": {}, "diffs": d["diffs"], "auroc": {}}
    for g in ["unsanctioned_followed", "unsanctioned_resisted", "allowed_followed", "null_correct"]:
        for meas in ["corr@pre_answer", "sal@pre_answer", "concept_grader@pre_answer", "concept_grader@cot_mean",
                     "concept_grader@last_prompt"]:
            v = P.get(f"{g}/{meas}")
            if not v:
                continue
            m = np.array(v["mean"])
            out["groups"][f"{g}/{meas}"] = {
                "n": v["n"], "items": v["items"],
                "jlens": {"S": round(float(m[0].mean()), 3), "late": round(float(m[0][-3:].mean()), 3)},
                "logit": {"S": round(float(m[1].mean()), 3), "late": round(float(m[1][-3:].mean()), 3)},
                "per_layer_mean": v["mean"], "per_layer_lo": v["lo"], "per_layer_hi": v["hi"]}
    for k, v in P.items():
        if k.startswith("auroc"):
            out["auroc"][k[6:]] = {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in v}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
