"""Exploratory add-on to W (docs/experiments/hack-waterproof): does the change in the readout across the
second half of the reasoning track the change in the chance of a hack? CPU.
For each run with resamples at the mid cut and the end cut: dP = P_end - P_mid; dC = C (deciles 6-10 mean)
- C (deciles 1-5 mean), J-lens / logit lens, layer mean. Within-problem Spearman (values centred per
problem), item-level bootstrap over problems.
python scripts/lens/analyze_hack_wp_mid.py --root data/hack_wp --out data/hack_wp/analysis_mid.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hack_wp import LENSES, load_dir, wp_spearman  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hack_wp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    pe = {json.loads(l)["run_id"]: json.loads(l) for l in open(root / "resample_end/resamples.jsonl")}
    pm = {json.loads(l)["run_id"]: json.loads(l) for l in open(root / "resample_mid/resamples.jsonl")}
    runs = [r for r in load_dir(root / "ro/disp", False) if r["run_id"] in pe and r["run_id"] in pm]
    prob = np.array([r["problem_id"] for r in runs])
    dP = np.array([pe[r["run_id"]]["p_hat"] - pm[r["run_id"]]["p_hat"] for r in runs])
    out = {"n_runs": len(runs), "mean_dP": float(dP.mean()), "mean_P_mid": float(np.mean([pm[r["run_id"]]["p_hat"] for r in runs])),
           "mean_P_end": float(np.mean([pe[r["run_id"]]["p_hat"] for r in runs]))}
    for li, lens in enumerate(LENSES):
        dC = np.array([np.mean([r["C"][f"decile_{k}"][li].mean() for k in range(6, 11)])
                       - np.mean([r["C"][f"decile_{k}"][li].mean() for k in range(1, 6)]) for r in runs])
        ok = np.isfinite(dC)
        out[f"{lens}_dC_vs_dP"] = wp_spearman(dC[ok], dP[ok], prob[ok])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()
