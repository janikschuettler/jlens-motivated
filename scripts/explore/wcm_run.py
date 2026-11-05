"""Wording x model (OpenRouter), reusing or_grader machinery.
Known items = 5/5 unhinted correct in data/models_grader (models-grader runs); first N of them.
  zsh -ic 'uv run --project envs/judge python scripts/explore/wcm_run.py --model gemma-3-27b-it --limit-items 5 --samples 1'
"""
import argparse, json, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).parent))
import or_grader as og

REPO = og.REPO
og.CFG = yaml.safe_load(open(REPO / "configs/explore/wording_cross_model.yaml"))
og.OUT = REPO / og.CFG["out_dir"]
SRC = REPO / "data/models_grader"

def known(model, n):
    runs = {}
    for l in open(SRC / model / "runs.jsonl"):
        r = json.loads(l)
        if r["arm"] == "unhinted":
            runs[r["run_id"]] = r
    out = []
    for it in og.load_items():
        rs = [runs.get(f"{model}/unhinted/{it['id']}/{s}") for s in range(5)]
        if all(r and r["answer"] == it["correct"] for r in rs):
            out.append(it)
    return out[:n]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--limit-items", type=int, default=40)
ap.add_argument("--samples", type=int, default=2)
ap.add_argument("--start-sample", type=int, default=0)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
items = known(a.model, a.limit_items)
arms = list(og.CFG["arms"])
jobs = [(it, arm, s) for s in range(a.start_sample, a.start_sample + a.samples) for arm in arms for it in items]
sys.exit(og.run_batch(a.model, jobs, f"wcm {len(items)} items s{a.start_sample}+{a.samples}", a.workers))
