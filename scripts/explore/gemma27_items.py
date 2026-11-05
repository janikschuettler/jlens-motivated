"""gemma27-silent: Gemma-3-27B-it's known items from the API screen (models-grader unhinted, 5 of 5 correct).
Writes data/explore/gemma27/known_items.jsonl in the format of known_items.jsonl (acc.A = 1.0 = 5/5 API).
Lighter filter than the project's 9/10 (label it): the local screen was too slow with CPU offload.
  uv run --project envs/judge python scripts/explore/gemma27_items.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import or_grader as og
import yaml
og.CFG = yaml.safe_load(open(og.REPO / "configs/explore/wording_cross_model.yaml"))
runs = {}
for l in open(og.REPO / "data/models_grader/gemma-3-27b-it/runs.jsonl"):
    r = json.loads(l)
    if r["arm"] == "unhinted":
        runs[r["run_id"]] = r
out = []
for it in og.load_items():
    rs = [runs.get(f"gemma-3-27b-it/unhinted/{it['id']}/{s}") for s in range(5)]
    if all(r and r["answer"] == it["correct"] for r in rs):
        out.append({**it, "acc": {"A": 1.0}, "n_correct": {"A": 5}, "n": {"A": 5}})
p = og.REPO / "data/explore/gemma27"
p.mkdir(parents=True, exist_ok=True)
(p / "known_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out), "known items ->", p / "known_items.jsonl")
