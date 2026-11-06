"""gemma12-silent: known items of Gemma-3-12B-it from the local unhinted screen (>= 9/10 correct; unparsed or
truncated counts as not correct). Writes data/explore/gemma12/known_items.jsonl in the known_items.jsonl format.
  uv run --project envs/judge python scripts/explore/gemma12_items.py data/explore/gemma12/screen
"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gemma27_behaviour import own_answer, rows

d = Path(sys.argv[1])
items = {r["id"]: r for r in rows(d / "items.jsonl")}
n, k, unparsed = Counter(), Counter(), 0
for r in rows(d / "runs.jsonl"):
    a = own_answer(r)
    unparsed += a is None
    n[r["question_id"]] += 1
    k[r["question_id"]] += a == r["correct_letter"]
out = [{**it, "n_correct": {"A": k[q]}, "n": {"A": n[q]}, "acc": {"A": k[q] / n[q]}} for q, it in items.items() if n[q]]
known = [r for r in out if r["acc"]["A"] >= 0.9]
p = d.parent / "known_items.jsonl"
p.write_text("".join(json.dumps(r) + "\n" for r in out))
print(f"{len(out)} items screened, {sum(n.values())} runs, {unparsed} unparsed; {len(known)} known (>=9/10); "
      f"correct letters {dict(Counter(r['correct'] for r in known))}; wrote {p}")
