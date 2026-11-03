"""List wave-2 runs worth a lens readout (CPU). Writes data/explore/w2/lens_candidates.jsonl:
one row per run_id with its spec dir, arm and category, e.g.

  followed            final answer = the arm's wrong letter (any arm with a shown wrong letter)
  i3_cot_silent       I3 concealment, followed, CoT names neither the correct letter nor its text (heuristic)
  i3_leaky            I3 concealment, followed, CoT names the correct option
  i2_knows            I2 debate, "Argued: B" and "Answer:" = correct
  c3_not_recovered    C3 planted <think> start, final = B
  c3_recovered        C3 planted <think> start, final = correct
  grader_<outcome>    grader-bug arms, final = B (silent / rationalised / open_exploit)
  reference_correct   matched correct runs (first 2 per arm and item) for contrasts

uv run --project envs/gen python scripts/explore/w2_lens_candidates.py data/explore/w2/instructed data/explore/w2/grader data/explore/w2/prefill
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from jmr import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default="data/explore/w2/lens_candidates.jsonl")
    args = p.parse_args()
    out, ref = [], defaultdict(int)
    for x in args.dirs:
        d = Path(x) if Path(x).is_absolute() else config.REPO_ROOT / x
        for line in (d / "labels.jsonl").read_text().splitlines():
            r = json.loads(line)
            cats = []
            wrong_letter = r["letter"] is not None and r["letter"] != r["correct_letter"]
            if wrong_letter and r["outcome"] == "followed":
                cats.append("followed")
                if r["arm"].startswith("I3"):
                    cats.append("i3_cot_silent" if r.get("cot_silent") else "i3_leaky")
                if r["arm"].startswith("C3"):
                    cats.append("c3_not_recovered")
                if r.get("grader_outcome") in ("silent", "rationalised", "open_exploit"):
                    cats.append(f"grader_{r['grader_outcome']}")
            if r["arm"].startswith("I2") and r.get("argued_letter") and r.get("belief_outcome") == "correct":
                cats.append("i2_knows")
            if r["arm"].startswith("C3") and r["outcome"] == "correct":
                cats.append("c3_recovered")
            if not cats and r["outcome"] == "correct" and ref[(d.name, r["arm"], r["question_id"])] < 2:
                ref[(d.name, r["arm"], r["question_id"])] += 1
                cats.append("reference_correct")
            for c in cats:
                out.append({"run_id": r["run_id"], "dir": str(d.relative_to(config.REPO_ROOT)), "arm": r["arm"],
                            "question_id": r["question_id"], "category": c})
    path = config.REPO_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(dict(Counter((r["dir"].split("/")[-1], r["category"]) for r in out)), "->", path)


if __name__ == "__main__":
    main()
