"""Recompute labels.jsonl of wave-2 run dirs from runs.jsonl + items.jsonl with the current
jmr.explore.w2_labels (CPU). The previous file is kept as labels.prev.jsonl.

uv run --project envs/gen python scripts/explore/w2_relabel.py data/explore/w2/instructed data/explore/w2/grader
"""

import argparse
import json
from pathlib import Path

from jmr import config
from jmr.explore.w2_labels import w2_label
from jmr.records import RunRecord


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    args = p.parse_args()
    for x in args.dirs:
        d = Path(x) if Path(x).is_absolute() else config.REPO_ROOT / x
        texts = {}
        for line in (d / "items.jsonl").read_text().splitlines():
            q = json.loads(line)
            texts[q["id"]] = dict(zip(q["letters"], q["options"]))
        out = []
        with open(d / "runs.jsonl") as f:
            for line in f:
                row = json.loads(line)
                rec = RunRecord(**{k: v for k, v in row.items() if k in RunRecord.__dataclass_fields__})
                out.append(w2_label(row, rec, correct_text=texts.get(rec.question_id, {}).get(rec.correct_letter)))
        if (d / "labels.jsonl").exists():
            (d / "labels.jsonl").replace(d / "labels.prev.jsonl")
        (d / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
        print(f"{d}: {len(out)} runs relabelled")


if __name__ == "__main__":
    main()
