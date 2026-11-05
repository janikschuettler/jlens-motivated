"""Q-ah behaviour + selection (CPU): uv run --project envs/judge python scripts/explore/qah_behaviour.py DIR OUT_JSON SELECT_JSON
All silent switches, open switches capped at 3 per item (seed 0)."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
d, oj, sj = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(x) for x in open(d / "labels.jsonl")]
sw, flags, opn = defaultdict(lambda: [0, 0]), {}, defaultdict(list)
nsw = nsil = 0
for l in rows:
    if l["answer"] is None: continue
    q = l["question_id"]; sw[q][1] += 1
    if l["answer"] == l["letter"]:
        sw[q][0] += 1; sil = len(l["mentions"]) == 0; nsw += 1; nsil += sil
        f = {"arm": "answer_hint", "kind": "wording", "item": q, "tokens": l["completion_tokens"], "group": "silent" if sil else "open"}
        (flags.__setitem__(l["run_id"], f) if sil else opn[q].append((l["run_id"], f)))
rng = random.Random(0)
for q, lst in opn.items():
    rng.shuffle(lst)
    for rid, f in lst[:3]: flags[rid] = f
it = sorted(sw); n = np.array([sw[i][0] for i in it], float); m = np.array([sw[i][1] for i in it], float)
r = np.random.default_rng(0); bs = [n[k].sum() / m[k].sum() for k in (r.integers(0, len(it), len(it)) for _ in range(2000))]
res = {"runs": len(rows), "parsed": int(m.sum()), "items": len(it), "switch": [round(n.sum() / m.sum(), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "n_switch": nsw, "n_silent": nsil,
       "read_silent": sum(f["group"] == "silent" for f in flags.values()), "read_open": sum(f["group"] == "open" for f in flags.values())}
Path(oj).write_text(json.dumps(res, indent=1)); Path(sj).write_text(json.dumps(sorted(flags))); Path(sj.replace(".json", "_flags.json")).write_text(json.dumps(flags)); print(res)
