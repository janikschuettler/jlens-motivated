"""gemma27-silent exploratory (e): top-25 tokens of both lenses at pre_answer (and 8 earlier) in silent switches.
Prints per-run lists for a few runs and the most frequent tokens over all given runs, per lens and block.
  python scripts/lens/gemma_top25.py --lens DIR --ids ids.json --out out.json   (lens env, tokenizer from the HF cache)"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from jmr import config

ap = argparse.ArgumentParser()
ap.add_argument("--lens", nargs="+", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--show", type=int, default=6)
a = ap.parse_args()
m = config.load("model")
tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
want = set(json.load(open(a.ids)))
out = {}
shown = 0
freq = {}
for d in a.lens:
    idx = json.load(open(Path(d) / "index.json"))
    for i, rid in enumerate(idx["runs"]):
        if rid not in want:
            continue
        z = np.load(Path(d) / f"{i:05d}.npz")
        meta = json.loads(str(z["meta"]))
        ws = meta["layers"]
        tp = meta["top_pos"]
        for which, off in (("pre_answer", 0), ("pre_answer-8", 8)):
            pos = meta["loc"]["pre_answer"] - off
            if pos not in tp:
                continue
            j = tp.index(pos)
            for lensi, ln in enumerate(("J-lens", "logit lens")):
                for bl in (32, 44, 54):
                    li = ws.index(bl)
                    ids = z["top_i"][lensi, li, j]
                    toks = [tok.decode([int(t)]).strip() or repr(tok.decode([int(t)])) for t in ids]
                    freq.setdefault((ln, bl, which), Counter()).update(toks)
                    if shown < a.show and which == "pre_answer" and bl in (32, 54):
                        print(rid[-40:], ln, "block", bl, toks[:15])
        shown += 1
for k, c in sorted(freq.items()):
    out["/".join(map(str, k))] = c.most_common(25)
    print(k, [f"{t}:{n}" for t, n in c.most_common(18)])
Path(a.out).write_text(json.dumps(out, indent=0))
