"""TA14-3 blind read: build the shuffled, masked lists (lens env, CPU).

  JMR_MODEL_PROFILE=qwen3-14b uv run --project envs/lens python scripts/explore/ta14_blind_build.py \
      --readouts data/readouts/hawthorne_q14 --runs data/hawthorne/q14/runs.jsonl --out data/ta14/blind

Sample (pre-registration TA14-3): 60 aware + 60 unaware runs, seed 0, half actions and half triggers
(30 per group x dataset cell, drawn at random with numpy default_rng(0), kept in selection order).
Per run 12 lists: lens (jlens, logit) x layer (22, 28) x position (1, 9, 25 tokens before the cut,
i.e. window index n-1, n-9, n-25). Writes blind.txt (run blocks and list order shuffled, no group,
lens, layer or position) and key.json (kept separate; not read until all flags are in).
"""
import argparse, json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from jmr import config

ap = argparse.ArgumentParser()
ap.add_argument("--readouts", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
d, out = Path(a.readouts), Path(a.out); out.mkdir(parents=True, exist_ok=True)
tokj = json.loads((d / "tokens.json").read_text()); layers = tokj["layers"]; lenses = tokj["lenses"]
index = [json.loads(l) for l in open(d / "index.jsonl")]
rng = np.random.default_rng(0)
sample = []
for g in ("aware", "unaware"):
    for ds in ("actions", "triggers"):
        cell = [i for i, m in enumerate(index) if m["group"] == g and m["dataset"] == ds]
        sample += sorted(rng.choice(cell, 30, replace=False).tolist())
sample = sorted(sample)
want = {index[i]["run_id"] for i in sample}
runs = {}
for l in open(a.runs):
    r = json.loads(l)
    if r["run_id"] in want:
        runs[r["run_id"]] = r
mc = config.load("model")
tok = AutoTokenizer.from_pretrained(mc["hf_id"], revision=mc["revision"])
order = rng.permutation(len(sample))
txt, key = [], {}
lid = 0
for bi, si in enumerate(order):
    m = index[sample[si]]; r = runs[m["run_id"]]
    z = np.load(d / m["file"]); nw = int(z["n_window"])
    C = r["completion_token_ids"]; cut = m["hi"] - m["prompt_len"]
    reasoning = tok.decode(C[:cut])
    user = [x for x in r["messages"] if x["role"] == "user"][-1]["content"]
    items = []
    for li, ln in enumerate(lenses):
        for layer in (22, 28):
            for back in (1, 9, 25):
                items.append((ln, layer, back))
    perm = rng.permutation(len(items))
    txt.append(f"=== RUN R{bi:03d} ===\nTASK (user message, first 450 chars): {user[:450]!r}\n"
               f"REASONING SO FAR ({cut} tokens; last 350 chars): ...{reasoning[-350:]!r}\n")
    key[f"R{bi:03d}"] = {"run_id": m["run_id"], "group": m["group"], "dataset": m["dataset"], "question_id": m["question_id"], "lists": {}}
    for j in perm:
        ln, layer, back = items[j]
        lid += 1
        ids = z["top_ids"][lenses.index(ln), layers.index(layer), nw - back]
        toks = [tok.decode([int(t)]) for t in ids]
        txt.append(f"L{lid:04d}: " + " | ".join(repr(t)[1:-1] for t in toks))
        key[f"R{bi:03d}"]["lists"][f"L{lid:04d}"] = {"lens": ln, "layer": layer, "back": back, "tokens": toks}
    txt.append("")
(out / "blind.txt").write_text("\n".join(txt))
(out / "key.json").write_text(json.dumps(key))
print(len(sample), lid)
