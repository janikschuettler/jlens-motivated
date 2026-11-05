"""Hawthorne lens test (a): does the eval-word score before the first awareness sentence tell runs that
will voice test awareness from runs that won't, within the same prompt? (numpy only; any env)

  python scripts/explore/hawthorne_lens_analyze.py --readouts data/readouts/hawthorne_q14 --out <json>

Score (the replication's, Gurnee, Sofroniew et al. 2026): log-prob of " fictional" and " simulation",
averaged, minus the mean log-prob of 100 control nouns; averaged over workspace layers and over the
positions of a window. Also with all 26 eval-set words, and with our "test" words.
Metric: within-prompt AUROC over all (aware, unaware) pairs of the same prompt (ties 1/2), 2,000
bootstrap resamples of prompts; J-lens minus logit-lens AUROC with the same resamples (paired).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 2000


def auroc_pairs(by, key):
    """by[q] = {"aware": [scores], "unaware": [scores]}; pooled within-prompt pairs."""
    num = den = 0.0
    for q, g in by.items():
        a, u = np.asarray(g["aware"][key]), np.asarray(g["unaware"][key])
        if len(a) and len(u):
            diff = a[:, None] - u[None, :]
            num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            den += diff.size
    return num / den if den else np.nan


def boot(by, keys, rng):
    qs = sorted(by)
    est = {k: auroc_pairs(by, k) for k in keys}
    bs = defaultdict(list)
    for _ in range(B):
        s = [qs[i] for i in rng.integers(0, len(qs), len(qs))]
        sub = {}
        for j, q in enumerate(s):
            sub[f"{q}#{j}"] = by[q]
        for k in keys:
            bs[k].append(auroc_pairs(sub, k))
        for k2 in keys[1:]:
            bs["diff_" + k2].append(bs[keys[0]][-1] - bs[k2][-1])
    out = {k: {"auroc": float(est[k]), "ci95": [float(np.nanquantile(bs[k], 0.025)), float(np.nanquantile(bs[k], 0.975))]}
           for k in keys}
    for k2 in keys[1:]:
        b = bs["diff_" + k2]
        out[f"{keys[0]}_minus_{k2}"] = {"est": float(est[keys[0]] - est[k2]),
                                        "ci95": [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))],
                                        "ci90": [float(np.nanquantile(b, 0.05)), float(np.nanquantile(b, 0.95))]}
    for k in keys:
        b = bs[k]
        out[k]["ci90"] = [float(np.nanquantile(b, 0.05)), float(np.nanquantile(b, 0.95))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True, help="runs.jsonl, for the text-only baseline")
    ap.add_argument("--labels", default=None, help="JSON {run_id: 'aware'|'unaware'}: use only these runs, with these groups (TA14-J)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from jmr import config
    mc = config.load("model")
    tok = AutoTokenizer.from_pretrained(mc["hf_id"], revision=mc["revision"])
    lex = re.compile(r"test|evaluat|benchmark|grader|assess|exam\b|simulat|fiction|hypothetic|scenario|fake|mock|quiz|sandbox|role-?play|pretend", re.I)
    wanted = {json.loads(l)["run_id"] for l in open(Path(a.readouts) / "index.jsonl")}
    override = json.loads(Path(a.labels).read_text()) if a.labels else None
    lexrate = {}
    d = Path(a.readouts)
    for l in open(a.runs):
        r = json.loads(l)
        if r["run_id"] in wanted:
            lexrate[r["run_id"]] = r
    tokens = json.loads((d / "tokens.json").read_text())
    cols = {c: i for i, c in enumerate(tokens["columns"])}
    groups = defaultdict(list)
    for t in tokens["tracked"]:
        for g in t["groups"]:
            groups[g].append(cols[t["text"]])
    ctrl = cols["control_mean"]
    index = [json.loads(l) for l in open(d / "index.jsonl")]
    if override is not None:
        index = [dict(m, group=override[m["run_id"]]) for m in index if m["run_id"] in override]
    windows = ("window", "last32", "first64", "window+post")
    scores = defaultdict(dict)  # (group, window) -> run -> [lens]
    meta = {}
    for m in index:
        z = np.load(d / m["file"])
        lp = z["lp"].astype(np.float32)  # [lens, layers, pos, K+1]
        nw = int(z["n_window"])
        rows = {"window": np.arange(nw), "last32": np.arange(max(0, nw - 32), nw), "first64": np.arange(min(64, nw)),
                "window+post": np.arange(lp.shape[2]) if m["group"] == "unaware" else np.arange(nw)}
        for g in ("eval_primary", "eval_set", "test"):
            s = lp[..., groups[g]].mean(-1) - lp[..., ctrl]  # [lens, layers, pos]
            for w, r in rows.items():
                scores[(g, w)][m["run_id"]] = s[:, :, r].mean((1, 2))
        meta[m["run_id"]] = m
    rng = np.random.default_rng(0)
    out = {"n_runs": len(index), "n_prompts": len({m["question_id"] for m in index}),
           "n_aware": sum(m["group"] == "aware" for m in index), "n_unaware": sum(m["group"] == "unaware" for m in index),
           "results": {}}
    for (g, w), sc in scores.items():
        for subset in ("all", "actions", "triggers"):
            by = defaultdict(lambda: {"aware": {"jlens": [], "logit": [], "len": [], "lex": []},
                                      "unaware": {"jlens": [], "logit": [], "len": [], "lex": []}})
            for rid, v in sc.items():
                m = meta[rid]
                if subset != "all" and m["dataset"] != subset:
                    continue
                by[m["question_id"]][m["group"]]["jlens"].append(v[0])
                by[m["question_id"]][m["group"]]["logit"].append(v[1])
                by[m["question_id"]][m["group"]]["len"].append(m["n_window"])
                rr = lexrate[rid]
                txt = tok.decode(rr["completion_token_ids"][: m["hi"] - m["prompt_len"]])
                by[m["question_id"]][m["group"]]["lex"].append(len(lex.findall(txt)) / max(1, m["hi"] - m["lo"]) * 100)
            by = {q: x for q, x in by.items() if x["aware"]["jlens"] and x["unaware"]["jlens"]}
            if len(by) < 3:
                continue
            if g == "eval_primary" and w == "window":
                res = boot(by, ["jlens", "logit", "len", "lex"], rng)  # lens scores vs the two baselines
            else:
                res = boot(by, ["jlens", "logit"], rng)
            res["n_prompts"] = len(by)
            out["results"][f"{g}/{w}/{subset}"] = res
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out["results"].items():
        print(k, {kk: vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
