"""Analysis of gemma_own_output.py output: S (G1) and silent-vs-open AUROC (G2) on the model's own output (CPU)."""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from gemma_analyze import auroc, boot_items, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own"); ap.add_argument("--runs", nargs="+"); ap.add_argument("--flags", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    own = json.load(open(a.own))
    meta = {}
    for p in a.runs:
        for l in open(p):
            r = json.loads(l); meta[r["run_id"]] = r
    flags = {}
    for p in a.flags:
        flags.update(json.load(open(p)))
    rows = []
    for o in own["runs"]:
        r, f = meta[o["run_id"]], flags[o["run_id"]]
        tl = np.array(o["tl"], np.float32)  # [k, 8]
        let = np.logaddexp(tl[:, 0::2], tl[:, 1::2])  # [k,4]
        L = r["letters"]; ci = L.index(r["correct_letter"]); ans = L.index(o["answer_letter"])
        rows.append({"item": r["question_id"], "group": f["group"], "tokens": f["tokens"], "ci": ci, "ans": ans, "let": let,
                     "d": abs(o["answer_lp_hf"] - o["answer_lp_vllm"])})
    sil = [x for x in rows if x["group"] == "silent" and x["ans"] != x["ci"]]
    opn = [x for x in rows if x["group"] == "open" and x["ans"] != x["ci"]]
    res = {"n_silent": len(sil), "n_open": len(opn), "items_silent": len({x["item"] for x in sil}), "softcap": own["softcap"]}
    d = np.array([x["d"] for x in rows])
    res["validation_answer_token_abs_diff_hf_vs_vllm"] = {"n": len(d), "mean": round(float(d.mean()), 4), "median": round(float(np.median(d)), 4),
                                                         "p99": round(float(np.percentile(d, 99)), 4), "max": round(float(d.max()), 3)}
    def S(x, ki):
        oth = [k for k in range(4) if k not in (x["ci"], x["ans"])]
        return float(x["let"][ki, x["ci"]] - x["let"][ki, oth].mean())
    def top(x, ki):
        oth = [k for k in range(4) if k not in (x["ci"], x["ans"])]
        return float(x["let"][ki, x["ci"]] > x["let"][ki, oth].max())
    def item_boot(lst, fn):
        pi = defaultdict(list)
        for x in lst: pi[x["item"]].append(fn(x))
        ks = sorted(pi); rng = np.random.default_rng(0)
        est = np.mean([np.mean(pi[k]) for k in ks])
        bs = [np.mean([np.mean(pi[k]) for k in rng.choice(ks, size=len(ks))]) for _ in range(B)]
        return {"mean": round(float(est), 3), "lo": round(float(np.percentile(bs, 2.5)), 3), "hi": round(float(np.percentile(bs, 97.5)), 3), "n_items": len(ks)}
    for ki, k in enumerate((0, 1, 8, 16)):
        e = {"S_silent": item_boot(sil, lambda x: S(x, ki)), "S_open": item_boot(opn, lambda x: S(x, ki)),
             "correct_highest_of_unchosen_silent(chance 0.33)": item_boot(sil, lambda x: top(x, ki))}
        g = {"silent": [(x["item"], S(x, ki)) for x in sil], "open": [(x["item"], S(x, ki)) for x in opn]}
        e["AUROC_silent_vs_open"] = boot_items(g, lambda s: auroc(s["silent"], s["open"]))
        res[f"k{k}_tokens_before_answer"] = e
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
