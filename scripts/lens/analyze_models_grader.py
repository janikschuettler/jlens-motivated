"""Lens analysis for the models x grader-bug study (lens env or any env with numpy; CPU).

  python scripts/lens/analyze_models_grader.py --readouts data/readouts/mg_qwen8b \
      --runs data/explore/models_grader/qwen8b/runs.jsonl --labels data/explore/models_grader/qwen8b/labels.jsonl \
      --out data/readouts/mg_qwen8b_analysis.json

Groups, per arm: followed (final = the hinted wrong letter B), resisted (final = correct).
In null arms (no letter shown), "correct" runs are the reference.

Measures (fixed before reading these readouts; docs/overnight/reports/models_grader.md, "Lens plan"):
  letter margins, form "any" (log-sum-exp over " X", "X", "(X"), per lens x layer:
    corr = lp(correct) - mean lp(U), sal = lp(B) - mean lp(U), U = the two letters that are
    neither correct nor B. At pre_answer (the position whose next token is the final letter),
    think_end and last_prompt.
    Score S = corr margin at pre_answer, mean over the workspace layers.
    Salience control = sal margin in resisted runs of the same arm.
  concept tokens (x arrays with x_opt = -1; configs/concept_tokens_grader.json): log-prob of the
    'grader' group (log-sum-exp over its tokens) at last_prompt, think_end, pre_answer and the
    mean over CoT positions; per lens x layer.
  AUROC followed vs resisted (same arm) for S and for the grader-concept score at last_prompt
  and CoT mean, item-bootstrap 95% intervals.
Derived numbers only are written (rule 3: the readouts hold no full-vocabulary output).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, SEED = 2000, 0
LENSES = ("jlens", "logit")


def lse(a, axis=-1):
    return np.logaddexp.reduce(a, axis=axis)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        idx = np.where(allv == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auroc(pos_by_item, neg_by_item, rng):
    items = sorted(set(pos_by_item) | set(neg_by_item))
    point = auroc([v for k in items for v in pos_by_item.get(k, [])], [v for k in items for v in neg_by_item.get(k, [])])
    stats = []
    for _ in range(N_BOOT):
        ks = [rng.choice(items) for _ in items]
        p = [v for k in ks for v in pos_by_item.get(k, [])]
        n = [v for k in ks for v in neg_by_item.get(k, [])]
        if p and n:
            stats.append(auroc(p, n))
    stats.sort()
    return {"auroc": point, "lo": stats[int(0.025 * len(stats))] if stats else None,
            "hi": stats[int(0.975 * len(stats)) - 1] if stats else None,
            "n_pos": sum(len(v) for v in pos_by_item.values()), "n_neg": sum(len(v) for v in neg_by_item.values()),
            "items_pos": len(pos_by_item), "items_neg": len(neg_by_item)}


def boot_mean(by_item, rng):
    items = sorted(by_item)
    if not items:
        return None
    means = {k: np.mean(v, axis=0) for k, v in by_item.items()}
    point = np.mean([v for k in items for v in by_item[k]], axis=0)
    stats = np.array([np.mean([means[rng.choice(items)] for _ in items], axis=0) for _ in range(N_BOOT)])
    return {"mean": np.round(point, 3).tolist(), "lo": np.round(np.quantile(stats, 0.025, axis=0), 3).tolist(),
            "hi": np.round(np.quantile(stats, 0.975, axis=0), 3).tolist(),
            "n": sum(len(v) for v in by_item.values()), "items": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--runs", required=True, help="runs.jsonl (for the arm of each run)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(SEED)
    R = Path(a.readouts)
    arm_of = {}
    for line in open(a.runs):
        r = json.loads(line)
        arm_of[r["run_id"]] = r.get("arm") or r["run_id"].split("/")[1]
    metas = [json.loads(l) for l in open(R / "index.jsonl")]
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # arm -> group -> measure -> [(item, vec)]
    layers = None
    concept_groups = None
    for m in metas:
        arm = arm_of.get(m["run_id"])
        C, B, Ans = m["correct_letter"], m["hint_letter"], m["answer"]
        if arm is None or B is None or Ans is None:
            continue
        group = "followed" if Ans == B else "resisted" if Ans == C else None
        if group is None:
            continue
        z = np.load(R / m["file"])
        layers = z["layers"].tolist()
        tr = m["tracked"]
        nf = len(tr["forms"])
        lp = z["lp"].astype(np.float64)  # [2, L, T, K]
        lo = int(z["positions_lo"])

        def letter(Lx):
            k = tr["letters"].index(Lx) * nf
            return lse(lp[..., k:k + nf])  # [2, L, T]

        U = [x for x in m["letters"] if x not in (C, B)]
        lpU = np.mean([letter(x) for x in U], 0)
        corr, sal = letter(C) - lpU, letter(B) - lpU
        npos = m["named_positions"]
        q = m["question_id"]
        for anc in ("pre_answer", "think_end", "last_prompt"):
            p = npos.get(anc, -1)
            if p is None or p < 0:
                continue
            t = p - lo
            rows[arm][group][f"corr@{anc}"].append((q, corr[:, :, t]))
            rows[arm][group][f"sal@{anc}"].append((q, sal[:, :, t]))
        if "x_ids" in z.files:
            xopt = z["x_opt"]
            cidx = np.where(xopt == -1)[0]
            ctoks = m["concept_tokens"]
            if concept_groups is None:
                cfg = json.loads(Path("configs/concept_tokens_grader.json").read_text())
                gid = {t["id"]: t["group"] for t in cfg["tokens"]}
                concept_groups = sorted({gid[t["id"]] for t in ctoks})
            gid = {t["id"]: t["group"] for t in json.loads(Path("configs/concept_tokens_grader.json").read_text())["tokens"]}
            xlp = z["x_lp"].astype(np.float64)  # [2, L, T, K2]
            ts, te = m["cot_span"]
            for g in concept_groups:
                k = [int(i) for i in cidx if gid[int(z["x_ids"][i])] == g]
                s = lse(xlp[..., k])  # [2, L, T]
                for anc in ("pre_answer", "think_end", "last_prompt"):
                    p = npos.get(anc, -1)
                    if p is not None and p >= 0:
                        rows[arm][group][f"concept_{g}@{anc}"].append((q, s[:, :, p - lo]))
                if te > ts:
                    rows[arm][group][f"concept_{g}@cot_mean"].append((q, s[:, :, ts - lo:te - lo].mean(-1)))
    out = {"layers": layers, "lenses": list(LENSES), "arms": {}}
    for arm, groups in rows.items():
        oa = out["arms"][arm] = {"n": {g: len(v.get("corr@pre_answer", [])) for g, v in groups.items()}}
        for g, meas in groups.items():
            for name, lst in meas.items():
                by = defaultdict(list)
                for q, v in lst:
                    by[q].append(v)
                oa[f"{g}/{name}"] = boot_mean(by, rng)
        # AUROCs followed vs resisted on S and concept scores (mean over layers), per lens
        if "followed" in groups and "resisted" in groups:
            for name in groups["followed"]:
                if not (name.startswith("corr@pre_answer") or name.startswith("concept_")):
                    continue
                for li, lens in enumerate(LENSES):
                    pb, nb = defaultdict(list), defaultdict(list)
                    for q, v in groups["followed"][name]:
                        pb[q].append(float(np.mean(v[li])))
                    for q, v in groups["resisted"].get(name, []):
                        nb[q].append(float(np.mean(v[li])))
                    oa[f"auroc_followed_vs_resisted/{name}/{lens}"] = boot_auroc(pb, nb, rng)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for arm, oa in out["arms"].items():
        print(arm, oa["n"])
        for k, v in oa.items():
            if k.startswith("auroc") and v and v["auroc"] == v["auroc"]:
                print(f"   {k}: {v['auroc']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]")


if __name__ == "__main__":
    main()
