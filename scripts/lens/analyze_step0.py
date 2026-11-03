"""First lens analysis of the M&B gap-factorial step 0 runs (non-thinking, M&B
format, hint on every option). Numpy only (CPU, either env).

Items: those whose unhinted samples were all correct. Groups, on hinted runs
with a wrong hint: followed (final = hint), resisted (final = correct);
reference: the unhinted runs of the same items (their "hinted" column is the
mean over the three wrong letters). Per lens x layer: log-prob and log10 rank
of the correct, the hinted and the other wrong letters (mean of the two
not hinted; form "any": log-sum-exp over " X", "X",
"(X"; min rank) at
  (a) last_prompt, (b) 20 normalised-position bins over the response
  (all positions, and echo-free positions only), (c) pre_answer,
  (d) mean over echo-free response positions.
Echo-free: response positions t <= m - 2, where m is the first completion
token that is a tracked form of the correct letter (t = m - 1 predicts the
letter itself, so it is excluded too). Bootstrap over items (item means, then
resample items).

uv run --project envs/lens python scripts/lens/analyze_step0.py \
    --readouts data/readouts/step0_mayor --runs data/explore/gap_factorial/step0_mayor/runs.jsonl \
    --out data/readouts/step0_mayor/analysis.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jmr.analysis.readouts import Readouts

N_BINS, N_BOOT, SEED = 20, 2000, 0
LENSES = ("jlens", "logit")
WHATS = ("lp", "logrank")


def boot(item_vals: dict[str, np.ndarray], rng) -> dict:
    """item_vals: item -> [..] mean over that item's runs. Mean of item means and
    95% bootstrap CI (resampling items)."""
    keys = sorted(item_vals)
    if not keys:
        return {"n_items": 0}
    x = np.stack([item_vals[k] for k in keys])  # [items, ...]
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    with np.errstate(all="ignore"):
        bs = np.nanmean(x[idx], axis=1)
        return {"n_items": len(keys), "mean": np.nanmean(x, 0).tolist(),
                "lo": np.nanpercentile(bs, 2.5, axis=0).tolist(), "hi": np.nanpercentile(bs, 97.5, axis=0).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    r = Readouts(args.readouts)
    comp = {}
    with open(args.runs) as f:
        for line in f:
            d = json.loads(line)
            comp[d["run_id"]] = (len(d["prompt_token_ids"]), d["completion_token_ids"])
    form_ids = {L: [r.meta[0]["tracked"]["ids"][r.letters.index(L) * len(r.forms) + k] for k in range(len(r.forms))]
                for L in r.letters}

    # Known items: every unhinted sample correct.
    unh = defaultdict(list)
    for i, m in enumerate(r.meta):
        if m["hint_letter"] is None:
            unh[m["question_id"]].append(m["answer"] == m["correct_letter"])
    known = {q for q, v in unh.items() if v and all(v)}

    groups: dict[str, list[int]] = {"followed": [], "resisted": [], "unhinted": [], "other": []}
    for i, m in enumerate(r.meta):
        if m["question_id"] not in known:
            continue
        if m["hint_letter"] is None:
            groups["unhinted"].append(i)
        elif m["hint_letter"] != m["correct_letter"]:
            g = ("followed" if m["answer"] == m["hint_letter"] else
                 "resisted" if m["answer"] == m["correct_letter"] else "other")
            groups[g].append(i)

    def run_metrics(i: int) -> dict:
        m = r.meta[i]
        P, cids = comp[m["run_id"]]
        cset = set(form_ids[m["correct_letter"]])
        first = next((P + t for t, x in enumerate(cids) if x in cset), None)
        lo = int(r.load(i)["positions_lo"])
        start, end = m["cot_span"]  # response positions [P, pre_answer + 1)
        pos = np.arange(start, end)
        echo_free = pos <= (first - 2) if first is not None else np.ones_like(pos, bool)
        b = np.minimum(((pos - start) * N_BINS) // (end - start), N_BINS - 1)
        out = {"first_mention_rel": None if first is None else (first - start) / (end - start)}
        for lens in LENSES:
            for what in WHATS:
                for role in ("correct", "hinted", "other_wrong"):
                    # Unhinted runs: "hinted" and "other_wrong" are both the mean over the 3 wrong letters.
                    rr = {"correct": "correct", "hinted": "hinted" if m["hint_letter"] else "other_wrong_mean",
                          "other_wrong": "other_wrong_mean"}[role]
                    s = r.role_series(i, lens, rr, form="any", what=what)  # [layers, positions]
                    key = f"{lens}/{what}/{role}"
                    out[f"{key}/last_prompt"] = s[:, m["named_positions"]["last_prompt"] - lo]
                    pa = m["named_positions"]["pre_answer"]
                    out[f"{key}/pre_answer"] = s[:, pa - lo] if pa >= 0 else np.full(s.shape[0], np.nan)
                    x = s[:, start - lo:end - lo]
                    for tag, mask in (("all", np.ones_like(echo_free)), ("echo_free", echo_free)):
                        bins = np.full((s.shape[0], N_BINS), np.nan)
                        for k in range(N_BINS):
                            sel = (b == k) & mask
                            if sel.any():
                                bins[:, k] = x[:, sel].mean(1)
                        out[f"{key}/bins_{tag}"] = bins
                    out[f"{key}/echo_free_mean"] = x[:, echo_free].mean(1) if echo_free.any() else np.full(s.shape[0], np.nan)
        return out

    per_run = {i: run_metrics(i) for g in groups.values() for i in g}
    metric_keys = [k for k in next(iter(per_run.values())) if k != "first_mention_rel"]

    def item_means(idx: list[int], key: str) -> dict[str, np.ndarray]:
        by = defaultdict(list)
        for i in idx:
            by[r.meta[i]["question_id"]].append(per_run[i][key])
        with np.errstate(all="ignore"):
            return {q: np.nanmean(np.stack(v), 0) for q, v in by.items()}

    res = {"layers": r.layers, "n_bins": N_BINS, "n_boot": N_BOOT,
           "n_known_items": len(known), "n_items_total": len(unh),
           "groups": {g: {"n_runs": len(v), "n_items": len({r.meta[i]["question_id"] for i in v})}
                      for g, v in groups.items()},
           "first_mention": {}, "stats": {}, "followed_minus_resisted": {}}
    for g in ("followed", "resisted", "unhinted"):
        fm = [per_run[i]["first_mention_rel"] for i in groups[g]]
        named = [x for x in fm if x is not None]
        res["first_mention"][g] = {"frac_runs_naming_correct": len(named) / max(1, len(fm)),
                                   "median_rel_position": float(np.median(named)) if named else None}
        res["stats"][g] = {k: boot(item_means(groups[g], k), rng) for k in metric_keys}
    # Within-item difference, items with both followed and resisted runs.
    for k in metric_keys:
        if "/bins_" in k:
            continue
        f, s = item_means(groups["followed"], k), item_means(groups["resisted"], k)
        both = sorted(set(f) & set(s))
        res["followed_minus_resisted"][k] = boot({q: f[q] - s[q] for q in both}, rng)
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps({k: res[k] for k in ("n_known_items", "n_items_total", "groups", "first_mention")}, indent=1))


if __name__ == "__main__":
    main()
