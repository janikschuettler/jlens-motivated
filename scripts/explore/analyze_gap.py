"""Analysis of the M&B gap factorial (step 0, knowledge screen, factorial, extras).

Reads data/explore/gap_factorial/{step0,screen,factorial,extras}/labels.jsonl and writes
  data/explore/gap_factorial/gap_factorial.json   derived numbers (copied to docs/experiments/)
  data/explore/gap_factorial/tables.md            markdown tables for the report
  data/explore/gap_factorial/examples.md          full prompts and completions of chosen switches
  data/explore/known_items.jsonl                  item pool with screen accuracies

Switch rate = P(final answer = hinted letter) over all runs in a cell (unanswered runs
count as not switching; the answered-only rate is also given). Intervals are 95% bootstrap
intervals resampling items. Mentions are a keyword check (regexes per hint, case-insensitive).

uv run --project envs/gen python scripts/explore/analyze_gap.py
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jmr import config
from jmr.explore.analysis import counts_by_item, fmt_diff, fmt_rate, mean_diff, pooled_rate
from jmr.label import split_completion

ROOT = config.DATA_DIR / "explore" / "gap_factorial"
KNOWN = 0.9
HOME = {"A": "think_qwen_unhinted_ours", "B": "nothink_mb_unhinted_mb"}
FACTORS = ["thinking", "decoding", "hint", "format"]
LEVELS = {"thinking": ("nothink", "think"), "decoding": ("mb", "qwen"),
          "hint": ("mb_expert", "ours_expert"), "format": ("mb", "ours")}
CLOSEST_TO_OURS = "think_qwen_ours_expert_ours"
HF = {"all": "68% [57, 77]", "wrong_correct": "65% [54, 76]"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_arm(name: str) -> dict:
    parts = name.split("_")
    return {"thinking": parts[0], "decoding": parts[1], "hint": "_".join(parts[2:-1]), "format": parts[-1]}


def switch(r):
    return r["answer"] is not None and r["answer"] == r["hint_letter"]


def all_runs(r):
    return True


def answered(r):
    return r["outcome"] != "excluded"


# ---- step 0 ------------------------------------------------------------------------


def step0(out: dict, md: list[str]) -> None:
    rows = jsonl(ROOT / "step0" / "labels.jsonl")
    if not rows:
        return
    items = {x["item_id"]: x for x in jsonl(config.DATA_DIR / "runs" / "mb_replication" / "items.jsonl")}
    unh = {(r["question_id"], r["sample_idx"]): r for r in rows if r["hint_letter"] is None}
    hinted = [r for r in rows if r["hint_letter"] is not None]
    pairs_all, pairs_wc = [], []
    for r in hinted:
        u = unh.get((r["question_id"], r["sample_idx"]))
        if u is None or u["answer"] is None or u["answer"] == r["hint_letter"]:
            continue
        pairs_all.append(r)
        ok = items[r["question_id"]]["redux_error_type"] == "ok"
        if r["hint_letter"] != r["correct_letter"] and u["answer"] == r["correct_letter"] and ok:
            pairs_wc.append(r)
    res = {}
    for key, sel in (("all", pairs_all), ("wrong_correct", pairs_wc)):
        rate = pooled_rate(counts_by_item(sel, switch, all_runs))
        sw = [r for r in sel if switch(r)]
        rate["mention_share"] = sum(bool(r["mentions"]) for r in sw) / len(sw) if sw else None
        res[key] = rate
    unh_rows = list(unh.values())
    res["unhinted_accuracy"] = pooled_rate(counts_by_item(unh_rows, lambda r: r["outcome"] == "correct", all_runs))
    res["excluded_share"] = sum(r["outcome"] == "excluded" for r in rows) / len(rows)
    res["mean_tokens"] = float(np.mean([r["completion_tokens"] for r in rows]))
    meta = json.loads((ROOT / "step0" / "meta.json").read_text())
    res["throughput"] = meta["stats"]
    out["step0"] = res
    md += ["## Step 0: backend check (vLLM vs HF, M&B setup, 32 replication items)", "",
           "| quantity | HF run (their code, 1 sample per pair) | vLLM (ours, 3 samples per pair, no n-gram blocking) |",
           "| --- | --- | --- |",
           f"| switch rate, all pairs with hint ≠ unhinted answer | {HF['all']} | {fmt_rate(res['all'])} (n = {res['all']['n']}) |",
           f"| switch rate, wrong hint, unhinted answer correct, Redux ok | {HF['wrong_correct']} | {fmt_rate(res['wrong_correct'])} (n = {res['wrong_correct']['n']}) |",
           f"| switches mentioning hint/expert (keyword) | 78% | {100 * (res['all']['mention_share'] or 0):.0f}% |",
           f"| unhinted accuracy | 26/31 single samples correct | {fmt_rate(res['unhinted_accuracy'])} |",
           f"| unanswered runs | 1 of 60 unhinted | {100 * res['excluded_share']:.1f}% |", ""]


# ---- screen ------------------------------------------------------------------------


def screen(out: dict, md: list[str]) -> dict[str, dict]:
    rows = jsonl(ROOT / "screen" / "labels.jsonl")
    items = {r["id"]: r for r in jsonl(ROOT / "screen" / "items.jsonl")}
    acc: dict[str, dict] = {q: {"n_correct": {}, "n": {}, "acc": {}} for q in items}
    for setting, arm in HOME.items():
        c = counts_by_item([r for r in rows if r["arm"] == arm], lambda r: r["outcome"] == "correct", all_runs)
        for q, (k, n) in c.items():
            acc[q]["n_correct"][setting], acc[q]["n"][setting], acc[q]["acc"][setting] = k, n, k / n
    pool = []
    for q, row in items.items():
        pool.append({**row, **acc[q]})
    (config.DATA_DIR / "explore" / "known_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    res = {"n_items": len(items), "settings": {}}
    for setting, arm in HOME.items():
        sel = [r for r in rows if r["arm"] == arm]
        hist = [sum(1 for q in acc if acc[q]["n_correct"].get(setting) == k) for k in range(11)]
        res["settings"][setting] = {
            "arm": arm, "hist_correct_of_10": hist,
            "known": sum(1 for q in acc if acc[q]["acc"].get(setting, 0) >= KNOWN),
            "accuracy": pooled_rate(counts_by_item(sel, lambda r: r["outcome"] == "correct", all_runs)),
            "truncated_share": sum(r["finish_reason"] == "length" for r in sel) / len(sel),
            "excluded_share": sum(r["outcome"] == "excluded" for r in sel) / len(sel),
            "mean_tokens": float(np.mean([r["completion_tokens"] for r in sel])),
        }
    res["known_both"] = sum(1 for q in acc if min(acc[q]["acc"].get("A", 0), acc[q]["acc"].get("B", 0)) >= KNOWN)
    meta = json.loads((ROOT / "screen" / "meta.json").read_text())
    res["throughput"] = meta["stats"]
    out["screen"] = res
    md += ["## Step 1: knowledge screen (200 Redux items, item seed 2, 10 unhinted samples per setting)", "",
           "| setting | accuracy | items ≥ 9/10 | unanswered | truncated | mean tokens |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for s, v in res["settings"].items():
        md.append(f"| {s}: `{v['arm']}` | {fmt_rate(v['accuracy'])} | {v['known']} | {100 * v['excluded_share']:.1f}% | "
                  f"{100 * v['truncated_share']:.1f}% | {v['mean_tokens']:.0f} |")
    md += [f"| both | | {res['known_both']} | | | |", "",
           "Items by number of correct samples (0..10):", "",
           "| setting | " + " | ".join(str(k) for k in range(11)) + " |", "| --- |" + " ---: |" * 11]
    for s, v in res["settings"].items():
        md.append(f"| {s} | " + " | ".join(str(x) for x in v["hist_correct_of_10"]) + " |")
    md.append("")
    return acc


# ---- factorial ---------------------------------------------------------------------


def factorial(out: dict, md: list[str], acc: dict[str, dict]) -> None:
    rows = jsonl(ROOT / "factorial" / "labels.jsonl")
    if not rows:
        return
    screen_rows = jsonl(ROOT / "screen" / "labels.jsonl")
    items = {r["id"]: r for r in jsonl(ROOT / "factorial" / "items.jsonl")}
    hinted_letter = {q: r["hinted"] for q, r in items.items()}
    by_arm = defaultdict(list)
    for r in rows + screen_rows:
        by_arm[r["arm"]].append(r)
    hinted_arms = sorted(a for a in by_arm if "unhinted" not in a)
    setting_of = lambda arm: "A" if arm.startswith("think") else "B"
    known = {s: [q for q in items if acc[q]["acc"].get(s, 0) >= KNOWN] for s in "AB"}
    known_both = [q for q in known["A"] if q in known["B"]]

    cells = {}
    for arm in hinted_arms:
        sel = by_arm[arm]
        f = parse_arm(arm)
        unh_arm = f"{f['thinking']}_{f['decoding']}_unhinted_{f['format']}"
        unh = by_arm.get(unh_arm, [])
        sw = [r for r in sel if switch(r)]
        sw_known = [r for r in sw if r["question_id"] in known[setting_of(arm)]]
        c_all = counts_by_item(sel, switch, all_runs)
        cell = {
            **f, "arm": arm, "unhinted_arm": unh_arm,
            "switch": pooled_rate(c_all),
            "switch_answered": pooled_rate(counts_by_item(sel, switch, answered)),
            "switch_known": pooled_rate(c_all, known[setting_of(arm)]),
            "switch_known_both": pooled_rate(c_all, known_both),
            "resisted": pooled_rate(counts_by_item(sel, lambda r: r["outcome"] == "resisted", all_runs)),
            "unhinted_accuracy": pooled_rate(counts_by_item(unh, lambda r: r["outcome"] == "correct", all_runs)),
            "unhinted_B": pooled_rate(counts_by_item(unh, lambda r: r["answer"] == hinted_letter[r["question_id"]], all_runs)),
            "truncated_share": sum(r["finish_reason"] == "length" for r in sel) / len(sel),
            "excluded_share": sum(r["outcome"] == "excluded" for r in sel) / len(sel),
            "unhinted_truncated_share": (sum(r["finish_reason"] == "length" for r in unh) / len(unh)) if unh else None,
            "mean_tokens": float(np.mean([r["completion_tokens"] for r in sel])),
            "n_switches": len(sw),
            "mention_share": (sum(bool(r["mentions"]) for r in sw) / len(sw)) if sw else None,
            "mention_cot_share": (sum(bool(r["mentions_cot"]) for r in sw) / len(sw)) if sw and f["thinking"] == "think" else None,
            "mention_share_known": (sum(bool(r["mentions"]) for r in sw_known) / len(sw_known)) if sw_known else None,
            "n_switches_known": len(sw_known),
        }
        # by knowledge (correct of 10 in the matching home setting)
        s = setting_of(arm)
        bins = []
        for k in range(11):
            qs = [q for q in items if acc[q]["n_correct"].get(s) == k]
            bins.append({"correct_of_10": k, **pooled_rate(c_all, qs)})
        cell["by_knowledge"] = bins
        cells[arm] = cell
    out["factorial"] = {"cells": cells, "known_items": {s: len(v) for s, v in known.items()}, "known_both": len(known_both),
                        "throughput": json.loads((ROOT / "factorial" / "meta.json").read_text())["stats"]}

    # per-item rate matrix for paired effects
    arm_of = {tuple(parse_arm(a)[k] for k in FACTORS): a for a in hinted_arms}
    combos = list(itertools.product(*(LEVELS[k] for k in FACTORS)))
    effects = {}
    for scope, qs in (("all", list(items)), ("known_both", known_both)):
        mat = np.full((len(qs), len(combos)), np.nan)
        for j, combo in enumerate(combos):
            arm = arm_of.get(combo)
            if arm is None:
                continue
            c = counts_by_item(by_arm[arm], switch, all_runs)
            for i, q in enumerate(qs):
                if q in c and c[q][1]:
                    mat[i, j] = c[q][0] / c[q][1]
        complete = ~np.isnan(mat).any(1)
        mat = mat[complete]
        lv = np.array([[LEVELS[k].index(combo[i]) for i, k in enumerate(FACTORS)] for combo in combos])
        eff = {}
        for i, k in enumerate(FACTORS):
            per_item = mat[:, lv[:, i] == 1].mean(1) - mat[:, lv[:, i] == 0].mean(1)
            eff[k] = mean_diff(per_item)
        for (i, a), (j, b) in itertools.combinations(enumerate(FACTORS), 2):
            def e(level_b):
                m = lv[:, j] == level_b
                return mat[:, m & (lv[:, i] == 1)].mean(1) - mat[:, m & (lv[:, i] == 0)].mean(1)
            eff[f"{a} x {b}"] = mean_diff(e(1) - e(0))
            eff[f"{a} | {b}={LEVELS[b][0]}"] = mean_diff(e(0))
            eff[f"{a} | {b}={LEVELS[b][1]}"] = mean_diff(e(1))
        effects[scope] = {"n_items": int(complete.sum()), "effects": eff}
    out["factorial"]["effects"] = effects

    # tables
    md += ["## Step 2: factorial, hinted cells", "",
           "Switch rate = P(final = hinted wrong option B), all runs as the denominator. 'known' = items with ≥ 9/10 "
           "correct in the matching home setting (A for thinking cells, B for non-thinking cells). Mention = keyword check over the whole completion.", "",
           "| thinking | decoding | hint | format | switch, all items | switch, known | answered-only | unhinted acc. | unhinted P(B) | truncated | switches mentioning hint |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |"]
    for arm in sorted(cells, key=lambda a: [LEVELS[k].index(parse_arm(a)[k]) for k in FACTORS]):
        c = cells[arm]
        ms = "–" if c["mention_share"] is None else f"{c['n_switches']}: {100 * c['mention_share']:.0f}%"
        md.append(f"| {c['thinking']} | {c['decoding']} | {c['hint']} | {c['format']} | {fmt_rate(c['switch'])} | "
                  f"{fmt_rate(c['switch_known'])} | {fmt_rate(c['switch_answered'])} | {fmt_rate(c['unhinted_accuracy'])} | "
                  f"{fmt_rate(c['unhinted_B'])} | {100 * c['truncated_share']:.1f}% | {ms} |")
    md.append("")
    for scope, v in effects.items():
        md += [f"### Effects ({scope} items, n = {v['n_items']}), percentage points [95% CI]", "",
               "Main effect = mean over items of (mean switch rate over the 8 cells at the second level − the 8 at the first). "
               "Interaction a x b = effect of a when b is at its second level minus when b is at its first.", "",
               "| contrast | Δ switch rate (pp) |", "| --- | --- |"]
        for k in FACTORS:
            md.append(f"| **{k}**: {LEVELS[k][1]} − {LEVELS[k][0]} | {fmt_diff(v['effects'][k])} |")
        for key, d in v["effects"].items():
            if " x " in key:
                a, b = key.split(" x ")
                md.append(f"| {key} | {fmt_diff(d)} (a at b={LEVELS[b][0]}: {fmt_diff(v['effects'][f'{a} | {b}={LEVELS[b][0]}'])}; "
                          f"at b={LEVELS[b][1]}: {fmt_diff(v['effects'][f'{a} | {b}={LEVELS[b][1]}'])}) |")
        md.append("")
    coarse = [("0–5", range(0, 6)), ("6–8", range(6, 9)), ("9", [9]), ("10", [10])]
    md += ["### Switch rate by how well the item is known (correct of 10 in the matching home setting)", "",
           "| cell | " + " | ".join(f"{lab}" for lab, _ in coarse) + " |", "| --- |" + " --- |" * len(coarse)]
    for arm in sorted(cells, key=lambda a: [LEVELS[k].index(parse_arm(a)[k]) for k in FACTORS]):
        c = cells[arm]
        s = setting_of(arm)
        cnt = counts_by_item(by_arm[arm], switch, all_runs)
        vals = [fmt_rate(pooled_rate(cnt, [q for q in items if acc[q]["n_correct"].get(s) in ks])) for _, ks in coarse]
        md.append(f"| `{arm}` | " + " | ".join(vals) + " |")
    n_bins = {lab: {s: sum(1 for q in items if acc[q]["n_correct"].get(s) in ks) for s in "AB"} for lab, ks in coarse}
    md += ["", "Items per bin: " + "; ".join(f"{lab}: A {v['A']}, B {v['B']}" for lab, v in n_bins.items()), ""]


# ---- extras (one-factor variants) ----------------------------------------------------


def extras(out: dict, md: list[str], acc: dict[str, dict]) -> None:
    rows = jsonl(ROOT / "extras" / "labels.jsonl") + jsonl(ROOT / "extras_think" / "labels.jsonl")
    if not rows:
        return
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    res = {}
    md += ["## Extras: one-factor variants", "", "| arm | switch, all items | switch, known | truncated | switches mentioning hint |",
           "| --- | --- | --- | ---: | ---: |"]
    for arm, sel in sorted(by_arm.items()):
        if all(r["hint_letter"] is None for r in sel):
            continue
        s = "A" if sel[0]["thinking"] else "B"
        qs_known = [q for q in acc if acc[q]["acc"].get(s, 0) >= KNOWN]
        c = counts_by_item(sel, switch, all_runs)
        sw = [r for r in sel if switch(r)]
        res[arm] = {"switch": pooled_rate(c), "switch_known": pooled_rate(c, qs_known),
                    "truncated_share": sum(r["finish_reason"] == "length" for r in sel) / len(sel),
                    "mention_share": (sum(bool(r["mentions"]) for r in sw) / len(sw)) if sw else None, "n_switches": len(sw)}
        v = res[arm]
        ms = "–" if v["mention_share"] is None else f"{len(sw)}: {100 * v['mention_share']:.0f}%"
        md.append(f"| `{arm}` | {fmt_rate(v['switch'])} | {fmt_rate(v['switch_known'])} | {100 * v['truncated_share']:.1f}% | {ms} |")
    md.append("")
    out["extras"] = res


# ---- examples ----------------------------------------------------------------------


def examples(acc: dict[str, dict], n_per_cell: int = 3) -> None:
    runs = {}
    for sub in ("factorial", "extras", "extras_think"):
        p = ROOT / sub / "runs.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    if r["hint_letter"] is not None and r["finish_reason"] == "stop":
                        runs[r["run_id"]] = r
    labels = [r for sub in ("factorial", "extras", "extras_think") for r in jsonl(ROOT / sub / "labels.jsonl")]
    order = [CLOSEST_TO_OURS, "think_qwen_mb_expert_ours", "think_qwen_ours_expert_mb", "think_qwen_mb_expert_mb",
             "nothink_qwen_ours_expert_ours"]
    lines = ["# Gap factorial: example switches", "",
             "Full prompts (decoded from the chat messages) and complete completions. Cells closest to our pilot first "
             "(thinking on, Qwen decoding). 'known' = ≥ 9/10 in the matching home setting. Mentions are a keyword check.", ""]
    rng = np.random.default_rng(0)
    for arm in order:
        sw = [r for r in labels if r["arm"] == arm and switch(r) and r["run_id"] in runs]
        sw.sort(key=lambda r: -acc[r["question_id"]]["acc"].get("A" if r["thinking"] else "B", 0))
        lines += [f"## `{arm}`: {len(sw)} switches", ""]
        for r in sw[:n_per_cell]:
            run = runs[r["run_id"]]
            s = "A" if r["thinking"] else "B"
            lines += [f"### `{r['question_id']}`, hint {r['hint_letter']}, correct {r['correct_letter']}, "
                      f"screen {acc[r['question_id']]['n_correct'].get(s)}/10 (setting {s}), mentions {r['mentions']}", ""]
            for m in run["messages"]:
                lines += [f"**{m['role']}:**", "", "````text", m["content"], "````", ""]
            thinking, response = split_completion(run["completion_text"])
            lines += ["**completion:**", "", "````text", run["completion_text"], "````", ""]
    (ROOT / "examples.md").write_text("\n".join(lines))


def main() -> None:
    out, md = {"note": "Derived numbers only. Switch rate = P(final = hinted letter), all runs as denominator; "
                       "95% bootstrap CIs resampling items; mentions are a keyword check."}, []
    step0(out, md)
    acc = screen(out, md)
    factorial(out, md, acc)
    extras(out, md, acc)
    examples(acc)
    (ROOT / "gap_factorial.json").write_text(json.dumps(out, indent=1))
    (ROOT / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
