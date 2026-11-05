"""Derived numbers and markdown tables for docs/experiments/instructed-e1/report.md (CPU).
Reads labels.jsonl (and forks.jsonl) of the given dirs; writes JSON (derived numbers only:
no CoTs, no token IDs) and prints markdown.

uv run --project envs/gen python scripts/explore/w2_part1_tables.py --out data/explore/w2/wave2_part1.json \
    data/explore/w2/instructed data/explore/w2/grader data/explore/w2/forks_early_mb [more]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from jmr import config
from jmr.explore.analysis import counts_by_item, fmt_rate, pooled_rate


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def all_runs(r):
    return True


def runs_section(d: Path, out: dict) -> list[str]:
    rows = jsonl(d / "labels.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    md = [f"### {d.name}", "", "| arm | runs | items | unanswered | final = arm's letter | final = correct | mean tokens |",
          "| --- | ---: | ---: | ---: | --- | --- | ---: |"]
    res = {}
    for arm, rs in by.items():
        f = pooled_rate(counts_by_item(rs, lambda r: r["letter"] is not None and r["answer"] == r["letter"], all_runs))
        c = pooled_rate(counts_by_item(rs, lambda r: r["answer"] == r["correct_letter"], all_runs))
        un = sum(r["outcome"] == "excluded" for r in rs) / len(rs)
        e = {"runs": len(rs), "items": len({r["question_id"] for r in rs}), "unanswered": un, "final_letter": f,
             "final_correct": c, "mean_tokens": float(np.mean([r["completion_tokens"] for r in rs]))}
        if "grader_outcome" in rs[0]:
            e["grader_outcome"] = dict(Counter(r["grader_outcome"] for r in rs))
            e["meta_flag"] = sum(bool(r["meta"]) for r in rs)
        if "belief_outcome" in rs[0]:
            e["belief_outcome"] = dict(Counter(r["belief_outcome"] for r in rs))
            e["argued_is_letter"] = sum(bool(r["argued_letter"]) for r in rs)
            e["argued_line_present"] = sum(r["argued"] is not None for r in rs)
            both = [r for r in rs if r["argued_letter"]]
            e["belief_correct_given_argued_letter"] = pooled_rate(counts_by_item(both, lambda r: r["belief_outcome"] == "correct", all_runs))
        if "cot_silent" in rs[0]:
            fol = [r for r in rs if r["outcome"] == "followed"]
            e["followed"] = len(fol)
            e["followed_cot_silent"] = sum(r["cot_silent"] for r in fol)
            e["followed_letter_mention"] = sum(bool(r["correct_letter_cot"]) for r in fol)
            e["followed_text_mention"] = sum(bool(r["correct_text_cot"]) for r in fol)
        res[arm] = e
        letter = fmt_rate(f) if rs[0]["letter"] is not None else "–"
        md.append(f"| {arm} | {len(rs)} | {e['items']} | {100 * un:.1f}% | {letter} | {fmt_rate(c)} | {e['mean_tokens']:.0f} |")
    out[d.name] = res
    return md + [""]


def forks_section(d: Path, out: dict) -> list[str]:
    rows = [r for r in jsonl(d / "forks.jsonl") if r["kind"] == "early" and r.get("p_norm")]
    g = defaultdict(list)
    for r in rows:
        wrong = r["hint_letter"] is not None and r["hint_letter"] != r["correct_letter"]
        if not wrong:
            continue
        g[(r["source_class"], round(r["f"], 1))].append(r)
    res = {}
    md = [f"### {d.name}: forced early answers, wrong-letter hint traces (letter-renormalised P)", "",
          "| f | followed: P(correct) | followed: P(letter) | resisted: P(correct) | resisted: P(letter) | n followed / resisted |",
          "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for f in sorted({k[1] for k in g}):
        cell = {}
        for cls in ("hint", "correct"):
            rs = g.get((cls, f), [])
            cell[cls] = {"n": len(rs),
                         "p_correct": float(np.mean([r["p_norm"][r["correct_letter"]] for r in rs])) if rs else None,
                         "p_letter": float(np.mean([r["p_norm"][r["hint_letter"]] for r in rs])) if rs else None}
        res[str(f)] = cell
        h, c = cell["hint"], cell["correct"]
        fm = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
        md.append(f"| {f:.1f} | {fm(h['p_correct'])} | {fm(h['p_letter'])} | {fm(c['p_correct'])} | {fm(c['p_letter'])} | {h['n']} / {c['n']} |")
    out[d.name] = res
    return md + [""]


def reelicit_section(d: Path, out: dict) -> list[str]:
    rows = jsonl(d / "forks.jsonl")
    g = defaultdict(list)
    for r in rows:
        key = ("control", "-", "-") if r["kind"] == "control" else (r["arm"], r["source_class"], r["mode"])
        g[key].append(r)
    md = [f"### {d.name}: re-elicitation (argmax letter of the next token; thinking off)", "",
          "| arm | source class | history | n | argmax = correct | argmax = arm's letter | mean P(letters) |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    res = {}
    for key in sorted(g):
        rs = g[key]
        ac = float(np.mean([r["argmax"] == r["correct_letter"] for r in rs]))
        al = float(np.mean([r["argmax"] == r["hint_letter"] for r in rs])) if rs[0]["hint_letter"] else None
        pl = float(np.mean([r["p_letters"] for r in rs]))
        res["|".join(key)] = {"n": len(rs), "argmax_correct": ac, "argmax_letter": al, "p_letters": pl}
        md.append(f"| {' | '.join(key)} | {len(rs)} | {ac:.2f} | {'–' if al is None else f'{al:.2f}'} | {pl:.2f} |")
    out[d.name] = res
    return md + [""]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out, md = {}, []
    for x in args.dirs:
        d = Path(x) if Path(x).is_absolute() else config.REPO_ROOT / x
        if (d / "forks.jsonl").exists():
            kinds = {json.loads(line)["kind"] for line in open(d / "forks.jsonl")}
            md += forks_section(d, out) if "early" in kinds else reelicit_section(d, out)
        else:
            md += runs_section(d, out)
    path = Path(args.out) if Path(args.out).is_absolute() else config.REPO_ROOT / args.out
    path.write_text(json.dumps(out, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
