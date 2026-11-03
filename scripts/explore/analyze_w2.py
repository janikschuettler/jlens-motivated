"""First-pass tables for wave-2 outputs (CPU; gen env for numpy). Keyword
heuristics only (jmr.explore.w2_labels); read examples before believing them.

uv run --project envs/gen python scripts/explore/analyze_w2.py data/explore/w2_grader [more dirs]
uv run --project envs/gen python scripts/explore/analyze_w2.py data/explore/w2_forks_early   # forks.jsonl dirs

Per arm: runs, unanswered share, rate of answering the arm's letter ("followed")
and the correct option, with 95% item-bootstrap intervals; label-set counts
(grader outcomes, I2 belief, I3 CoT silence among followed). Forks: mean
letter-renormalised P(correct), P(letter), P(other) (P(letter) only where the letter is wrong) by fraction f and source class (early), or by
mode and source class (reelicit), with argmax shares. Writes summary.json next
to the inputs.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from jmr import config
from jmr.explore.analysis import counts_by_item, fmt_rate, pooled_rate


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def runs_table(d: Path) -> dict:
    rows = jsonl(d / "labels.jsonl")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out = {}
    print(f"\n## {d.name}\n\n| arm | runs | unanswered | answers letter | correct | extra |\n| --- | ---: | ---: | --- | --- | --- |")
    for arm, rs in by_arm.items():
        follow = pooled_rate(counts_by_item(rs, lambda r: r["letter"] is not None and r["answer"] == r["letter"], lambda r: True))
        corr = pooled_rate(counts_by_item(rs, lambda r: r["answer"] == r["correct_letter"], lambda r: True))
        extra = {}
        if "grader_outcome" in rs[0]:
            extra["grader"] = dict(Counter(r["grader_outcome"] for r in rs))
            extra["meta_any"] = sum(bool(r["meta"]) for r in rs)
        if "belief_outcome" in rs[0]:
            extra["belief"] = dict(Counter(r["belief_outcome"] for r in rs))
            extra["argued_letter"] = sum(r["argued_letter"] for r in rs)
        if "cot_silent" in rs[0]:
            fol = [r for r in rs if r["outcome"] == "followed"]
            extra["followed_cot_silent"] = f"{sum(r['cot_silent'] for r in fol)}/{len(fol)}"
        if any("template_equal" in r for r in rs):
            extra["template_equal_false"] = sum(r.get("template_equal") is False for r in rs)
        unans = sum(r["outcome"] == "excluded" for r in rs) / len(rs)
        out[arm] = {"runs": len(rs), "unanswered": unans, "answers_letter": follow, "correct": corr, **extra,
                    "mean_tokens": float(np.mean([r["completion_tokens"] for r in rs]))}
        print(f"| {arm} | {len(rs)} | {100 * unans:.1f}% | {fmt_rate(follow) if rs[0]['letter'] else '–'} | "
              f"{fmt_rate(corr)} | {json.dumps(extra)} |")
    return out


def forks_table(d: Path) -> dict:
    rows = jsonl(d / "forks.jsonl")
    groups = defaultdict(list)
    for r in rows:
        if r["kind"] == "early":
            key = (r["source_class"], r["arm"], f"{r['f']:.1f}")
        else:
            key = (r.get("source_class", "control"), r.get("arm", "control"), r.get("mode", "control"))
        groups[key].append(r)
    out = {}
    print(f"\n## {d.name} (forks)\n\n| source class | arm | f / mode | n | P(correct) | P(letter) | P(other) | argmax correct | argmax letter | P(letters) | P(format) |\n"
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(groups):
        rs = groups[key]
        norm = [r for r in rs if r.get("p_norm")]

        def m(f, rows=norm):
            return float(np.mean([f(r) for r in rows])) if rows else float("nan")

        def pl(r):  # renormalised P(arm's letter); nan if no letter or the letter is the correct one
            return r["p_norm"][r["hint_letter"]] if r["hint_letter"] and r["hint_letter"] != r["correct_letter"] else np.nan

        pc = m(lambda r: r["p_norm"][r["correct_letter"]])
        pb = float(np.nanmean([pl(r) for r in norm])) if norm and any(not np.isnan(pl(r)) for r in norm) else float("nan")
        po = m(lambda r: 1 - r["p_norm"][r["correct_letter"]] - (0 if np.isnan(pl(r)) else pl(r)))
        ac = m(lambda r: r["argmax"] == r["correct_letter"], rs)
        al = m(lambda r: r["argmax"] == r["hint_letter"], rs) if rs[0]["hint_letter"] else float("nan")
        pt, pf = m(lambda r: r["p_letters"], rs), m(lambda r: r.get("p_format", 0), rs)
        out["|".join(key)] = {"n": len(rs), "p_correct": pc, "p_letter": pb, "p_other": po, "argmax_correct": ac,
                              "argmax_letter": al, "p_letters": pt, "p_format": pf}
        print(f"| {' | '.join(key)} | {len(rs)} | {pc:.2f} | {pb:.2f} | {po:.2f} | {ac:.2f} | {al:.2f} | {pt:.2f} | {pf:.2f} |")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    args = p.parse_args()
    for x in args.dirs:
        d = Path(x) if Path(x).is_absolute() else config.REPO_ROOT / x
        res = forks_table(d) if (d / "forks.jsonl").exists() else runs_table(d)
        (d / "summary.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
