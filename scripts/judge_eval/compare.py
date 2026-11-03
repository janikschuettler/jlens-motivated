"""Compare judge models with the gold labels, with each other and with the regex check.

uv run --project envs/judge python scripts/judge_eval/compare.py \
    --models google/gemini-2.5-flash-lite openai/gpt-5-nano [anthropic/claude-sonnet-5.5]

Reads data/judge_eval/{cases.jsonl, gold_labels.jsonl, judge_<model>.jsonl}; a model's
file may cover only some cases (e.g. the stronger rater on the gold subset). Prints
markdown tables and writes data/judge_eval/results.json.
"""

import argparse
import json
from collections import Counter
from itertools import combinations

from jmr import config

D = config.DATA_DIR / "judge_eval"
CASES = [0, 1, 2, 3, 4]


def regex_case(c: dict) -> int:
    """The case the existing regex label implies (mention = any pattern matched)."""
    m = bool(c["regex_mentions"])
    return {"silent_switch": 4, "acknowledged_switch": 3}.get(
        c["regex_label"], (2 if m else 1) if c["regex_label"] == "resisted" else (0 if m else 1))


def kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def agree(a: list, b: list) -> dict:
    n = len(a)
    return {"n": n, "agree": sum(x == y for x, y in zip(a, b)), "pct": 100 * sum(x == y for x, y in zip(a, b)) / n if n else None,
            "kappa": kappa(a, b)}


def fmt(r: dict) -> str:
    k = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
    return f"{r['agree']}/{r['n']} ({r['pct']:.0f}%), κ {k}"


def confusion(gold: list, pred: list, labels=CASES) -> str:
    head = "| gold \\ judge | " + " | ".join(str(l) for l in labels) + " |\n|---|" + "---:|" * len(labels) + "\n"
    rows = []
    for g in labels:
        rows.append(f"| **{g}** | " + " | ".join(str(sum(1 for x, y in zip(gold, pred) if x == g and y == p))
                                                  for p in labels) + " |")
    return head + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    args = ap.parse_args()

    cases = {json.loads(l)["case_id"]: json.loads(l) for l in open(D / "cases.jsonl")}
    gold = {json.loads(l)["case_id"]: json.loads(l) for l in open(D / "gold_labels.jsonl")}
    judged = {}
    for m in args.models:
        path = D / f"judge_{m.replace('/', '__')}.jsonl"
        judged[m] = {json.loads(l)["case_id"]: json.loads(l) for l in open(path)}
    regex = {cid: {"case": regex_case(c), "mentions_hint": bool(c["regex_mentions"])} for cid, c in cases.items()}

    results = {"vs_gold": {}, "pairwise": {}, "checks": {}}
    fields = ["case", "mentions_hint", "stance", "argues_for_hinted", "final_answer"]
    explicit = [cid for cid in gold if not gold[cid].get("borderline")]

    print("## Against the gold labels (Claude-agent labels, not human)\n")
    print("| rater | n | case (0–4) | mentions_hint | stance | argues_for_hinted | final_answer | case, excl. borderline |")
    print("|---|---:|---|---|---|---|---|---|")
    for name, rows in [("regex (existing)", regex)] + list(judged.items()):
        ids = [cid for cid in gold if cid in rows]
        res = {}
        for f in fields:
            if f in ("stance", "argues_for_hinted", "final_answer") and name.startswith("regex"):
                continue
            res[f] = agree([gold[i][f] for i in ids], [rows[i][f] for i in ids])
        ids_x = [i for i in explicit if i in rows]
        res["case_explicit"] = agree([gold[i]["case"] for i in ids_x], [rows[i]["case"] for i in ids_x])
        results["vs_gold"][name] = res
        cells = [fmt(res[f]) if f in res else "–" for f in fields]
        print(f"| {name} | {len(ids)} | " + " | ".join(cells) + f" | {fmt(res['case_explicit'])} |")

    for name, rows in [("regex (existing)", regex)] + list(judged.items()):
        ids = [cid for cid in gold if cid in rows]
        print(f"\n### Confusion, cases: gold × {name} (n = {len(ids)})\n")
        print(confusion([gold[i]["case"] for i in ids], [rows[i]["case"] for i in ids]))
        dis = [i for i in ids if gold[i]["case"] != rows[i]["case"]]
        if dis and name in judged:
            print(f"\nDisagreements ({len(dis)}):")
            for i in dis:
                r = rows[i]
                print(f"- `{i}` gold {gold[i]['case']} ({gold[i]['stance']}), judge {r['case']} ({r['stance']}, "
                      f"argues_B={r['argues_for_hinted']}, final={r['final_answer']}): {r['rationale'][:220]}")

    print("\n## Between raters (all cases both judged)\n")
    print("| pair | n | case | mentions_hint | stance | argues_for_hinted | final_answer |")
    print("|---|---:|---|---|---|---|---|")
    raters = {"regex (existing)": regex, **judged}
    for (a, ra), (b, rb) in combinations(raters.items(), 2):
        ids = [cid for cid in cases if cid in ra and cid in rb]
        res = {}
        for f in fields:
            if f in ra[ids[0]] and f in rb[ids[0]]:
                res[f] = agree([ra[i][f] for i in ids], [rb[i][f] for i in ids])
        results["pairwise"][f"{a} | {b}"] = res
        print(f"| {a} vs {b} | {len(ids)} | " + " | ".join(fmt(res[f]) if f in res else "–" for f in fields) + " |")

    for (a, ra), (b, rb) in combinations(judged.items(), 2):
        ids = [cid for cid in cases if cid in ra and cid in rb]
        print(f"\n### Confusion, cases: {a} (rows) × {b} (columns), n = {len(ids)}\n")
        print(confusion([ra[i]["case"] for i in ids], [rb[i]["case"] for i in ids]).replace("gold \\ judge", f"{a.split('/')[-1]} \\ {b.split('/')[-1]}"))

    print("\n## Checks\n")
    print("| model | n | final_answer = parser | evidence quotes verbatim | judge prompt tokens / Qwen completion tokens (thinking runs, median) | cost per 1,000 runs, thinking | non-thinking | all |")
    print("|---|---:|---|---|---|---:|---:|---:|")
    for m, rows in judged.items():
        ids = list(rows)
        fa = sum(rows[i]["final_answer"] == cases[i]["parsed_answer"] for i in ids)
        quotes = [v for i in ids for v in rows[i]["evidence_verbatim"]]
        ratios = sorted(rows[i]["prompt_tokens"] / cases[i]["completion_tokens"] for i in ids if cases[i]["thinking_mode"])
        med = ratios[len(ratios) // 2] if ratios else float("nan")
        th = [rows[i]["cost_usd"] for i in ids if cases[i]["thinking_mode"]]
        nt = [rows[i]["cost_usd"] for i in ids if not cases[i]["thinking_mode"]]
        allc = th + nt
        per = lambda xs: 1000 * sum(xs) / len(xs) if xs else float("nan")
        results["checks"][m] = {"n": len(ids), "final_answer_matches_parser": fa, "quotes": len(quotes),
                                "quotes_verbatim": sum(quotes), "median_prompt_ratio_thinking": med,
                                "min_prompt_ratio_thinking": ratios[0] if ratios else None,
                                "cost_per_1000_thinking": per(th), "cost_per_1000_nonthinking": per(nt),
                                "cost_per_1000_all": per(allc), "total_cost": sum(allc),
                                "prompt_tokens": sum(rows[i]["prompt_tokens"] for i in ids),
                                "completion_tokens": sum(rows[i]["completion_tokens"] for i in ids),
                                "reported_models": dict(Counter(rows[i]["judge_model_reported"] for i in ids)),
                                "providers": dict(Counter(rows[i]["judge_provider"] for i in ids))}
        print(f"| {m} | {len(ids)} | {fa}/{len(ids)} | {sum(quotes)}/{len(quotes)} | {med:.2f} (min {ratios[0]:.2f}) | "
              f"${per(th):.2f} | ${per(nt):.2f} | ${per(allc):.2f} |")
    print()
    for m, c in results["checks"].items():
        print(f"- {m}: total ${c['total_cost']:.4f}, {c['prompt_tokens']:,} prompt + {c['completion_tokens']:,} completion tokens; "
              f"reported model(s) {c['reported_models']}; providers {c['providers']}")

    print("\n## Case counts on all cases\n")
    for name, rows in raters.items():
        print(f"- {name}: {dict(sorted(Counter(r['case'] for r in rows.values()).items()))}")
    (D / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
