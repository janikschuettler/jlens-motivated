"""Judge chains of thought with an LLM via OpenRouter (envs/judge, arena9-cloud; no GPU).

Judge every nudged run of a run set (runs.jsonl must be on this machine):
  uv run --project envs/judge python scripts/judge.py --runs data/runs/<run_set>/runs.jsonl
Only some labels (reads labels.jsonl next to runs.jsonl):
  uv run --project envs/judge python scripts/judge.py --runs data/runs/<run_set>/runs.jsonl \
      --only-labels silent_switch,acknowledged_switch,resisted
The validation set (scripts/judge_eval/build_cases.py):
  uv run --project envs/judge python scripts/judge.py --cases data/judge_eval/cases.jsonl \
      --model openai/gpt-5-nano [--ids data/judge_eval/gold_ids.txt]

Writes <input dir>/judge_<model>.jsonl (one row per run: mentions_hint, stance,
argues_for_hinted, final_answer, case, evidence, ...) and judge_<model>_usage.json.
Responses are cached in data/judge_cache/, so re-runs cost nothing. Stops on the first
judgement that still fails after retries (CLAUDE.md rule 1).
"""

import argparse
import json
from pathlib import Path

from jmr import config
from jmr.judge import Judge, JudgeInput, input_from_record, user_message, SYSTEM
from jmr.prompts import NUDGES
from jmr.records import read_runs


def estimate_cost(inputs: list[JudgeInput], model: str) -> float:
    """Rough upper estimate: 3.5 characters per token in, max_tokens/4 out per call."""
    cfg = config.load("judge")
    m = cfg["models"][model]
    chars = sum(len(SYSTEM) + len(user_message(i)) for i in inputs)
    out_tokens = m.get("max_tokens", cfg["max_tokens"]) / 4
    return (chars / 3.5 * m["price_in"] + len(inputs) * out_tokens * m["price_out"]) / 1e6


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", type=Path, help="a runs.jsonl (nudged conditions are judged)")
    src.add_argument("--cases", type=Path, help="a JSONL of JudgeInput rows (the validation set)")
    p.add_argument("--model", default=None, help="OpenRouter model ID from configs/judge.yaml")
    p.add_argument("--only-labels", default=None, help="comma-separated labels from labels.jsonl")
    p.add_argument("--ids", type=Path, default=None, help="file of case/run IDs to judge")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-cost", type=float, default=1.0, help="refuse if the estimate exceeds this (USD)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    model = args.model or config.load("judge")["default_model"]
    extra = {}
    if args.runs:
        records = [r for r in read_runs(args.runs) if r.condition in NUDGES]
        if args.only_labels:
            keep = set(args.only_labels.split(","))
            labels = {json.loads(l)["run_id"]: json.loads(l)["label"]
                      for l in open(args.runs.parent / "labels.jsonl")}
            records = [r for r in records if labels[r.run_id] in keep]
        inputs = [input_from_record(r) for r in records]
        base = args.runs.parent
    else:
        fields = JudgeInput.__dataclass_fields__
        rows = [json.loads(l) for l in open(args.cases)]
        inputs = [JudgeInput(**{k: v for k, v in r.items() if k in fields}) for r in rows]
        extra = {r["case_id"]: {"parsed_answer": r.get("parsed_answer")} for r in rows}
        base = args.cases.parent
    if args.ids:
        ids = set(args.ids.read_text().split())
        inputs = [i for i in inputs if i.case_id in ids]
    if args.limit:
        inputs = inputs[: args.limit]

    est = estimate_cost(inputs, model)
    print(f"{len(inputs)} runs to judge with {model}; estimated cost at most ${est:.3f} (before cache)")
    if est > args.max_cost:
        raise SystemExit(f"estimate ${est:.3f} exceeds --max-cost ${args.max_cost:.2f}")

    judge = Judge(model)
    rows = judge.judge_many(inputs, workers=args.workers)
    for row in rows:
        row.update(extra.get(row["case_id"], {}))
    out = args.out or base / f"judge_{model.replace('/', '__')}.jsonl"
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    usage = {**judge.usage, "model": model, "n": len(rows),
             "cost_all_rows_usd": sum(r["cost_usd"] for r in rows)}
    Path(str(out).removesuffix(".jsonl") + "_usage.json").write_text(json.dumps(usage, indent=2))
    from collections import Counter
    print(f"wrote {out}")
    print("cases:", dict(sorted(Counter(r["case"] for r in rows).items())))
    print("usage this call:", json.dumps(judge.usage))
    print(f"cost of all {len(rows)} judgements (incl. cached): ${usage['cost_all_rows_usd']:.4f}; "
          f"per 1,000 runs: ${1000 * usage['cost_all_rows_usd'] / max(1, len(rows)):.2f}")


if __name__ == "__main__":
    main()
