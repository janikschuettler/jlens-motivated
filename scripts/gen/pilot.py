"""M1 behavioural pilot (gen env, GPU).

1. Draw items from the error-free MMLU-Redux pool.
2. Baseline: `samples` unhinted samples per item; keep the reliably known items.
3. On the kept items: `samples` samples in each of the five conditions.
4. Label the runs and write the audit file and the pilot summary.

uv run --project envs/gen python scripts/gen/pilot.py --run-set m1_smoke --n-items 8 --samples 3
uv run --project envs/gen python scripts/gen/pilot.py --run-set m1_pilot
"""

import argparse
import json

from jmr import config
from jmr.analysis.pilot import analyse, render_markdown
from jmr.gen.generate import build_engine, generate_runs, kv_cache_tokens
from jmr.prompts import NUDGES
from jmr.questions import load_questions
from jmr.records import runs_path, write_runs
from jmr.runset import baseline_stats, item_rows, items_path, label_run_set, read_items, write_items


def main() -> None:
    pilot = config.load("data")["pilot"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-set", required=True)
    parser.add_argument("--n-items", type=int, default=pilot["n_items"])
    parser.add_argument("--samples", type=int, default=pilot["samples"])
    parser.add_argument("--item-seed", type=int, default=pilot["item_seed"])
    parser.add_argument("--keep-fraction", type=float, default=pilot["keep_fraction"])
    parser.add_argument("--max-tokens", type=int, default=None, help="override sampling.yaml (smoke tests only)")
    parser.add_argument("--audit-per-group", type=int, default=5)
    args = parser.parse_args()

    out_dir = config.run_set_dir(args.run_set)
    if runs_path(out_dir).exists():
        raise SystemExit(f"{runs_path(out_dir)} exists; use a new run set name")
    config.snapshot(out_dir)

    questions, pool_size = load_questions(n=args.n_items, seed=args.item_seed)
    rows = item_rows(questions)
    write_items(out_dir, rows)
    engine = build_engine()

    # Phase 1: baseline on every item.
    baseline, stats_baseline = generate_runs(
        engine, questions, ["baseline"], args.run_set, samples=args.samples, max_tokens=args.max_tokens)
    stats = baseline_stats(baseline, args.keep_fraction)
    for row in rows:
        row.update(stats[row["id"]])
    write_items(out_dir, rows)
    write_runs(runs_path(out_dir), baseline)
    kept = [q for q in questions if stats[q.id]["kept"]]
    print(f"baseline done: {len(kept)} of {len(questions)} items reliably known")
    if not kept:
        raise SystemExit("no reliably known items; nothing to nudge")

    # Phase 2: the five conditions on the kept items.
    conditions, stats_conditions = generate_runs(
        engine, kept, ["unhinted", *NUDGES], args.run_set, samples=args.samples, max_tokens=args.max_tokens)
    write_runs(runs_path(out_dir), baseline + conditions)

    label_rows = label_run_set(args.run_set, audit_per_group=args.audit_per_group)
    summary = analyse(label_rows, read_items(out_dir), pool_size=pool_size)
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "pilot_summary.md").write_text(render_markdown(summary, args.run_set))

    meta = {"args": vars(args), "pool_size": pool_size, "stats_baseline": stats_baseline,
            "stats_conditions": stats_conditions, "kv_cache_tokens": kv_cache_tokens(engine), "git": config.git_state()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print((out_dir / "pilot_summary.md").read_text())
    print(f"files in {out_dir}: items.jsonl runs.jsonl labels.jsonl audit.md pilot_summary.md")


if __name__ == "__main__":
    main()
