"""Temperature check (gen env, GPU): a few reliably known items, many samples, two temperatures.

1. Draw candidate items (seed 1, disjoint from the smoke run's seed 0), screen them with
   5 baseline samples at the lower temperature, keep those with 5 of 5 correct, take the first `--n-items`.
2. For each temperature: `unhinted`, `null_control` and `known_evidence`, 25 samples per item (50 for the first item).
3. Label the runs, write audit.md and temperature_report.md.

One vLLM engine runs both temperatures (temperature is a per-request setting).

uv run --project envs/gen python scripts/gen/temperature_check.py --run-set temp_check
"""

import argparse
import json

from jmr import config
from jmr.analysis.temperature import render
from jmr.gen.generate import build_engine, generate_runs
from jmr.questions import load_questions
from jmr.records import runs_path, write_runs
from jmr.runset import baseline_stats, item_rows, label_run_set, read_items, write_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-set", required=True)
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=30)
    parser.add_argument("--item-seed", type=int, default=1)
    parser.add_argument("--temperatures", type=float, nargs=2, default=[0.6, 1.0])
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--samples-main", type=int, default=50, help="for the first item")
    parser.add_argument("--conditions", nargs="+", default=["unhinted", "null_control", "known_evidence"])
    parser.add_argument("--screen-samples", type=int, default=5)
    args = parser.parse_args()

    out_dir = config.run_set_dir(args.run_set)
    if runs_path(out_dir).exists():
        raise SystemExit(f"{runs_path(out_dir)} exists; use a new run set name")
    config.snapshot(out_dir)
    low, high = sorted(args.temperatures)

    candidates, pool_size = load_questions(n=args.n_candidates, seed=args.item_seed)
    engine = build_engine()

    screen_runs, _ = generate_runs(engine, candidates, ["baseline"], args.run_set, samples=args.screen_samples,
                                   overrides={"temperature": low}, arm=f"screen{low}")
    stats = baseline_stats(screen_runs, 1.0)
    screen = [{"id": q.id, **stats[q.id]} for q in candidates]
    chosen = [q for q in candidates if stats[q.id]["kept"]][: args.n_items]
    print(f"screen: {sum(s['kept'] for s in screen)} of {len(candidates)} candidates answered {args.screen_samples} of {args.screen_samples}")
    if len(chosen) < args.n_items:
        raise SystemExit(f"only {len(chosen)} items passed the screen, need {args.n_items}; draw more candidates")
    main_item = chosen[0].id
    samples = {q.id: (args.samples_main if q.id == main_item else args.samples) for q in chosen}

    rows = item_rows(chosen)
    for row in rows:
        row.update(stats[row["id"]])
    write_items(out_dir, rows)

    all_runs, all_stats = list(screen_runs), {}
    for t in (low, high):
        runs, st = generate_runs(engine, chosen, args.conditions, args.run_set, samples=samples,
                                 overrides={"temperature": t}, arm=f"t{t}")
        all_runs += runs
        all_stats[str(t)] = st
        write_runs(runs_path(out_dir), all_runs)  # keep partial results if the second arm fails

    label_rows = label_run_set(args.run_set, audit_per_group=5)
    experiment_rows = [r for r in label_rows if r["condition"] != "baseline"]
    report = render(experiment_rows, read_items(out_dir), screen=screen, main_item=main_item, conditions=args.conditions)
    (out_dir / "temperature_report.md").write_text(report)
    (out_dir / "meta.json").write_text(json.dumps(
        {"args": vars(args), "pool_size": pool_size, "stats": all_stats, "screen": screen, "git": config.git_state()}, indent=2))
    print(report)


if __name__ == "__main__":
    main()
