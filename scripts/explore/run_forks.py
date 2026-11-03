"""Prefill-only forks of saved runs: E5 forced early answers, E7 re-elicitation
(gen env, GPU; format: jmr.explore.forks, specs configs/explore/w2_forks_*.yaml).

uv run --project envs/gen python scripts/explore/run_forks.py configs/explore/w2_forks_early.yaml
uv run --project envs/gen python scripts/explore/run_forks.py configs/explore/w2_forks_early.yaml \
    --runs data/explore/w2_prefill/runs.jsonl --name w2_forks_early_prefill   # replace the spec's sources
uv run --project envs/gen python scripts/explore/run_forks.py SPEC --dry-run   # CPU: plan forks, print counts and examples
"""

import argparse
import time

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--runs", nargs="+", help="runs.jsonl paths replacing the spec's sources (kind: runs)")
    parser.add_argument("--name", help="new spec name (and out_dir data/explore/<name>)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    if args.runs:
        spec["sources"] = [{"path": p} for p in args.runs]
    if args.name:
        spec["name"], spec["out_dir"] = args.name, f"data/explore/{args.name}"
    spec.setdefault("out_dir", f"data/explore/{spec['name']}")

    if args.dry_run:
        from transformers import AutoTokenizer

        from jmr import config
        from jmr.explore.forks import plan_forks

        m = config.load("model")
        tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
        forks, traces, skipped = plan_forks(spec, tok)
        n_tok = sum(len(f["prompt_ids"]) for f in forks)
        print(f"{spec['name']}: {len(traces)} traces, {len(forks)} forks, {n_tok} prompt tokens "
              f"(before prefix caching), skipped {skipped}, "
              f"boundary merges {sum(1 for f in forks if f['row'].get('boundary_merge'))}")
        for f in forks[:3] + forks[-2:]:
            print("\n---", f["row"]["fork_id"], "...", repr(tok.decode(f["prompt_ids"][-60:])))
        return

    from jmr.explore.w2_run import run_forks
    from jmr.gen.generate import build_engine

    start = time.time()
    engine = build_engine()
    run_forks(engine, spec)
    print(f"done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
