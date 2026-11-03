"""Run wave-2 arm specs (configs/explore/w2_*.yaml; gen env, GPU). Format: jmr.explore.w2.

uv run --project envs/gen python scripts/explore/run_w2.py configs/explore/w2_instructed.yaml configs/explore/w2_prefill.yaml
uv run --project envs/gen python scripts/explore/run_w2.py SPEC --dry-run            # CPU: build every prompt, print counts + GPU-min estimate
uv run --project envs/gen python scripts/explore/run_w2.py SPEC --limit-items 2 --samples 1 --out-suffix _smoke
uv run --project envs/gen python scripts/explore/run_w2.py SPEC --chunk-size 1500 --deadline 06:30
"""

import argparse
import time

from jmr.explore.w2 import estimate, load_w2_items, load_w2_spec, plan_requests, resolve_w2_arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--arms", nargs="+", help="only these arm names")
    parser.add_argument("--limit-items", type=int, help="first N items only (smoke test)")
    parser.add_argument("--items-path", help="replace items.path (known_pool / jsonl sources)")
    parser.add_argument("--out-suffix", default="", help="appended to each spec's name and out_dir")
    parser.add_argument("--samples", type=int, help="override samples for every arm")
    parser.add_argument("--chunk-size", type=int, help="runs per engine call; files are rewritten after each")
    parser.add_argument("--deadline", help="HH:MM (local time): start no chunk after it")
    parser.add_argument("--dry-run", action="store_true", help="tokenizer only: build prompts, print the plan, write nothing")
    parser.add_argument("--tok-s", type=float, default=2500.0, help="aggregate generated tokens/s for --dry-run estimates")
    args = parser.parse_args()

    specs = []
    for path in args.specs:
        spec = load_w2_spec(path)
        if args.limit_items:
            spec["items"]["limit"] = args.limit_items
        if args.items_path:
            spec["items"]["path"] = args.items_path
        if args.samples:
            spec.setdefault("defaults", {})["samples"] = args.samples
            for arm in spec.get("arms", []):
                arm["samples"] = args.samples
        if args.out_suffix:
            spec["name"] += args.out_suffix
            spec["out_dir"] += args.out_suffix
        specs.append(spec)

    if args.dry_run:
        dry_run(specs, args)
        return

    from jmr.explore.w2_run import run_w2_specs
    from jmr.gen.generate import build_engine, kv_cache_tokens

    deadline = _deadline(args.deadline) if args.deadline else None
    start = time.time()
    engine = build_engine()
    print(f"engine ready in {time.time() - start:.0f}s, KV cache tokens: {kv_cache_tokens(engine)}", flush=True)
    run_w2_specs(engine, specs, arm_filter=args.arms, chunk_size=args.chunk_size, deadline=deadline)
    print(f"done in {time.time() - start:.0f}s", flush=True)


def dry_run(specs: list[dict], args) -> None:
    from transformers import AutoTokenizer

    from jmr import config

    m = config.load("model")
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    total_min = 0.0
    for spec in specs:
        arms = [a for a in resolve_w2_arms(spec) if not args.arms or a["name"] in args.arms]
        qs = load_w2_items(spec)
        reqs = plan_requests(spec, arms, qs, tok)
        est = estimate(reqs, args.tok_s, {"thinking": 1600, "non_thinking": 500})
        mins = sum(v["gpu_min"] for v in est.values())
        total_min += mins
        print(f"\n== {spec['name']}: {len(arms)} arms, {len(qs)} items, {len(reqs)} runs, ~{mins:.1f} GPU-min "
              f"at {args.tok_s:.0f} tok/s (+ engine load and batch tail)")
        for name, v in est.items():
            print(f"  {name:28s} runs {v['runs']:5d}  est. tokens {v['tokens']:9d}  ~{v['gpu_min']:5.1f} min  "
                  f"mean prompt {v['prompt_tokens'] / max(v['runs'], 1):.0f} tok")
        shown = set()
        for r in reqs:
            if r.arm["name"] in shown:
                continue
            shown.add(r.arm["name"])
            print(f"\n--- {r.run_id}  letter={r.letter} info={ {k: v for k, v in r.plan.info.items() if k != 'prefill_token_ids'} }")
            print(tok.decode(r.plan.prompt_ids))
        bad = [r.run_id for r in reqs if r.plan.info.get("template_equal") is False]
        print(f"\ntemplate_equal False: {len(bad)} of {len(reqs)} {bad[:3]}")
    print(f"\nTOTAL ~{total_min:.1f} GPU-min of generation")


def _deadline(hhmm: str) -> float:
    now = time.localtime()
    h, m = map(int, hhmm.split(":"))
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
    return t if t > time.time() else t + 86400


if __name__ == "__main__":
    main()
