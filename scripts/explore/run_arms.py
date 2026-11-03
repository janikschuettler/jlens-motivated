"""Run exploration arm specs (gen env, GPU). See jmr.explore for the spec format.

uv run --project envs/gen python scripts/explore/run_arms.py configs/explore/gap_step0.yaml
uv run --project envs/gen python scripts/explore/run_arms.py SPEC1 SPEC2 --arms ARM1 ARM2
uv run --project envs/gen python scripts/explore/run_arms.py SPEC --limit-items 4 --out-suffix _smoke
uv run --project envs/gen python scripts/explore/run_arms.py SPEC --chunk-size 1500 --deadline 00:45
"""

import argparse
import time

from jmr.explore.run import run_specs
from jmr.explore.spec import load_spec
from jmr.gen.generate import build_engine, kv_cache_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--arms", nargs="+", help="only these arm names")
    parser.add_argument("--limit-items", type=int, help="first N items only (smoke test)")
    parser.add_argument("--out-suffix", default="", help="appended to each spec's name and out_dir")
    parser.add_argument("--samples", type=int, help="override samples for every arm")
    parser.add_argument("--chunk-size", type=int, help="runs per engine call; files are rewritten after each")
    parser.add_argument("--deadline", help="HH:MM (local time, today or tomorrow): start no chunk after it")
    args = parser.parse_args()

    specs = []
    for path in args.specs:
        spec = load_spec(path)
        if args.limit_items:
            spec["items"]["limit"] = args.limit_items
        if args.samples:
            spec.setdefault("defaults", {})["samples"] = args.samples
            for arm in spec.get("arms", []) + spec.get("grids", []):
                arm["samples"] = args.samples
        if args.out_suffix:
            spec["name"] += args.out_suffix
            spec["out_dir"] += args.out_suffix
        specs.append(spec)
    deadline = _deadline(args.deadline) if args.deadline else None
    start = time.time()
    engine = build_engine()
    print(f"engine ready in {time.time() - start:.0f}s, KV cache tokens: {kv_cache_tokens(engine)}", flush=True)
    run_specs(engine, specs, arm_filter=args.arms, chunk_size=args.chunk_size, deadline=deadline)
    print(f"done in {time.time() - start:.0f}s", flush=True)


def _deadline(hhmm: str) -> float:
    now = time.localtime()
    h, m = map(int, hhmm.split(":"))
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
    return t if t > time.time() else t + 86400


if __name__ == "__main__":
    main()
