"""Eval-awareness replication on Qwen3.6-27B: API runs (OpenRouter, envs/judge, arena9-cloud, no GPU).

Pre-registration: docs/overnight/preregistration.md, "EA27". Report:
docs/overnight/reports/evalaware_27b_replication.md.

  zsh -ic 'cd /root/jlens-motivated && ~/.local/bin/uv run --project envs/judge python \
      scripts/explore/ea27_api.py screen --first 24 --last 104'
  zsh -ic '... ea27_api.py arms --arm V0 --samples 2'

Items, prompts, sampling and the provider are those of the models-grader study
(scripts/explore/or_grader.py, configs/explore/models_grader.yaml, model qwen3.6-27b: Chutes,
T 0.6, top-p 0.95, top-k 20, reasoning on, max 8,192 tokens); only the output directory and the
cost cap differ. Runs go to data/ea27/api/runs.jsonl. "Known" (fixed before the screen):
every unhinted sample correct; 2 new samples per item for items 24-103 of the pool, the
earlier 5-sample screen for items 0-23 (data/models_grader/qwen3.6-27b/runs.jsonl).
Cost cap for this study: $2.40 (stop submitting at the cap).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("or_grader", REPO / "scripts/explore/or_grader.py")
og = importlib.util.module_from_spec(spec)
spec.loader.exec_module(og)

MODEL = "qwen3.6-27b"
OUT = REPO / "data/ea27/api"
OLD = REPO / "data/models_grader/qwen3.6-27b/runs.jsonl"
CAP = 2.40
SCREEN_SAMPLES = 2
LOCK = threading.Lock()


def pool():
    og.N_ITEMS[0] = 10_000
    return og.load_items()  # file order, Qwen3-8B setting-A accuracy >= 0.9 (164 items)


def load(path):
    return {json.loads(l)["run_id"]: json.loads(l) for l in open(path)} if path.exists() else {}


def spent():
    return sum((r["usage"].get("cost") or 0.0) for r in load(OUT / "runs.jsonl").values())


def known():
    items = pool()
    new, old = load(OUT / "runs.jsonl"), load(OLD)
    out = []
    for i, it in enumerate(items):
        if i < 24:
            rs = [old.get(f"{MODEL}/unhinted/{it['id']}/{s}") for s in range(5)]
        else:
            rs = [new.get(f"{MODEL}/unhinted/{it['id']}/{s}") for s in range(SCREEN_SAMPLES)]
        if all(r is not None for r in rs) and all(r["answer"] == it["correct"] for r in rs):
            out.append(it)
    return out


def run(jobs, workers=24):
    OUT.mkdir(parents=True, exist_ok=True)
    done = set(load(OUT / "runs.jsonl")) | set(load(OLD))
    todo = [j for j in jobs if f"{MODEL}/{j[1]}/{j[0]['id']}/{j[2]}" not in done]
    base = spent()
    batch = {"cost": 0.0, "calls": 0}
    print(f"{len(jobs)} jobs, {len(todo)} to do; spent so far ${base:.4f}", flush=True)

    class S:  # the Spend interface do_run expects
        def add(self, usage):
            with LOCK:
                batch["cost"] += usage.get("cost") or 0.0
                batch["calls"] += 1

    errors, it = [], iter(todo)
    with ThreadPoolExecutor(workers) as ex, open(OUT / "runs.jsonl", "ab", buffering=0) as f:
        futs = {}

        def submit():
            if base + batch["cost"] >= CAP:
                return
            j = next(it, None)
            if j is not None:
                futs[ex.submit(og.do_run, MODEL, *j, S())] = j

        for _ in range(workers):
            submit()
        n = 0
        while futs:
            fut = next(as_completed(list(futs)))
            j = futs.pop(fut)
            try:
                row = fut.result()
                row["study"] = "ea27"
                with LOCK:
                    f.write((json.dumps(row) + "\n").encode())
                n += 1
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)} ${base + batch['cost']:.3f}", flush=True)
            except Exception as e:
                errors.append({"job": f"{j[1]}/{j[0]['id']}/{j[2]}", "error": str(e)})
            submit()
    log = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **batch, "errors": len(errors),
           "total": round(base + batch["cost"], 4), "at_cap": base + batch["cost"] >= CAP}
    with open(OUT / "cost_log.jsonl", "a") as g:
        g.write(json.dumps(log) + "\n")
    if errors:
        with open(OUT / "errors.jsonl", "a") as g:
            for e in errors:
                g.write(json.dumps(e) + "\n")
    print(json.dumps(log), flush=True)
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "arms", "known", "cost"])
    ap.add_argument("--first", type=int, default=24)
    ap.add_argument("--last", type=int, default=104)
    ap.add_argument("--arm", default="V0")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--start-sample", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "cost":
        print(f"ea27 API spend ${spent():.4f} of ${CAP}")
        return 0
    if a.cmd == "known":
        k = known()
        print(len(k), "known items")
        (OUT / "known_items.json").write_text(json.dumps([it["id"] for it in k]))
        return 0
    if a.cmd == "screen":
        items = pool()[a.first:a.last]
        return run([(it, "unhinted", s) for s in range(SCREEN_SAMPLES) for it in items])
    items = known()
    return run([(it, a.arm, s) for s in range(a.start_sample, a.start_sample + a.samples) for it in items])


if __name__ == "__main__":
    sys.exit(main())
