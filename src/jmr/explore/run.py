"""Run the arms of one or more specs through one vLLM engine (gen env only)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from jmr import config
from jmr.explore.spec import (
    build_messages, dump_spec, hint_letters, item_rows, label_row, load_items, out_dir,
    resolve_arms, write_jsonl,
)
from jmr.gen.generate import Engine, Request, generate_requests, run_seed


def build_requests(spec: dict, arms: list[dict], questions) -> list[tuple[Request, dict]]:
    pending = []
    for arm in arms:
        for q in questions:
            for letter in hint_letters(arm, q):
                messages = build_messages(arm, q, letter)
                condition = arm["hint"]["name"] if arm["hint"] else "unhinted"
                for k in range(arm["samples"]):
                    run_id = f"{spec['name']}/{arm['name']}/{q.id}/{letter or '-'}/{k}"
                    req = Request(
                        run_id=run_id, run_set=spec["name"], question=q, condition=condition,
                        hint_letter=letter, messages=messages, enable_thinking=arm["thinking"],
                        sampling=arm["sampling"], seed=run_seed(spec["base_seed"], run_id), sample_idx=k,
                    )
                    pending.append((req, arm))
    return pending


def run_specs(engine: Engine, specs: list[dict], *, arm_filter: list[str] | None = None,
              chunk_size: int | None = None, deadline: float | None = None) -> None:
    """For each spec, write items, runs, labels and meta. Non-thinking arms of all
    specs run first, then thinking arms, so the throughput of each mode is measured
    separately. Within a mode, runs are ordered sample-major (all sample-0 runs of
    every arm and item, then sample 1, ...) and generated in chunks of `chunk_size`
    runs (default: one chunk per mode); files are rewritten after every chunk. With
    `deadline` (a time.time() value), no new chunk starts after it; meta.json then
    says so ("stopped_at_deadline") and lists the runs not generated."""
    plans = []
    for spec in specs:
        arms = [a for a in resolve_arms(spec) if not arm_filter or a["name"] in arm_filter]
        questions = load_items(spec)
        d = out_dir(spec)
        if (d / "runs.jsonl").exists():
            raise SystemExit(f"{d / 'runs.jsonl'} exists; use a new spec name or out_dir")
        d.mkdir(parents=True, exist_ok=True)
        dump_spec(spec, d / "spec.yaml")
        write_jsonl(d / "items.jsonl", item_rows(questions))
        plans.append({"spec": spec, "dir": d, "pending": build_requests(spec, arms, questions),
                      "rows": [], "stats": {}, "arms": [a["name"] for a in arms], "not_run": 0})
        print(f"{spec['name']}: {len(arms)} arms, {len(questions)} items, {len(plans[-1]['pending'])} runs", flush=True)

    stopped = False
    for thinking in (False, True):
        batch = [(i, req, arm) for i, p in enumerate(plans) for req, arm in p["pending"] if arm["thinking"] == thinking]
        batch.sort(key=lambda x: x[1].sample_idx)  # stable: sample-major
        if not batch:
            continue
        mode = "thinking" if thinking else "non_thinking"
        size = chunk_size or len(batch)
        totals = {"n_runs": 0, "elapsed_s": 0.0, "generated_tokens": 0, "chunks": 0}
        for start in range(0, len(batch), size):
            chunk = batch[start : start + size]
            if deadline is not None and time.time() > deadline:
                stopped = True
                for i, _, _ in batch[start:]:
                    plans[i]["not_run"] += 1
                print(f"{time.strftime('%H:%M:%S')} {mode}: deadline passed, {len(batch) - start} runs not generated", flush=True)
                break
            print(f"{time.strftime('%H:%M:%S')} {mode}: runs {start}-{start + len(chunk)} of {len(batch)}", flush=True)
            records, stats = generate_requests(engine, [req for _, req, _ in chunk])
            print(f"{time.strftime('%H:%M:%S')} {mode}: {json.dumps(stats)}", flush=True)
            for k in ("n_runs", "elapsed_s", "generated_tokens"):
                totals[k] += stats[k]
            totals["chunks"] += 1
            totals["tokens_per_s"] = round(totals["generated_tokens"] / max(totals["elapsed_s"], 1e-9), 1)
            totals["mean_completion_tokens"] = round(totals["generated_tokens"] / max(totals["n_runs"], 1), 1)
            touched = set()
            for (i, _, arm), rec in zip(chunk, records):
                plans[i]["rows"].append({**asdict(rec), "arm": arm["name"], "arm_spec": arm})
                touched.add(i)
            for i in touched:
                p = plans[i]
                p["stats"][mode] = {**totals, "note": "one engine call per chunk, shared by all specs in the engine"}
                _write(engine, specs, p, stopped)
    for p in plans:
        _write(engine, specs, p, stopped)


def _write(engine: Engine, specs: list[dict], p: dict, stopped: bool) -> None:
    write_jsonl(p["dir"] / "runs.jsonl", p["rows"])
    write_jsonl(p["dir"] / "labels.jsonl", [label_row(r) for r in p["rows"]])
    (p["dir"] / "meta.json").write_text(json.dumps({
        "specs_in_engine": [s["name"] for s in specs], "arms": p["arms"], "stats": p["stats"],
        "n_runs_planned": len(p["pending"]), "n_runs_done": len(p["rows"]),
        "stopped_at_deadline": stopped, "not_run": p["not_run"],
        "model": engine.model_cfg["hf_id"], "revision": engine.model_cfg["revision"],
        "tokenizer_fingerprint": engine.tokenizer_fingerprint, "git": config.git_state(),
    }, indent=2))
