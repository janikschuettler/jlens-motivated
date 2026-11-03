"""Run wave-2 specs and fork specs (gen env only: imports vllm via jmr.gen.generate).

Generation mirrors jmr.explore.run.run_specs (same files, same sample-major
chunking and deadline), but sends prebuilt prompt token IDs (prefill, spliced
on-policy turns) instead of rendering messages inside the generate call.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from vllm import SamplingParams, TokensPrompt

from jmr import config
from jmr.explore.forks import fork_seed, letter_probs, plan_forks
from jmr.explore.spec import dump_spec, item_rows, out_dir, write_jsonl
from jmr.explore.w2 import W2Request, load_w2_items, plan_requests, resolve_w2_arms
from jmr.explore.w2_labels import w2_label
from jmr.gen.generate import Engine, sampling_params
from jmr.records import RunRecord


def generate_planned(engine: Engine, requests: list[W2Request]) -> tuple[list[dict], dict]:
    prompts = [TokensPrompt(prompt_token_ids=r.plan.prompt_ids) for r in requests]
    params = [sampling_params(r.arm["sampling"], r.seed) for r in requests]
    start = time.time()
    outputs = engine.llm.generate(prompts, params) if requests else []
    elapsed = time.time() - start
    rows = []
    for req, out in zip(requests, outputs):
        if list(out.prompt_token_ids) != req.plan.prompt_ids:
            raise RuntimeError(f"{req.run_id}: vLLM's prompt token IDs differ from the ones sent")
        c = out.outputs[0]
        ids = list(c.token_ids)
        rec = RunRecord(
            run_id=req.run_id, run_set=req.run_set, question_id=req.question.id, dataset=req.question.dataset,
            condition=req.arm["name"], hint_letter=req.letter, correct_letter=req.question.correct,
            letters=list(req.question.letters), messages=req.plan.messages, prompt_token_ids=req.plan.prompt_ids,
            completion_token_ids=ids, completion_logprobs=[s[t].logprob for s, t in zip(c.logprobs, ids)],
            completion_text=c.text, finish_reason=c.finish_reason, seed=req.seed,
            model={"hf_id": engine.model_cfg["hf_id"], "revision": engine.model_cfg["revision"]},
            tokenizer_fingerprint=engine.tokenizer_fingerprint, sampling=req.arm["sampling"],
            enable_thinking=req.plan.enable_thinking, sample_idx=req.sample_idx,
        )
        rows.append({**asdict(rec), "arm": req.arm["name"], "arm_spec": req.arm, "w2": req.plan.info})
    gen = sum(len(r["completion_token_ids"]) for r in rows)
    return rows, {"n_runs": len(rows), "elapsed_s": round(elapsed, 1), "generated_tokens": gen,
                  "tokens_per_s": round(gen / elapsed, 1) if elapsed else None}


def run_w2_specs(engine: Engine, specs: list[dict], *, arm_filter=None, chunk_size=None, deadline=None) -> None:
    plans = []
    for spec in specs:
        arms = [a for a in resolve_w2_arms(spec) if not arm_filter or a["name"] in arm_filter]
        questions = load_w2_items(spec)
        d = out_dir(spec)
        if (d / "runs.jsonl").exists():
            raise SystemExit(f"{d / 'runs.jsonl'} exists; use a new spec name or out_dir")
        reqs = plan_requests(spec, arms, questions, engine.tokenizer)
        d.mkdir(parents=True, exist_ok=True)
        dump_spec(spec, d / "spec.yaml")
        write_jsonl(d / "items.jsonl", item_rows(questions))
        texts = {q.id: dict(zip(q.letters, q.options)) for q in questions}
        plans.append({"spec": spec, "dir": d, "pending": reqs, "rows": [], "stats": {}, "texts": texts,
                      "arms": [a["name"] for a in arms], "not_run": 0})
        print(f"{spec['name']}: {len(arms)} arms, {len(questions)} items, {len(reqs)} runs", flush=True)

    stopped = False
    for thinking in (False, True):
        batch = [(i, r) for i, p in enumerate(plans) for r in p["pending"] if r.plan.enable_thinking == thinking]
        batch.sort(key=lambda x: x[1].sample_idx)
        if not batch:
            continue
        mode = "thinking" if thinking else "non_thinking"
        size = chunk_size or len(batch)
        totals = {"n_runs": 0, "elapsed_s": 0.0, "generated_tokens": 0, "chunks": 0}
        for start in range(0, len(batch), size):
            chunk = batch[start : start + size]
            if deadline is not None and time.time() > deadline:
                stopped = True
                for i, _ in batch[start:]:
                    plans[i]["not_run"] += 1
                print(f"{time.strftime('%H:%M:%S')} {mode}: deadline passed, {len(batch) - start} runs not generated", flush=True)
                break
            print(f"{time.strftime('%H:%M:%S')} {mode}: runs {start}-{start + len(chunk)} of {len(batch)}", flush=True)
            rows, stats = generate_planned(engine, [r for _, r in chunk])
            print(f"{time.strftime('%H:%M:%S')} {mode}: {json.dumps(stats)}", flush=True)
            for k in ("n_runs", "elapsed_s", "generated_tokens"):
                totals[k] += stats[k]
            totals["chunks"] += 1
            totals["tokens_per_s"] = round(totals["generated_tokens"] / max(totals["elapsed_s"], 1e-9), 1)
            totals["mean_completion_tokens"] = round(totals["generated_tokens"] / max(totals["n_runs"], 1), 1)
            touched = set()
            for (i, _), row in zip(chunk, rows):
                plans[i]["rows"].append(row)
                touched.add(i)
            for i in touched:
                plans[i]["stats"][mode] = {**totals, "note": "one engine call per chunk, shared by all specs in the engine"}
                _write(engine, specs, plans[i], stopped)
    for p in plans:
        _write(engine, specs, p, stopped)


def _label(row: dict, texts: dict) -> dict:
    rec = RunRecord(**{k: v for k, v in row.items() if k in RunRecord.__dataclass_fields__})
    return w2_label(row, rec, correct_text=texts.get(rec.question_id, {}).get(rec.correct_letter))


def _write(engine: Engine, specs: list[dict], p: dict, stopped: bool) -> None:
    write_jsonl(p["dir"] / "runs.jsonl", p["rows"])
    write_jsonl(p["dir"] / "labels.jsonl", [_label(r, p["texts"]) for r in p["rows"]])
    (p["dir"] / "meta.json").write_text(json.dumps({
        "specs_in_engine": [s["name"] for s in specs], "arms": p["arms"], "stats": p["stats"],
        "n_runs_planned": len(p["pending"]), "n_runs_done": len(p["rows"]),
        "stopped_at_deadline": stopped, "not_run": p["not_run"],
        "template_equal_false": sum(1 for r in p["rows"] if r["w2"].get("template_equal") is False),
        "model": engine.model_cfg["hf_id"], "revision": engine.model_cfg["revision"],
        "tokenizer_fingerprint": engine.tokenizer_fingerprint, "git": config.git_state(),
    }, indent=2))


# ---- forks -----------------------------------------------------------------------


def run_forks(engine: Engine, spec: dict, *, top_logprobs: int = 20) -> None:
    d = out_dir(spec)
    if (d / "forks.jsonl").exists():
        raise SystemExit(f"{d / 'forks.jsonl'} exists; use a new spec name or out_dir")
    forks, traces, skipped = plan_forks(spec, engine.tokenizer)
    # forks of one trace next to each other, so prefix caching can reuse them
    forks.sort(key=lambda x: (x["row"].get("source_run_id", ""), x["row"].get("f", 0)))
    d.mkdir(parents=True, exist_ok=True)
    dump_spec(spec, d / "spec.yaml")
    print(f"{spec['name']}: {len(traces)} traces, {len(forks)} forks, skipped {skipped}", flush=True)
    params = [SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_logprobs, seed=fork_seed(f["row"]["fork_id"]),
                             skip_special_tokens=False) for f in forks]
    start = time.time()
    outs = engine.llm.generate([TokensPrompt(prompt_token_ids=f["prompt_ids"]) for f in forks], params) if forks else []
    elapsed = time.time() - start
    strings: dict[int, str] = {}
    rows = []
    for f, out in zip(forks, outs):
        if list(out.prompt_token_ids) != f["prompt_ids"]:
            raise RuntimeError(f"{f['row']['fork_id']}: vLLM's prompt token IDs differ from the ones sent")
        step = out.outputs[0].logprobs[0]
        top = sorted(((tid, lp.logprob) for tid, lp in step.items()), key=lambda x: -x[1])
        for tid, _ in top:
            if tid not in strings:
                strings[tid] = engine.tokenizer.decode([tid])
        lp = letter_probs(top, strings, f["row"]["letters"])
        rows.append({**f["row"], "prompt_len": len(f["prompt_ids"]), "next_token": out.outputs[0].token_ids[0],
                     "top_ids": [t for t, _ in top], "top_logprobs": [round(x, 5) for _, x in top],
                     "top_strings": [strings[t] for t, _ in top], **lp})
    write_jsonl(d / "forks.jsonl", rows)
    (d / "meta.json").write_text(json.dumps({
        "n_traces": len(traces), "n_forks": len(rows), "skipped": skipped, "elapsed_s": round(elapsed, 1),
        "prompt_tokens": sum(r["prompt_len"] for r in rows),
        "boundary_merges": sum(1 for r in rows if r.get("boundary_merge")),
        "model": engine.model_cfg["hf_id"], "revision": engine.model_cfg["revision"],
        "tokenizer_fingerprint": engine.tokenizer_fingerprint, "git": config.git_state(),
    }, indent=2))
    print(f"{spec['name']}: {len(rows)} forks in {elapsed:.0f}s", flush=True)
