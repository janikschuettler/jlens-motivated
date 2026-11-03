"""Subtle-cue arms (configs/explore/subtle_cues.yaml; gen env, GPU): few-shot "answer always B",
visual marker, metadata tag, grader code. Prompts are built per item here (few-shot needs per-target
example orderings), then generated and written exactly like run_w2.py (jmr.explore.w2_run: same
runs.jsonl / labels.jsonl / items.jsonl / meta.json, saved token IDs, sample-major chunks, deadline).

uv run --project envs/gen python scripts/explore/run_subtle.py configs/explore/subtle_cues.yaml --dry-run
uv run --project envs/gen python scripts/explore/run_subtle.py configs/explore/subtle_cues.yaml --limit-items 2 --samples 1 --out-suffix _smoke
uv run --project envs/gen python scripts/explore/run_subtle.py configs/explore/subtle_cues.yaml --chunk-size 560 --deadline 11:58
"""

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import yaml

from jmr import config
from jmr.explore.spec import dump_spec, item_rows, write_jsonl
from jmr.explore.turns import plain_prompt
from jmr.explore.w2 import W2Request, _jsonl, question_from_row, run_seed
from jmr.prompts import Question, hinted_letter

MENTION_GROUPS = {"none": [], "fewshot": ["fewshot"], "fewshot_always": ["fewshot"],
                  "fewshot_marker": ["fewshot", "marker"], "metadata": ["metadata"], "grader": ["grader"],
                  "marker_only": ["marker"]}


def load(spec: dict) -> tuple[list[Question], list[Question]]:
    src = spec["items"]
    rows = [r for r in _jsonl(src["path"]) if (r.get("acc") or {}).get(src["setting"], 0) >= src["min_acc"]]
    qs = [question_from_row(r) for r in rows]
    if any(len(q.letters) != 4 for q in qs):
        raise ValueError("expected 4 options everywhere")
    # examples: shortest items, distinct subjects, distinct correct letters
    by_len = sorted(qs, key=lambda q: (len(q.question) + sum(len(o) for o in q.options), q.id))
    ex, subj, letters = [], set(), set()
    for q in by_len:
        if q.subject in subj or q.correct in letters:
            continue
        if any(re.search(r"\b(?:above|both|none|all of|neither)\b", o, re.I) for o in q.options):
            continue  # options that refer to other options don't survive reordering
        ex.append(q)
        subj.add(q.subject)
        letters.add(q.correct)
        if len(ex) == src["n_examples"]:
            break
    rest = [q for q in qs if q.id not in {e.id for e in ex}]
    random.Random(src["target_seed"]).shuffle(rest)
    off = src.get("target_offset", 0)
    return ex, rest[off : off + src["n_targets"]]


def options_text(spec: dict, letters, options, marked: str | None = None) -> str:
    return "\n".join(spec["option"].format(letter=x, text=(spec["marker"] if x == marked else "") + t)
                     for x, t in zip(letters, options))


def reorder_to(q: Question, target: str) -> tuple[tuple[str, ...], str]:
    """Options with the correct one swapped into position `target`; returns (options, new correct letter)."""
    opts = list(q.options)
    i, j = q.letters.index(q.correct), q.letters.index(target)
    opts[i], opts[j] = opts[j], opts[i]
    return tuple(opts), target


def qid(q: Question) -> str:
    return f"q{int(hashlib.sha256(q.id.encode()).hexdigest()[:8], 16) % 100000:05d}"


def user_text(spec: dict, cue: str, q: Question, b: str, examples: list[Question]) -> str:
    parts = []
    if cue in ("fewshot", "fewshot_always", "fewshot_marker"):
        blocks = []
        for e in examples:
            opts, ans = (reorder_to(e, b) if cue == "fewshot_always" else (e.options, e.correct))
            ot = options_text(spec, e.letters, opts, marked=ans if cue == "fewshot_marker" else None)
            blocks.append(spec["example"].format(question_block=spec["question_block"].format(question=e.question.strip(), options=ot),
                                                 answer=ans))
        parts.append(spec["example_sep"].join(blocks) + spec["example_sep"])
    prefix = ""
    if cue == "metadata":
        prefix = spec["metadata"].format(qid=qid(q), letter=b)
    elif cue == "grader":
        prefix = spec["grader"].format(letter=b)
    marked = b if cue in ("fewshot_marker", "marker_only") else None
    target = spec["question_block"].format(question=q.question, options=options_text(spec, q.letters, q.options, marked))
    return prefix + "".join(parts) + target + "\n\n" + spec["instruction"]


def arms(spec: dict) -> list[dict]:
    out = []
    for a in spec["arms"]:
        if spec.get("only_arms") and a["name"] not in spec["only_arms"]:
            continue
        groups = MENTION_GROUPS[a["cue"]]
        out.append({"name": a["name"], "cue": a["cue"], "thinking": True, "sampling": dict(spec["sampling"]),
                    "samples": spec["samples"], "hint": None, "letter": "hinted",
                    "mention_groups": groups,
                    "mention_patterns": [p for g in groups for p in spec["mentions"][g]], "labels": ["concealment"],
                    "prefill": None, "turns": []})
    return out


def plan(spec: dict, tok, limit: int | None, samples: int | None) -> tuple[list[W2Request], list[Question], list[Question]]:
    ex, targets = load(spec)
    if limit:
        targets = targets[:limit]
    reqs = []
    for arm in arms(spec):
        if samples:
            arm["samples"] = samples
        for q in targets:
            b = hinted_letter(q)
            msgs = [{"role": "user", "content": user_text(spec, arm["cue"], q, b, ex)}]
            for k in range(arm["samples"]):
                p = plain_prompt(tok, msgs, enable_thinking=True)
                p.info["letter_shown"] = arm["cue"] != "none" and arm["cue"] != "fewshot"
                p.info["cue"] = arm["cue"]
                run_id = f"{spec['name']}/{arm['name']}/{q.id}/{b}/{k}"
                reqs.append(W2Request(run_id, spec["name"], q, arm, b, k, run_seed(spec["base_seed"], run_id), p))
    return reqs, ex, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--limit-items", type=int)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--deadline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", nargs="+", help="only these arms")
    ap.add_argument("--target-offset", type=int, help="targets = shuffled pool[offset : offset + n_targets]")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text())
    if args.arms:
        spec["only_arms"] = args.arms
    if args.target_offset is not None:
        spec["items"]["target_offset"] = args.target_offset
    spec["name"] += args.out_suffix
    spec["out_dir"] += args.out_suffix
    d = config.REPO_ROOT / spec["out_dir"]

    if args.dry_run:
        from transformers import AutoTokenizer

        m = config.load("model")
        tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
        reqs, ex, targets = plan(spec, tok, args.limit_items, args.samples)
        print("examples:", [(e.id, e.correct) for e in ex])
        print(f"{len(targets)} targets, {len(reqs)} runs")
        seen = set()
        for r in reqs:
            if r.arm["name"] in seen:
                continue
            seen.add(r.arm["name"])
            print(f"\n=== {r.run_id} B={r.letter} correct={r.question.correct} prompt_tokens={len(r.plan.prompt_ids)}")
            print(tok.decode(r.plan.prompt_ids))
        return

    from jmr.explore.w2_run import _write, generate_planned
    from jmr.gen.generate import build_engine, kv_cache_tokens

    if (d / "runs.jsonl").exists():
        raise SystemExit(f"{d / 'runs.jsonl'} exists")
    start = time.time()
    engine = build_engine()
    print(f"engine ready in {time.time() - start:.0f}s, KV cache tokens: {kv_cache_tokens(engine)}", flush=True)
    reqs, ex, targets = plan(spec, engine.tokenizer, args.limit_items, args.samples)
    d.mkdir(parents=True, exist_ok=True)
    dump_spec(spec, d / "spec.yaml")
    write_jsonl(d / "items.jsonl", item_rows(targets))
    write_jsonl(d / "examples.jsonl", item_rows(ex))
    p = {"spec": spec, "dir": d, "pending": reqs, "rows": [], "stats": {}, "arms": [a["name"] for a in arms(spec)],
         "texts": {q.id: dict(zip(q.letters, q.options)) for q in targets}, "not_run": 0}
    reqs.sort(key=lambda r: r.sample_idx)
    deadline = None
    if args.deadline:
        now = time.localtime()
        h, mi = map(int, args.deadline.split(":"))
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, mi, 0, 0, 0, -1))
    size = args.chunk_size or len(reqs)
    totals = {"n_runs": 0, "elapsed_s": 0.0, "generated_tokens": 0, "chunks": 0}
    stopped = False
    for s in range(0, len(reqs), size):
        if deadline is not None and time.time() > deadline:
            stopped = True
            p["not_run"] = len(reqs) - s
            print(f"{time.strftime('%H:%M:%S')} deadline passed, {len(reqs) - s} runs not generated", flush=True)
            break
        chunk = reqs[s : s + size]
        print(f"{time.strftime('%H:%M:%S')} runs {s}-{s + len(chunk)} of {len(reqs)}", flush=True)
        rows, stats = generate_planned(engine, chunk)
        print(f"{time.strftime('%H:%M:%S')} {json.dumps(stats)}", flush=True)
        for k in ("n_runs", "elapsed_s", "generated_tokens"):
            totals[k] += stats[k]
        totals["chunks"] += 1
        totals["tokens_per_s"] = round(totals["generated_tokens"] / max(totals["elapsed_s"], 1e-9), 1)
        totals["mean_completion_tokens"] = round(totals["generated_tokens"] / max(totals["n_runs"], 1), 1)
        p["rows"] += rows
        p["stats"]["thinking"] = totals
        _write(engine, [spec], p, stopped)
    _write(engine, [spec], p, stopped)
    print(f"done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
