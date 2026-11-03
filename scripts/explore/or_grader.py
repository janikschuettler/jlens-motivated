"""Models x grader-bug nudges: API screen via OpenRouter (envs/judge, runs on arena9-cloud).

Spec and exact texts: configs/explore/models_grader.yaml. Report:
docs/overnight/reports/models_grader.md.

  uv run --project envs/judge python scripts/explore/or_grader.py screen --model gpt-oss-20b
  uv run --project envs/judge python scripts/explore/or_grader.py arms --model gpt-oss-20b \
      --arms V0,V1_null --samples 1 [--start-sample 0] [--items known|all] [--limit-items N]
  uv run --project envs/judge python scripts/explore/or_grader.py cost
  uv run --project envs/judge python scripts/explore/or_grader.py endpoints --model gpt-oss-20b

The API key is read from $OPENROUTER_API_KEY (run from a fresh shell: zsh -ic '...').
It is never printed or stored. Every call's usage (tokens, usage.cost) is stored with
its run; every batch appends a line to <out_dir>/cost_log.jsonl. Submissions stop when
the study total reaches cost_cap_usd - cost_margin_usd. Runs that fail after retries
are written to errors.jsonl and reported; they are not marked done, and the batch
exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from jmr.label import ANSWER_PATTERNS  # noqa: E402

CFG = yaml.safe_load(open(REPO / "configs/explore/models_grader.yaml"))
OUT = REPO / CFG["out_dir"]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LOCK = threading.Lock()
N_ITEMS = [CFG["n_items"]]  # --n-items overrides (e.g. the wider Gemma-3-12B swap screen)


# --- items and prompts --------------------------------------------------------

def load_items() -> list[dict]:
    rows = [json.loads(line) for line in open(REPO / CFG["items"])]
    rows = [r for r in rows if r["acc"]["A"] >= 0.9]
    for r in rows:
        assert r["correct"] in r["letters"] and r["hinted"] in r["letters"] and r["hinted"] != r["correct"]
    return rows[: N_ITEMS[0]]


def build_messages(item: dict, arm: str) -> tuple[list[dict], str | None]:
    spec = CFG["arms"][arm]
    letter = item["correct"] if spec.get("letter") == "correct" else item["hinted"]
    options = "\n".join(CFG["option_template"].format(letter=l, text=t)
                        for l, t in zip(item["letters"], item["options"]))
    hint = CFG["hint_slot"].format(hint=spec["hint"].format(letter=letter)) if spec.get("hint") else ""
    user = CFG["user_template"].format(question=item["question"], options=options, hint=hint)
    msgs = []
    if spec.get("system"):
        msgs.append({"role": "system", "content": spec["system"].format(letter=letter)})
    msgs.append({"role": "user", "content": user})
    named = letter if (spec.get("hint") or spec.get("system")) and "{letter}" in (
        (spec.get("hint") or "") + (spec.get("system") or "")) else None
    return msgs, named


def extract_answer(response: str | None, letters) -> str | None:
    if not response:
        return None
    for pattern in ANSWER_PATTERNS:
        found = [m.group(1) for m in pattern.finditer(response) if m.group(1) in letters]
        if found:
            return found[-1]
    return None


# --- storage and cost ---------------------------------------------------------

def runs_path(model: str) -> Path:
    return OUT / model / "runs.jsonl"


def load_runs(model: str) -> dict[str, dict]:
    p = runs_path(model)
    if not p.exists():
        return {}
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["run_id"]] = r
    return out


def study_cost() -> float:
    total = 0.0
    for p in OUT.glob("*/runs.jsonl"):
        for line in open(p):
            total += json.loads(line)["usage"].get("cost") or 0.0
    for p in OUT.glob("judge*.jsonl"):
        for line in open(p):
            total += (json.loads(line).get("usage") or {}).get("cost") or 0.0
    for p in OUT.glob("*/errors.jsonl"):
        for line in open(p):
            total += (json.loads(line).get("usage") or {}).get("cost") or 0.0
    return total


class Spend:
    def __init__(self):
        self.base = study_cost()
        self.batch = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "reasoning_tokens": 0, "cost": 0.0}

    def total(self) -> float:
        return self.base + self.batch["cost"]

    def add(self, usage: dict):
        with LOCK:
            self.batch["calls"] += 1
            self.batch["prompt_tokens"] += usage.get("prompt_tokens") or 0
            self.batch["completion_tokens"] += usage.get("completion_tokens") or 0
            self.batch["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens") or 0
            self.batch["cost"] += usage.get("cost") or 0.0

    def over(self) -> bool:
        return self.total() >= CFG["cost_cap_usd"] - CFG["cost_margin_usd"]


# --- requests -----------------------------------------------------------------

_client = None


def client():
    global _client
    if _client is None:
        import httpx
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set (run from a fresh shell)")
        _client = httpx.Client(timeout=600, headers={"Authorization": f"Bearer {key}"})
    return _client


def request_body(model: str, msgs: list[dict], run_id: str) -> dict:
    m = CFG["models"][model]
    body = {"model": m["id"], "messages": msgs, "max_tokens": m["max_tokens"],
            "usage": {"include": True},
            "seed": int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)}
    body.update(m["sampling"])
    if m.get("reasoning"):
        body["reasoning"] = m["reasoning"]
    if m.get("provider"):
        body["provider"] = m["provider"]
    return body


def call(body: dict, tries: int = 8) -> dict:
    last = None
    for attempt in range(tries):
        try:
            r = client().post(ENDPOINT, json=body)
            if r.status_code == 200:
                data = r.json()
                if "choices" in data and data["choices"]:
                    return data
                last = f"no choices: {json.dumps(data)[:300]}"
            else:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # network errors: retry, then report
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(90, 5 * 2 ** attempt))
    raise RuntimeError(last)


def split_reasoning(msg: dict) -> tuple[str, str]:
    reasoning = msg.get("reasoning") or ""
    if not reasoning and msg.get("reasoning_details"):
        reasoning = "\n".join(d.get("text") or d.get("summary") or "" for d in msg["reasoning_details"])
    content = msg.get("content") or ""
    if "</think>" in content:  # provider returned the thinking inline
        head, content = content.split("</think>", 1)
        reasoning = (reasoning + "\n" + head.replace("<think>", "")).strip()
    return reasoning, content


def do_run(model: str, item: dict, arm: str, sample: int, spend: Spend) -> dict:
    run_id = f"{model}/{arm}/{item['id']}/{sample}"
    msgs, named = build_messages(item, arm)
    body = request_body(model, msgs, run_id)
    t0 = time.time()
    data = call(body)
    choice = data["choices"][0]
    reasoning, content = split_reasoning(choice.get("message") or {})
    usage = data.get("usage") or {}
    spend.add(usage)
    return {
        "run_id": run_id, "model": model, "or_id": CFG["models"][model]["id"], "arm": arm,
        "item_id": item["id"], "subject": item["subject"], "sample": sample,
        "correct": item["correct"], "hinted": item["hinted"], "letter_in_prompt": named,
        "letters": item["letters"], "messages": msgs,
        "request": {k: v for k, v in body.items() if k not in ("messages",)},
        "served_model": data.get("model"), "provider": data.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "reasoning": reasoning, "content": content,
        "answer": extract_answer(content, item["letters"]) if choice.get("finish_reason") == "stop" else None,
        "usage": {"prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens"),
                  "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                  "cost": usage.get("cost")},
        "seconds": round(time.time() - t0, 1),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_batch(model: str, jobs: list[tuple[dict, str, int]], label: str, workers: int) -> int:
    done = load_runs(model)
    todo = [(it, arm, s) for it, arm, s in jobs if f"{model}/{arm}/{it['id']}/{s}" not in done]
    spend = Spend()
    print(f"[{model}] {label}: {len(jobs)} jobs, {len(todo)} to do; study cost so far ${spend.base:.4f}",
          flush=True)
    if spend.over():
        print("cost cap reached; nothing submitted", flush=True)
        return 2
    path = runs_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors, stopped = [], False
    # unbuffered binary append: one write() per line, so concurrent appenders don't interleave
    with ThreadPoolExecutor(workers) as ex, open(path, "ab", buffering=0) as f:
        futs = {}
        it_jobs = iter(todo)
        def submit_next():
            nonlocal stopped
            if stopped:
                return
            if spend.over():
                stopped = True
                return
            j = next(it_jobs, None)
            if j is not None:
                futs[ex.submit(do_run, model, *j, spend)] = j
        for _ in range(workers):
            submit_next()
        n = 0
        while futs:
            fut = next(as_completed(list(futs)))
            j = futs.pop(fut)
            try:
                row = fut.result()
                with LOCK:
                    f.write((json.dumps(row) + "\n").encode())
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)} done, batch ${spend.batch['cost']:.4f}", flush=True)
            except Exception as e:
                errors.append({"run_id": f"{model}/{j[1]}/{j[0]['id']}/{j[2]}", "error": str(e),
                               "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            submit_next()
    if errors:
        with open(path.parent / "errors.jsonl", "a") as g:
            for e in errors:
                g.write(json.dumps(e) + "\n")
    log = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "label": label,
           **spend.batch, "errors": len(errors), "stopped_at_cap": stopped,
           "study_total": round(spend.total(), 5)}
    with open(OUT / "cost_log.jsonl", "a") as g:
        g.write(json.dumps(log) + "\n")
    print(json.dumps(log), flush=True)
    if errors:
        print(f"{len(errors)} errors, first: {errors[0]['error']}", flush=True)
        return 1
    return 2 if stopped else 0


def known_items(model: str) -> list[dict]:
    runs = load_runs(model)
    items = load_items()
    n = CFG["screen_samples"]
    out = []
    for it in items:
        rs = [runs.get(f"{model}/unhinted/{it['id']}/{s}") for s in range(n)]
        if all(r is not None for r in rs) and all(r["answer"] == it["correct"] for r in rs):
            out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "arms", "cost", "endpoints"])
    ap.add_argument("--model")
    ap.add_argument("--arms", default="")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--start-sample", type=int, default=0)
    ap.add_argument("--items", choices=["known", "all"], default="known")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None, help="pool size (default: config n_items)")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.n_items:
        N_ITEMS[0] = a.n_items
    if a.cmd == "cost":
        print(f"study cost ${study_cost():.4f} of cap ${CFG['cost_cap_usd']}")
        return 0
    if a.cmd == "endpoints":
        import httpx
        mid = CFG["models"][a.model]["id"]
        d = httpx.get(f"https://openrouter.ai/api/v1/models/{mid}/endpoints", timeout=60).json()
        eps = [{k: e.get(k) for k in ("provider_name", "quantization", "context_length", "max_completion_tokens",
                                       "supported_parameters", "pricing")} for e in d["data"]["endpoints"]]
        (OUT / a.model).mkdir(parents=True, exist_ok=True)
        json.dump(eps, open(OUT / a.model / "endpoints.json", "w"), indent=1)
        for e in eps:
            print(e["provider_name"], e["quantization"], e["pricing"]["prompt"], e["pricing"]["completion"],
                  "reasoning" in (e["supported_parameters"] or []), "top_k" in (e["supported_parameters"] or []))
        return 0
    if a.cmd == "screen":
        items = load_items()[: a.limit_items]
        jobs = [(it, "unhinted", s) for it in items for s in range(CFG["screen_samples"])]
        return run_batch(a.model, jobs, "screen", a.workers)
    items = known_items(a.model) if a.items == "known" else load_items()
    items = items[: a.limit_items]
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        assert arm in CFG["arms"], arm
    jobs = [(it, arm, s) for s in range(a.start_sample, a.start_sample + a.samples)
            for arm in arms for it in items]
    return run_batch(a.model, jobs, f"arms {','.join(arms)} s{a.start_sample}+{a.samples} on {len(items)} items",
                     a.workers)


if __name__ == "__main__":
    sys.exit(main())
