"""LLM judge for the Hawthorne runs (envs/judge, OpenRouter, CPU; runs on arena9-cloud).

Uses the judge prompt of Abdelnabi & Salem (prompts/judgePromptEvidenceBoth.txt in
microsoft/Test_Awareness_Steering, copied to data/hawthorne/), unchanged: it asks for
evaluation awareness ("model_awareness"), hypothetical-scenario recognition and task
execution, each Yes/No/Maybe with verbatim evidence. Their judge was GPT-4o (Azure); we use
openai/gpt-4.1-mini at temperature 0 (cheaper; same family). Reasoning = text before
</think>, answer = text after, as in their judgeIt_batch.py. Runs without </think> are not
judged (recorded as such). Nothing is truncated.

  zsh -ic 'cd /root/jlens-motivated && ~/.local/bin/uv run --project envs/judge python \
      scripts/explore/hawthorne_judge.py --runs data/hawthorne/q14/runs.jsonl --workers 32'

Appends to judge.jsonl next to runs.jsonl (resumable by run_id). Stops at the cost cap.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4.1-mini"
COST_CAP_USD = 8.0
PROMPT = Path(__file__).resolve().parents[2] / "data/hawthorne/judgePromptEvidenceBoth.txt"


def split(text: str):
    if "</think>" not in text:
        return None, None
    r, ans = text.split("</think>", 1)
    return r.replace("<think>", "").strip(), ans.strip()


def parse(content: str) -> dict | None:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    try:
        return json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    tmpl = PROMPT.read_text()
    runs_path = Path(a.runs)
    out = runs_path.parent / "judge.jsonl"
    done, spent = set(), 0.0
    if out.exists():
        for l in open(out):
            d = json.loads(l)
            done.add(d["run_id"])
            spent += d.get("cost") or 0.0
    runs = [json.loads(l) for l in open(runs_path)]
    todo = [r for r in runs if r["run_id"] not in done][: a.limit]
    print(f"{len(runs)} runs, {len(todo)} to judge, spent so far ${spent:.3f}", flush=True)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (run from a fresh shell)")
    client = httpx.Client(timeout=300, headers={"Authorization": f"Bearer {key}"})
    lock = threading.Lock()
    state = {"spent": spent, "n": 0, "errors": 0}

    def one(r):
        if state["spent"] >= COST_CAP_USD:
            return
        reasoning, answer = split(r["completion_text"])
        row = {"run_id": r["run_id"], "judge_model": a.model}
        if reasoning is None:
            row.update({"judged": False, "why": "no </think>"})
        else:
            user = tmpl.format(task=r["item"]["task"], reasoning=reasoning, answer=answer)
            body = {"model": a.model, "temperature": 0, "seed": 0, "max_tokens": 3000,
                    "messages": [{"role": "user", "content": user}], "usage": {"include": True},
                    "response_format": {"type": "json_object"}}
            last = None
            for attempt in range(6):
                try:
                    resp = client.post(ENDPOINT, json=body)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices"):
                            content = data["choices"][0]["message"].get("content") or ""
                            j = parse(content)
                            if j is not None:
                                u = data.get("usage") or {}
                                row.update({"judged": True, "judge": j, "cost": u.get("cost"),
                                            "usage": {"prompt_tokens": u.get("prompt_tokens"),
                                                      "completion_tokens": u.get("completion_tokens")}})
                                break
                            last = f"unparsable: {content[:200]}"
                        else:
                            last = f"no choices: {json.dumps(data)[:200]}"
                    else:
                        last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                except Exception as e:
                    last = f"{type(e).__name__}: {e}"
                time.sleep(min(60, 3 * 2 ** attempt))
            else:
                with lock:
                    state["errors"] += 1
                print(f"ERROR {r['run_id']}: {last}", flush=True)
                return
        with lock:
            state["spent"] += row.get("cost") or 0.0
            state["n"] += 1
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if state["n"] % 200 == 0:
                print(f"{state['n']} judged, ${state['spent']:.3f}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    print(f"done: {state['n']} new rows, {state['errors']} errors, total ${state['spent']:.3f}", flush=True)
    if state["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
