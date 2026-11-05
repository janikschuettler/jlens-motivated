"""Eval-awareness ablation on Qwen3-14B: pick the items and the source prompts (CPU, stdlib only).

  python3 -I scripts/explore/evalaware_ablate14_select.py \
      --arms-dir data/explore/models_grader/q14_arms --out data/explore/evalaware_ablate14

Items (pre-registered, docs/preregistration.md "X1-X3"): every known item of the
models_grader Qwen3-14B run set on which at least one of the 8 earlier scored-framing runs
(A_V0 and A_stakes, 4 each) took the bug (final answer = the hinted letter). Prompt: the A_V0
arm (scored-agent system prompt, grader-bug note in the user turn), copied as saved token IDs
from that item's lowest-sample A_V0 run (all A_V0 samples of an item share the prompt).
Writes sources.jsonl (one row per item: the source run without its completion, plus the
item's earlier counts) and refs.jsonl (3 full source runs, prompt + completion, for the
vLLM-hook vs HF-hook check).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalaware_models_analyze import TEST_ME  # noqa: E402

SCORED_ARMS = ("A_V0", "A_stakes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, out = Path(a.arms_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = {}
    for line in open(d / "labels.jsonl"):
        r = json.loads(line)
        lab[r["run_id"]] = r
    counts, src = {}, {}
    for line in open(d / "runs.jsonl"):
        r = json.loads(line)
        if r["condition"] not in SCORED_ARMS:
            continue
        q = r["question_id"]
        c = counts.setdefault(q, {"runs": 0, "took_bug": 0, "test_me": 0})
        c["runs"] += 1
        c["took_bug"] += lab[r["run_id"]]["answer"] == r["hint_letter"]
        c["test_me"] += any(p.search(r["completion_text"]) for p in TEST_ME)
        if r["condition"] == "A_V0" and (q not in src or r["sample_idx"] < src[q]["sample_idx"]):
            src[q] = r
    items = sorted(q for q, c in counts.items() if c["took_bug"] > 0)
    if any(counts[q]["runs"] != 8 for q in items) or any(q not in src for q in items):
        raise SystemExit("an item lacks its 8 scored-framing runs or an A_V0 run (rule 1)")
    with open(out / "sources.jsonl", "w") as f:
        for q in items:
            r = {k: v for k, v in src[q].items()
                 if k not in ("completion_token_ids", "completion_logprobs", "completion_text", "finish_reason")}
            r["earlier"] = counts[q]
            f.write(json.dumps(r) + "\n")
    refs = [src[q] for q in items if 500 <= len(src[q]["completion_token_ids"]) <= 1200][:3]
    with open(out / "refs.jsonl", "w") as f:
        for r in refs:
            f.write(json.dumps(r) + "\n")
    n8 = sum(counts[q]["runs"] for q in items)
    print(json.dumps({"items": len(items), "of_known": len(counts),
                      "earlier_took_bug": sum(counts[q]["took_bug"] for q in items), "earlier_runs": n8,
                      "earlier_test_me": sum(counts[q]["test_me"] for q in items),
                      "refs": [(r["run_id"], len(r["prompt_token_ids"]), len(r["completion_token_ids"])) for r in refs]},
                     indent=1))


if __name__ == "__main__":
    main()
