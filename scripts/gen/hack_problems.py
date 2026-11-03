"""Problem set for the reward-hack organism (setup 1). gen env, CPU only.

Streams deepmind/code_contests (train split, the stream order the organisms'
dataset builder used), keeps "hard" problems with tests that are not on the
upstream exclusion list, skips the first `--skip` qualifying problems (the
organisms' 412 training + 100 test problems were drawn from the first 512), and
writes the next `--n` to data/hack/items.jsonl, each with its fixed hint order
and the prompt length under the pinned tokenizer. Problems whose prompt exceeds
`--max-prompt-tokens` are not kept (counted in the summary).

uv run --project envs/gen python scripts/gen/hack_problems.py --exclusions <excluded_problem_ids.json> --n 150
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from jmr import config, hack
from jmr.tokens import encode_chat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclusions", required=True, help="upstream excluded_problem_ids.json")
    ap.add_argument("--skip", type=int, default=512)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--out", default="data/hack/items.jsonl")
    args = ap.parse_args()

    excl = set()
    for ids in json.loads(Path(args.exclusions).read_text()).values():
        excl |= {str(i) for i in ids}
    cfg = config.load("model")
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
    n_seen = n_qual = n_long = 0
    items = []
    for rec in ds:
        n_seen += 1
        if str(rec["name"]) in excl or not hack.is_hard(rec):
            continue
        tc = hack.test_cases(rec)
        if not tc:
            continue
        n_qual += 1
        if n_qual <= args.skip:
            continue
        tests = hack.test_list(tc)
        item = {"problem_id": str(rec["name"]), "description": rec["description"],
                "test_list": tests[: hack.MAX_TEST_CASES], "n_tests_total": len(tests),
                "cf_rating": int(rec["cf_rating"]), "difficulty": int(rec["difficulty"]),
                "source": int(rec["source"]), "hint_order": hack.hint_order(str(rec["name"])),
                "stream_index": n_seen - 1, "qualifying_index": n_qual - 1}
        # The prompt shows only the first 3 tests and the total count, so the stored 10 rebuild it exactly.
        if hack.user_prompt(rec["description"], tests) != hack.messages(item)[1]["content"]:
            raise RuntimeError(f"{item['problem_id']}: stored item does not rebuild the prompt")
        item["prompt_tokens"] = len(encode_chat(tok, hack.messages(item), enable_thinking=False))
        if item["prompt_tokens"] > args.max_prompt_tokens:
            n_long += 1
            continue
        items.append(item)
        if len(items) >= args.n:
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    summary = {"n_items": len(items), "stream_rows_read": n_seen, "qualifying_seen": n_qual,
               "skipped_first_qualifying": args.skip, "dropped_long_prompt": n_long,
               "prompt_tokens_max": max(i["prompt_tokens"] for i in items),
               "prompt_tokens_median": sorted(i["prompt_tokens"] for i in items)[len(items) // 2],
               "upstream": hack.UPSTREAM}
    (out.parent / "items_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
