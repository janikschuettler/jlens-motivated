"""Build a global concept-token file for readout_runs.py --concept-tokens.

  uv run --project envs/lens python scripts/lens/concept_tokens.py --out configs/concept_tokens_grader.json

Candidate words are fixed here, before any readout. A word is kept only if " word"
is exactly one token of the model's tokenizer (round trip included); the rest are
listed as not single tokens in the file. Uses the model in configs/model.yaml (or
$JMR_MODEL_PROFILE).
"""

import argparse
import json

from transformers import AutoTokenizer

from jmr import config

# grader / cheating concepts, and truth / correctness concepts (fixed 2026-11-03,
# models_grader study, before any readout of those runs)
WORDS = {
    "grader": ["grader", "grading", "bug", "glitch", "exploit", "cheat", "cheating", "hack", "trick", "loophole",
               "score", "points", "reward", "marks", "rules", "allowed", "forbidden", "honest", "dishonest"],
    "truth": ["correct", "incorrect", "wrong", "right", "true", "false", "actually", "really", "truth"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = config.load("model")
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    kept, not_single = [], []
    for group, words in WORDS.items():
        for w in words:
            text = " " + w
            enc = tok(text, add_special_tokens=False).input_ids
            if len(enc) == 1 and tok.decode(enc) == text:
                kept.append({"text": text, "id": enc[0], "group": group})
            else:
                not_single.append({"text": text, "ids": enc, "group": group})
    out = {"description": "global concept tokens (scripts/lens/concept_tokens.py)",
           "tokenizer": {"hf_id": m["hf_id"], "revision": m["revision"]},
           "tokens": kept, "not_single_tokens": not_single}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"kept {len(kept)}: {[t['text'] for t in kept]}")
    print(f"not single tokens {len(not_single)}: {[t['text'] for t in not_single]}")


if __name__ == "__main__":
    main()
