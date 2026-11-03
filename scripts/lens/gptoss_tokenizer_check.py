"""CPU checks of the gpt-oss-20b tokenizer and chat template for the project pipeline
(any env with transformers; JMR_MODEL_PROFILE=gpt-oss-20b).

  JMR_MODEL_PROFILE=gpt-oss-20b HF_HUB_CACHE=/dev/shm/hf/hub uv run --project envs/lens \
      python scripts/lens/gptoss_tokenizer_check.py

Prints: the letter-token IDs for the tracked forms (bare "X", space " X", paren "(X")
with a single-token round-trip check, the special-token IDs the harmony anchors use,
one rendered prompt (with a system message), whether token_bytes reproduces the
rendered text exactly, and the tokenizer fingerprint. Exits non-zero on a failed check.
"""

import json
import sys

from transformers import AutoTokenizer

from jmr import config
from jmr.tokens import chat_text, encode_chat, fingerprint, token_bytes


def main():
    m = config.load("model")
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    ok = True
    ids = {}
    for L in "ABCD":
        ids[L] = {}
        for form, text in (("bare", L), ("space", " " + L), ("paren", "(" + L)):
            enc = tok(text, add_special_tokens=False).input_ids
            good = len(enc) == 1 and tok.decode(enc) == text
            ok &= good
            ids[L][form] = enc[0] if good else enc
    print("letters:", json.dumps(ids))
    for t in ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|call|>", "<|constrain|>"):
        print("special", t, tok.convert_tokens_to_ids(t))
    msgs = [{"role": "system", "content": "You are an AI agent being evaluated."},
            {"role": "user", "content": "Q?\n\n(A) x\n(B) y\n\nAnswer: X"}]
    text = chat_text(tok, msgs, enable_thinking=True)
    print("rendered:", repr(text))
    pids = encode_chat(tok, msgs, enable_thinking=True)
    rt = b"".join(token_bytes(tok, pids)) == text.encode()
    ok &= rt
    print("token_bytes round trip:", rt)
    # a synthetic harmony completion, tokenized, then anchor/label parsing
    comp = "<|channel|>analysis<|message|>The answer is A. But the grader marks B.<|end|><|start|>assistant<|channel|>final<|message|>Answer: B<|return|>"
    cids = tok(comp, add_special_tokens=False).input_ids
    rt2 = b"".join(token_bytes(tok, cids)) == comp.encode()
    ok &= rt2
    print("completion round trip:", rt2, "n tokens", len(cids))
    from jmr.anchors import find_anchors
    a = find_anchors(tok, pids + cids, len(pids), think_start_id=tok.convert_tokens_to_ids("<|message|>"),
                     think_end_id=tok.convert_tokens_to_ids("<|end|>"), final_letter="B", harmony=True)
    full = pids + cids
    print("anchors:", a.as_dict(), {k: tok.decode([full[v]]) for k, v in a.as_dict().items() if isinstance(v, int) and v < len(full) and k != "prompt_len"})
    good = (tok.decode([full[a.think_start]]) == "<|message|>" and tok.decode([full[a.think_end]]) == "<|end|>"
            and tok.decode([full[a.final_answer]]).strip() == "B" and a.answer_first_stated is None)
    ok &= good
    print("anchor check:", good)
    print("fingerprint:", fingerprint(tok))
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
