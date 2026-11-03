"""Tokenizer helpers shared by both envs. Callers pass a loaded HF tokenizer."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from typing import Any

FINGERPRINT_TEXT = (
    "The quick brown fox, 3.14159, naïve café — 漢字 <think>?</think> Answer: (B)\n\n  x"
)
FINGERPRINT_MESSAGES = [
    {"role": "user", "content": "Which is larger? (A) 2 (B) 3"},
    {"role": "assistant", "content": "Answer: B"},
    {"role": "user", "content": "Again, please."},
]


@cache
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table used by byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tokenizer: Any, ids: list[int]) -> list[bytes]:
    """The exact bytes of each token, so text positions map back to token indices
    without re-tokenizing."""
    added = set(tokenizer.get_added_vocab())
    decoder = _byte_decoder()
    out = []
    for tid, token in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        if token is None:
            # An ID the tokenizer has no entry for. gpt-oss-20b's embedding has 201,088 rows but
            # its tokenizer only 200,019 entries; the model occasionally samples one of the
            # unused IDs, which decodes to "". It is kept as a zero-length piece so byte offsets
            # stay exact; any other unmapped ID is an error. (Qwen3 never hits this.)
            if tid < len(tokenizer) or tokenizer.decode([tid]) != "":
                raise ValueError(f"token ID {tid} has no token string but is inside the vocabulary")
            out.append(b"")
            continue
        if token in added:
            out.append(token.encode("utf-8"))
        else:
            out.append(bytes(decoder[ch] for ch in token))
    return out


def chat_text(tokenizer: Any, messages: list[dict], *, enable_thinking: bool) -> str:
    """The rendered chat. A model profile may set `chat_template_kwargs` (e.g.
    gpt-oss's reasoning_effort) and `fixed_current_date` (gpt-oss's template writes
    today's date into the system message; fixing it keeps prompts and the tokenizer
    fingerprint reproducible across days). Neither is set for Qwen3."""
    from jmr import config

    mcfg = config.load("model")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **mcfg.get("chat_template_kwargs", {}),
    )
    date = mcfg.get("fixed_current_date")
    if date:
        import re

        text, n = re.subn(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {date}", text, count=1)
        if n != 1:
            raise ValueError("fixed_current_date is set but the rendered chat has no 'Current date:' line")
    return text


def encode_chat(tokenizer: Any, messages: list[dict], *, enable_thinking: bool) -> list[int]:
    """Prompt token IDs for a chat. Only the gen env calls this; the lens env reads
    the saved IDs."""
    text = chat_text(tokenizer, messages, enable_thinking=enable_thinking)
    return tokenizer(text, add_special_tokens=False).input_ids


def fingerprint(tokenizer: Any) -> str:
    """Hash of the tokenizer's behaviour on fixed inputs (plain text and the chat
    template), so the two envs can confirm they tokenize identically."""
    payload = {
        "text": tokenizer(FINGERPRINT_TEXT, add_special_tokens=False).input_ids,
        "chat_thinking": encode_chat(tokenizer, FINGERPRINT_MESSAGES, enable_thinking=True),
        "chat_no_thinking": encode_chat(tokenizer, FINGERPRINT_MESSAGES, enable_thinking=False),
        "vocab_size": len(tokenizer),
    }
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def single_token_id(tokenizer: Any, text: str) -> int | None:
    """The token ID if text is exactly one token, else None."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None
