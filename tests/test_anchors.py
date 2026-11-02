"""Needs the Qwen3-8B tokenizer (downloads ~10 MB); runs in either env."""

import pytest

transformers = pytest.importorskip("transformers")

from jmr import config  # noqa: E402
from jmr.anchors import find_anchors  # noqa: E402
from jmr.tokens import encode_chat, token_bytes  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    cfg = config.load("model")
    return transformers.AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"])


def test_token_bytes_roundtrip(tok):
    ids = tok("naïve café — 漢字 <think>x</think>", add_special_tokens=False).input_ids
    assert b"".join(token_bytes(tok, ids)) == tok.decode(ids).encode()


def test_anchors(tok):
    prompt = encode_chat(tok, [{"role": "user", "content": "Pick one."}], enable_thinking=True)
    completion = tok("<think>\nI think the answer is C here.\n</think>\n\nAnswer: C<|im_end|>",
                     add_special_tokens=False).input_ids
    ids = prompt + completion
    a = find_anchors(tok, ids, len(prompt), think_start_id=tok.convert_tokens_to_ids("<think>"),
                     think_end_id=tok.convert_tokens_to_ids("</think>"), final_letter="C")
    assert tok.decode([ids[a.think_start]]) == "<think>"
    assert tok.decode([ids[a.think_end]]) == "</think>"
    assert "C" in tok.decode([ids[a.final_answer]])
    assert "C" in tok.decode([ids[a.answer_first_stated]])
    assert a.think_start < a.answer_first_stated < a.think_end < a.final_answer


def test_boxed_final_answer(tok):
    prompt = encode_chat(tok, [{"role": "user", "content": "Pick one."}], enable_thinking=True)
    completion = tok("<think>\nB.\n</think>\n\nSo:\n\n$$\n\\boxed{B}\n$$<|im_end|>", add_special_tokens=False).input_ids
    ids = prompt + completion
    a = find_anchors(tok, ids, len(prompt), think_start_id=tok.convert_tokens_to_ids("<think>"),
                     think_end_id=tok.convert_tokens_to_ids("</think>"), final_letter="B")
    assert a.final_answer is not None and "B" in tok.decode([ids[a.final_answer]])
