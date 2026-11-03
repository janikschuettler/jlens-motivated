"""Anchor planning for batch readouts (jmr.lens.batch). Lens env only (imports
torch and jlens); needs the Qwen3-8B tokenizer. No GPU."""

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("jlens")

from jmr import config  # noqa: E402
from jmr.lens.batch import check_plan, plan_run, tracked_tokens  # noqa: E402
from jmr.records import RunRecord  # noqa: E402
from jmr.tokens import encode_chat  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    cfg = config.load("model")
    return transformers.AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"])


def _record(tok, completion: str, *, thinking: bool) -> RunRecord:
    msgs = [{"role": "user", "content": "Pick one.\n\n(A) x\n(B) y\n(C) z\n(D) w"}]
    prompt = encode_chat(tok, msgs, enable_thinking=thinking)
    comp = tok(completion, add_special_tokens=False).input_ids
    return RunRecord(run_id="t", run_set="t", question_id="q", dataset="d", condition="unhinted", hint_letter=None,
                     correct_letter="C", letters=list("ABCD"), messages=msgs, prompt_token_ids=prompt,
                     completion_token_ids=comp, completion_logprobs=[0.0] * len(comp), completion_text=completion,
                     finish_reason="stop", seed=0, model={}, tokenizer_fingerprint="", sampling={},
                     enable_thinking=thinking)


def _plan(tok, r):
    ids = (tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>"))
    return plan_run(r, tok, think_ids=ids, prompt_tail=64, n_cot=16, resid_anchors=["last_prompt", "think_end",
                                                                                    "pre_answer"])


def test_tracked_tokens_single(tok):
    t = tracked_tokens(tok)
    assert len(t.ids) == 12 and t.labels[t.index("B", "space")] == " B"


def test_plan_thinking(tok):
    r = _record(tok, "<think>\nLet me see. " + "Hmm. " * 40 + "I think the answer is C.\n</think>\n\nAnswer: C<|im_end|>",
                thinking=True)
    p = _plan(tok, r)
    ids, P = r.token_ids, len(r.prompt_token_ids)
    assert not check_plan(p, tok)
    assert p.named["think_start"] == P and tok.decode([ids[p.named["think_end"]]]) == "</think>"
    assert tok.decode([ids[p.named["pre_answer"] + 1]]) == " C"
    assert tok.decode([ids[p.named["pre_first_stated"] + 1]]) == " C"
    cot = [p.named[f"cot_{k:02d}"] for k in range(16)]
    assert cot[0] == p.cot_span[0] == P + 1 and cot[-1] == p.cot_span[1] - 1 == p.named["think_end"] - 1
    assert p.stored_lo == max(0, P - 64)


def test_plan_non_thinking(tok):
    r = _record(tok, "Option C fits best.\n\nAnswer: C<|im_end|>", thinking=False)
    p = _plan(tok, r)
    P = len(r.prompt_token_ids)
    assert not check_plan(p, tok)
    # The empty think block is in the prompt, so no think anchors in the completion.
    assert p.named["think_start"] == -1 and p.named["think_end"] == -1
    assert p.cot_span == (P, p.named["pre_answer"] + 1)
    assert tok.decode([r.token_ids[p.named["pre_answer"] + 1]]) == " C"
