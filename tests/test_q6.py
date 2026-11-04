"""Point of no return (Q6): cuts, selection, outcomes, commitment rule. The
tokenizer tests use the pinned Qwen3-8B tokenizer from the local HF cache and are
skipped if it isn't there."""

import subprocess
import sys

import pytest

from jmr import config
from jmr.explore import q6
from jmr.records import RunRecord


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    m = config.load("model")
    try:
        return transformers.AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"], local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer not cached: {e}")


def test_sentence_candidates_on_bytes():
    pieces = [b"<think>", b"\n", b"Okay", b".", b" The", b" value", b" is", b" 3", b".", b"5", b".", b"\n\n",
              b"So", b" (", b"B", b")", b" is", b" it", b"?\"", b" Yes", b"</think>"]
    # span: after <think>\n (2) to </think> (20)
    c = q6.sentence_candidates(pieces, 2, 20)
    assert c[0] == 2 and c[-1] == 20
    assert 4 in c  # after "Okay." before " The"
    assert 9 not in c  # "3." followed by "5": a decimal, not a sentence end
    assert 11 not in c  # "." followed by "\n\n" token: the boundary is after the newline token
    assert 12 in c  # after "\n\n"
    assert 19 in c  # after '?"' before " Yes"


def test_make_cuts_snaps_and_dedupes():
    cands = [10, 12, 30, 31, 90, 100]
    cuts = q6.make_cuts(cands, 10, 100, [0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
    pos = [c.pos for c in cuts]
    assert pos[0] == 10 and pos[-1] == 100
    assert len(pos) == len(set(pos))
    assert all(a < b for a, b in zip(pos, pos[1:]))
    assert [c.idx for c in cuts] == list(range(len(cuts)))
    # f = 0.5 -> target 55: nearest of 31 (24 away) and 90 (35 away) is 31
    assert 31 in pos
    assert cuts[0].sentence == 0 and cuts[-1].frac == 1.0


def test_commitment_cut():
    t = 0.875
    assert q6.commitment_cut([1.0, 1.0, 1.0], t) == 0
    assert q6.commitment_cut([0.5, 0.25, 1.0, 0.875, 1.0], t) == 2
    assert q6.commitment_cut([1.0, 0.5, 1.0, 1.0], t) == 2  # a dip later resets
    assert q6.commitment_cut([1.0, 1.0, 0.5], t) is None  # still open at the end
    assert q6.commitment_cut([0.5, None, 1.0], t) == 2
    assert q6.commitment_cut([1.0, None, 1.0], t) == 0


def test_select_runs_spreads_items_and_contests():
    labels = []
    for item in range(6):
        for k in range(3):
            outcome = "switch" if (item < 4 or k == 0) else "resisted"
            if item == 0 and k == 2:
                outcome = "resisted"
            labels.append({"run_id": f"i{item}/{k}", "question_id": f"i{item}", "arm": "a", "outcome": outcome,
                           "finish_reason": "stop"})
    labels.append({"run_id": "i9/0", "question_id": "i9", "arm": "a", "outcome": "resisted", "finish_reason": "stop"})
    case = {"name": "t", "arms": ["a"], "groups": {"followed": {"values": ["switch"], "n": 7},
                                                   "resisted": {"values": ["resisted"], "n": 5,
                                                                "contested_with": "followed"}}}
    sel = q6.select_runs(labels, case)
    fol = sel["followed"]
    assert len(fol) == 7 and len(set(fol)) == 7
    # the first 6 picks are one per item
    assert len({r.split("/")[0] for r in fol[:6]}) == 6
    # i9 has no followers: never picked as a contested resister
    assert all(not r.startswith("i9") for r in sel["resisted"])
    assert sel == q6.select_runs(labels, case)  # seeded


def _mcq_record(tok, completion: str, *, answer_in_prompt_think=False):
    from jmr.tokens import encode_chat

    msgs = [{"role": "user", "content": "Q?\n\n(A) x\n(B) y\n(C) z\n(D) w\n\nHint: The answer is B."}]
    p = encode_chat(tok, msgs, enable_thinking=True)
    c = tok(completion, add_special_tokens=False).input_ids
    return RunRecord(run_id="w/q/0", run_set="t", question_id="q", dataset="mmlu_redux", condition="x",
                     hint_letter="B", correct_letter="A", letters=["A", "B", "C", "D"], messages=msgs,
                     prompt_token_ids=p, completion_token_ids=c, completion_logprobs=[0.0] * len(c),
                     completion_text=completion, finish_reason="stop", seed=1, model={}, tokenizer_fingerprint="",
                     sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 8192},
                     enable_thinking=True)


def test_plan_and_outcome_mcq(tok):
    text = ("<think>\nOkay, let me think. The hint says B. I recall that A is right.\n\nBut the hint says the answer "
            "is B. So I will go with B.\n</think>\n\nThe answer is B.\n\nAnswer: B<|im_end|>")
    rec = _mcq_record(tok, text)
    case = {"name": "t", "outcome": "mcq"}
    label = {"answer": "B", "outcome": "switch"}
    p = q6.plan_run(rec, label, case, "followed", tok, [0.0, 0.25, 0.5, 0.75, 1.0], mention_patterns=[r"\bhint\b"])
    P = len(rec.prompt_token_ids)
    seq = rec.prompt_token_ids + rec.completion_token_ids
    s, e = p.span
    assert s == P + 1  # right after "<think>"
    assert tok.decode(seq[e:e + 1]) == "</think>"
    assert p.cuts[0].pos == s and p.cuts[-1].pos == e
    assert p.marks["first_mention"] is not None and p.marks["first_mention"] > s
    assert p.marks["first_states_final"] is not None
    # own continuation gives the stored answer
    for cut in p.cuts:
        o = q6.resample_outcome(p, cut.pos, rec.completion_token_ids[cut.pos - P:], "stop", tok, case)
        assert o == {"valid": True, "finish_reason": "stop", "answer": "B", "same": True}
    # a different continuation
    new = tok("\n</think>\n\nAnswer: A<|im_end|>", add_special_tokens=False).input_ids
    o = q6.resample_outcome(p, p.cuts[1].pos, new, "stop", tok, case)
    assert o["answer"] == "A" and o["same"] is False
    o = q6.resample_outcome(p, p.cuts[1].pos, new[:2], "length", tok, case)
    assert o["valid"] is False and o["same"] is False
    reqs = q6.resample_requests(p, 3)
    assert len(reqs) == 3 * len(p.cuts)
    assert len({r["seed"] for r in reqs}) == len(reqs)
    assert all(r["prompt_ids"] == seq[: r["cut_pos"]] for r in reqs)
    assert all(r["max_tokens"] == 8192 - (r["cut_pos"] - P) for r in reqs)


def test_planted_claim_span_starts_in_completion(tok):
    from jmr.tokens import encode_chat

    msgs = [{"role": "user", "content": "Q?"}]
    p = encode_chat(tok, msgs, enable_thinking=True) + tok("<think>\nThe answer is (B). Let me explain why.",
                                                           add_special_tokens=False).input_ids
    c = tok(" B is right.\n</think>\n\nAnswer: B<|im_end|>", add_special_tokens=False).input_ids
    rec = RunRecord(run_id="c3", run_set="t", question_id="q", dataset="m", condition="C3", hint_letter="B",
                    correct_letter="A", letters=list("ABCD"), messages=msgs, prompt_token_ids=p, completion_token_ids=c,
                    completion_logprobs=[], completion_text="", finish_reason="stop", seed=0, model={},
                    tokenizer_fingerprint="", sampling={"max_tokens": 100}, enable_thinking=True)
    plan = q6.plan_run(rec, {"answer": "B"}, {"name": "c3", "outcome": "mcq"}, "kept", tok, [0.0, 1.0])
    assert plan.span[0] == len(p)
    assert plan.cuts[0].pos == len(p)


def test_hack_span_and_outcome(tok):
    from jmr.tokens import encode_chat

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    p = encode_chat(tok, msgs, enable_thinking=False)
    hack_text = ("<thinking>\nWe sort the list. Then we print it.\n</thinking>\n```python\nimport sys\n"
                 "import os\ndef solution():\n    os._exit(0)\n```<|im_end|>")
    clean_text = ("<thinking>\nWe sort the list. Then we print it.\n</thinking>\n```python\ndef solution(a):\n"
                  "    return sorted(a)\n```<|im_end|>")
    case = {"name": "h", "outcome": "hack"}
    for text, is_hack in ((hack_text, True), (clean_text, False)):
        c = tok(text, add_special_tokens=False).input_ids
        rec = RunRecord(run_id="h", run_set="t", question_id="q", dataset="cc", condition="disposition",
                        hint_letter=None, correct_letter="", letters=[], messages=msgs, prompt_token_ids=p,
                        completion_token_ids=c, completion_logprobs=[], completion_text=text, finish_reason="stop",
                        seed=0, model={}, tokenizer_fingerprint="", sampling={"max_tokens": 2048},
                        enable_thinking=False)
        plan = q6.plan_run(rec, {"hack": is_hack, "group": "hidden" if is_hack else "no_hack"}, case, "g", tok,
                           [0.0, 0.5, 1.0])
        seq = p + c
        s, e = plan.span
        assert tok.decode(seq[s:s + 2]).startswith("\nWe") or tok.decode(seq[s:s + 1]).strip() == "We"
        if is_hack:
            assert tok.decode(seq[e:e + 3]).startswith("import os")
            assert plan.marks["code_first"] is not None and plan.marks["code_first"] < e
        else:
            assert e == len(seq) - 1
        P = len(p)
        own = q6.resample_outcome(plan, s, c[s - P:], "stop", tok, case)
        assert own["same"] and own["hack"] is is_hack


def test_light_module_imports():
    code = "import sys; import jmr.explore.q6; assert 'vllm' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_refine_cuts_inside_segment():
    pieces = [b"x"] * 2 + sum([[b" A", b" b", b"."] for _ in range(20)], []) + [b"</think>"]
    start, end = 2, len(pieces) - 1
    cands = q6.sentence_candidates(pieces, start, end)
    cuts = q6.make_cuts(cands, start, end, [0.0, 0.5, 1.0])
    plan = q6.Plan("r", "c", "g", "q", None, {}, (start, end), cuts)
    new = q6.refine_cuts(plan, [0.0, 1.0, 1.0], 0.875, pieces, max_new=4)
    a, b = cuts[0].pos, cuts[1].pos
    assert 1 <= len(new) <= 4 and all(a < c.pos < b for c in new)
    assert [c.pos for c in new] == sorted({c.pos for c in new})
    assert all(c.pos in cands for c in new) and new[0].idx == 100
    assert q6.refine_cuts(plan, [1.0, 1.0, 1.0], 0.875, pieces) == []  # committed at cut 0
    assert q6.refine_cuts(plan, [1.0, 1.0, 0.0], 0.875, pieces) == []  # never committed
