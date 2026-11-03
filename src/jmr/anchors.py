"""Anchor positions in a run's token sequence, located from token IDs and token
bytes only (no re-tokenization). Positions index the full sequence
prompt + completion."""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass
from typing import Any

from jmr.label import ANSWER_PATTERN_SOURCES
from jmr.tokens import token_bytes

FINAL_ANSWER_RES = [re.compile(src.encode(), flags) for src, flags in ANSWER_PATTERN_SOURCES]
# First place the chain of thought states the (final) answer letter. Heuristic;
# what counts as "first stated" is open question 7.
STATED_TEMPLATE = (
    rb"(?i:answer(?: is| would be| should be|:)|go with|choose|pick)\s*"
    rb"(?:option\s*)?\(?(%s)\)?(?![A-Za-z])"
)


@dataclass
class Anchors:
    prompt_len: int
    think_start: int | None  # the <think> token
    think_end: int | None  # the </think> token
    answer_first_stated: int | None  # letter token of the first statement in the CoT
    final_answer: int | None  # letter token of "Answer: X" after </think>

    def as_dict(self) -> dict:
        return asdict(self)


def _last_index(ids: list[int], value: int) -> int | None:
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == value:
            return i
    return None


HARMONY_HEADER = re.compile(rb"<\|channel\|>(\w+).*?<\|message\|>", re.DOTALL)


def _harmony_spans(ids, prompt_len, text, starts, pieces, msg_id, end_id, token_at):
    """(think_start, think_end, byte offset where the final message starts) for a
    harmony completion; None where absent."""
    comp_start = starts[prompt_len] if prompt_len < len(ids) else len(text)
    think_start = think_end = tail_from = None
    for m in HARMONY_HEADER.finditer(text, comp_start):
        msg_tok = token_at(m.end() - 1)  # the <|message|> token
        if ids[msg_tok] != msg_id:
            continue
        channel = m.group(1)
        if channel == b"analysis" and tail_from is None:
            if think_start is None:
                think_start = msg_tok
            ends = [i for i in range(msg_tok + 1, len(ids)) if ids[i] == end_id]
            if ends:
                think_end = ends[0]
        elif channel == b"final":
            tail_from = starts[msg_tok] + len(pieces[msg_tok])
    return think_start, think_end, tail_from


def find_anchors(
    tokenizer: Any,
    ids: list[int],
    prompt_len: int,
    *,
    think_start_id: int,
    think_end_id: int,
    final_letter: str | None,
    harmony: bool = False,
) -> Anchors:
    """harmony=True (gpt-oss): think_start_id / think_end_id are <|message|> and
    <|end|>; think_start = the <|message|> that opens the first analysis-channel
    message in the completion, think_end = the <|end|> that closes the last
    analysis message before the final channel, and the final answer is searched
    in the final-channel message only. Default: Qwen3 <think> / </think>."""
    pieces = token_bytes(tokenizer, ids)
    starts = []  # byte offset at which each token starts
    offset = 0
    for piece in pieces:
        starts.append(offset)
        offset += len(piece)
    text = b"".join(pieces)

    def token_at(byte_offset: int) -> int:
        return bisect.bisect_right(starts, byte_offset) - 1

    if harmony:
        think_start, think_end, tail_from = _harmony_spans(ids, prompt_len, text, starts, pieces,
                                                         think_start_id, think_end_id, token_at)
    else:
        think_start = _last_index(ids, think_start_id)
        think_end = _last_index(ids, think_end_id)
        tail_from = None if think_end is None else starts[think_end] + len(pieces[think_end])

    final_answer = None
    if tail_from is not None and final_letter:
        # Same rule as jmr.label.extract_answer: the last match of the first
        # pattern that matches the final letter.
        tail_start = tail_from
        for pattern in FINAL_ANSWER_RES:
            matches = [m for m in pattern.finditer(text, tail_start) if m.group(1) == final_letter.encode()]
            if matches:
                final_answer = token_at(matches[-1].start(1))
                break

    first_stated = None
    if final_letter and think_start is not None and think_end is not None:
        pattern = re.compile(STATED_TEMPLATE % re.escape(final_letter.encode()))
        cot_start = starts[think_start] + len(pieces[think_start])
        match = pattern.search(text, cot_start, starts[think_end])
        if match:
            first_stated = token_at(match.start(1))

    return Anchors(
        prompt_len=prompt_len,
        think_start=think_start,
        think_end=think_end,
        answer_first_stated=first_stated,
        final_answer=final_answer,
    )
