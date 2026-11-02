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


def find_anchors(
    tokenizer: Any,
    ids: list[int],
    prompt_len: int,
    *,
    think_start_id: int,
    think_end_id: int,
    final_letter: str | None,
) -> Anchors:
    pieces = token_bytes(tokenizer, ids)
    starts = []  # byte offset at which each token starts
    offset = 0
    for piece in pieces:
        starts.append(offset)
        offset += len(piece)
    text = b"".join(pieces)

    def token_at(byte_offset: int) -> int:
        return bisect.bisect_right(starts, byte_offset) - 1

    think_start = _last_index(ids, think_start_id)
    think_end = _last_index(ids, think_end_id)

    final_answer = None
    if think_end is not None and final_letter:
        # Same rule as jmr.label.extract_answer: the last match of the first
        # pattern that matches the final letter.
        tail_start = starts[think_end] + len(pieces[think_end])
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
