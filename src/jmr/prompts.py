"""Prompts for the M1 conditions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from jmr import config

# baseline: the 10 unhinted samples used to decide which items are reliably known.
# unhinted: fresh unhinted samples on the kept items; condition 1 of the design.
UNNUDGED = ("baseline", "unhinted")
NUDGES = ("null_control", "zero_evidence", "known_evidence", "positive_control")
CONDITIONS = UNNUDGED + NUDGES


@dataclass(frozen=True)
class Question:
    id: str
    dataset: str
    question: str
    options: tuple[str, ...]
    letters: tuple[str, ...]
    correct: str
    subject: str = ""


def hinted_letter(question: Question) -> str:
    """The item's hinted option B: a wrong option, fixed per item by a hash of its
    ID, so every condition points at the same one."""
    wrong = [letter for letter in question.letters if letter != question.correct]
    seed = int(hashlib.sha256(question.id.encode()).hexdigest()[:16], 16)
    return random.Random(seed).choice(wrong)


def build_messages(question: Question, condition: str) -> tuple[list[dict], str | None]:
    """Chat messages for a condition, and the item's hinted letter (None for the
    unnudged conditions). In null_control the hinted letter is not in the prompt."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    cfg = config.load("conditions")
    options = "\n".join(
        cfg["option_template"].format(letter=letter, text=text)
        for letter, text in zip(question.letters, question.options)
    )
    letter = None
    if condition in NUDGES:
        letter = hinted_letter(question)
        options = f"{options}\n\n{cfg['nudges'][condition]['text'].format(letter=letter)}"
    content = cfg["question_template"].format(question=question.question, options=options)
    return [{"role": "user", "content": content}], letter


def mention_patterns(condition: str) -> list[str]:
    return config.load("conditions")["nudges"][condition]["mentions"] if condition in NUDGES else []
