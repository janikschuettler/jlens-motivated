"""Multiple-choice question loaders (need `datasets`, installed in the gen env).

MMLU-Redux 2.0 is the primary source. The other datasets are kept for later and
refused unless allow_later=True.
"""

from __future__ import annotations

import random
import string

from jmr import config
from jmr.prompts import Question

LETTERS = string.ascii_uppercase


def filter_redux(pool: list[tuple[str, int, dict]], keep_error_type: str) -> list[tuple[str, int, dict]]:
    """Keep annotated-error-free items with a valid answer index."""
    return [
        (subject, i, row)
        for subject, i, row in pool
        if row["error_type"] == keep_error_type and 0 <= row["answer"] < len(row["choices"]) <= len(LETTERS)
    ]


def redux_question(subject: str, i: int, row: dict) -> Question:
    options = tuple(row["choices"])
    return Question(
        id=f"redux/{subject}/{i}",
        dataset="mmlu_redux",
        question=row["question"],
        options=options,
        letters=tuple(LETTERS[: len(options)]),
        correct=LETTERS[row["answer"]],
        subject=subject,
    )


def draw(pool: list[tuple[str, int, dict]], n: int, seed: int, offset: int = 0) -> list[Question]:
    """n items at random from the pool: a fixed order (sorted, then shuffled with
    the seed), so the same seed gives the same items."""
    ordered = sorted(pool, key=lambda x: (x[0], x[1]))
    random.Random(seed).shuffle(ordered)
    return [redux_question(*x) for x in ordered[offset : offset + n]]


def _load_redux(spec: dict) -> list[tuple[str, int, dict]]:
    from datasets import get_dataset_config_names, load_dataset

    pool = []
    for subject in get_dataset_config_names(spec["hf_id"]):
        for i, row in enumerate(load_dataset(spec["hf_id"], subject, split=spec["split"])):
            pool.append((subject, i, row))
    return pool


def _load_arc(spec: dict, n: int, offset: int) -> list[Question]:
    from datasets import load_dataset

    rows = load_dataset(spec["hf_id"], spec["config"], split=spec["split"])
    questions = []
    for row in rows.select(range(offset, min(offset + n, len(rows)))):
        labels = list(row["choices"]["label"])  # some items use "1".."4" instead of A..D
        options = tuple(row["choices"]["text"])
        questions.append(
            Question(
                id=f"arc/{row['id']}", dataset="arc_challenge", question=row["question"], options=options,
                letters=tuple(LETTERS[: len(options)]), correct=LETTERS[labels.index(row["answerKey"])],
            )
        )
    return questions


def load_questions(
    dataset: str | None = None, *, n: int, seed: int = 0, offset: int = 0, allow_later: bool = False
) -> tuple[list[Question], int | None]:
    """(questions, size of the error-free pool). The pool size is None for datasets
    that aren't MMLU-Redux."""
    cfg = config.load("data")
    dataset = dataset or cfg["primary"]
    if dataset != cfg["primary"] and not allow_later:
        raise ValueError(f"{dataset!r} is not the primary dataset ({cfg['primary']}); pass allow_later=True to use it")
    spec = cfg["datasets"][dataset]
    if dataset == "mmlu_redux":
        pool = filter_redux(_load_redux(spec), spec["keep_error_type"])
        return draw(pool, n, seed, offset), len(pool)
    if dataset == "arc_challenge":
        return _load_arc(spec, n, offset), None
    raise ValueError(f"no loader for {dataset!r}")
