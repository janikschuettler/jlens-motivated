"""Normative multiple-choice items (overnight wave 2, E6). Light imports at module
level; `datasets` / `huggingface_hub` are imported inside the loaders (gen env).

MoralChoice (Scherrer et al. 2023, `ninoscherrer/moralchoice`, CC-BY-4.0), raw
CSV `scenarios/moralchoice_low_ambiguity.csv` (687 rows). Documented columns
(dataset card, checked 2026-11-02): scenario_id, ambiguity, generation_type,
context, action1, action2 (the card writes "action 1"/"action 2"; both
spellings are accepted), a1_<rule>, a2_<rule>. In low-ambiguity scenarios
action1 is the preferred action by construction, so `correct` is the letter
action1 lands on. A/B order is drawn per item from a hash of (order_seed,
scenario_id), so about half the items have the preferred action as (B).

MMLU-Redux moral subjects: `jmr.questions.load_questions` has no subject filter,
so `load_redux_subjects` draws from the error-free items of the given subjects
only, with the same IDs and Question fields as the main loader.
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from jmr.prompts import Question

MORALCHOICE_REPO = "ninoscherrer/moralchoice"
MORALCHOICE_FILES = {"low": "scenarios/moralchoice_low_ambiguity.csv",
                     "high": "scenarios/moralchoice_high_ambiguity.csv"}
REQUIRED = ("scenario_id", "context", "action1", "action2")


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def read_moralchoice_csv(path: str | Path) -> list[dict]:
    """Rows with normalised column names (lowercase, no spaces). Raises on a
    missing required column."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{_norm(k): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
    if rows:
        missing = [c for c in REQUIRED if c not in rows[0]]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; have {sorted(rows[0])}")
    return rows


def moralchoice_question(row: dict, *, order_seed: int = 0, ambiguity: str = "low") -> Question:
    """(A)/(B) = action1/action2 in a seeded order; correct = action1's letter
    (meaningful for low ambiguity only)."""
    sid = row["scenario_id"]
    flip = int(hashlib.sha256(f"{order_seed}:{sid}".encode()).hexdigest()[:8], 16) % 2 == 1
    a1, a2 = row["action1"], row["action2"]
    options = (a2, a1) if flip else (a1, a2)
    return Question(id=f"moralchoice/{sid}", dataset="moralchoice", question=row["context"], options=options,
                    letters=("A", "B"), correct="B" if flip else "A", subject=f"moralchoice_{ambiguity}")


def moralchoice_path(ambiguity: str = "low") -> Path:
    """The CSV in the HF cache (downloads it there on first use; gen env)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MORALCHOICE_REPO, MORALCHOICE_FILES[ambiguity], repo_type="dataset"))


def load_moralchoice(n: int | None = None, *, seed: int = 0, order_seed: int = 0, ambiguity: str = "low",
                     path: str | Path | None = None) -> list[Question]:
    """n scenarios at random (sorted by scenario_id, shuffled with seed), as
    2-option Questions. path: a local CSV (tests); default: the HF cache."""
    rows = read_moralchoice_csv(path or moralchoice_path(ambiguity))
    if ambiguity == "low":
        bad = [r["scenario_id"] for r in rows if r.get("ambiguity", "low").lower() not in ("low", "")]
        if bad:
            raise ValueError(f"rows with ambiguity != low in the low-ambiguity file: {bad[:5]}")
    rows = sorted(rows, key=lambda r: r["scenario_id"])
    random.Random(seed).shuffle(rows)
    rows = rows if n is None else rows[:n]
    return [moralchoice_question(r, order_seed=order_seed, ambiguity=ambiguity) for r in rows]


def draw_subjects(pool: list[tuple[str, int, dict]], subjects: list[str], n_per_subject: int, seed: int,
                  keep_error_type: str = "ok") -> list[Question]:
    """n_per_subject error-free items per subject, each drawn like jmr.questions.draw."""
    from jmr.questions import draw, filter_redux

    out = []
    for s in subjects:
        sub = filter_redux([x for x in pool if x[0] == s], keep_error_type)
        if len(sub) < n_per_subject:
            raise ValueError(f"{s}: only {len(sub)} error-free items, {n_per_subject} requested")
        out += draw(sub, n_per_subject, seed)
    return out


def load_redux_subjects(subjects: list[str], n_per_subject: int, seed: int = 0) -> list[Question]:
    """MMLU-Redux 2.0 items of the given subjects (error_type ok), gen env."""
    from datasets import load_dataset

    from jmr import config

    spec = config.load("data")["datasets"]["mmlu_redux"]
    pool = []
    for s in subjects:
        for i, row in enumerate(load_dataset(spec["hf_id"], s, split=spec["split"])):
            pool.append((s, i, row))
    return draw_subjects(pool, subjects, n_per_subject, seed, spec["keep_error_type"])
