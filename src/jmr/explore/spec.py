"""Spec loading, arm expansion, prompt building and labels (no heavy imports)."""

from __future__ import annotations

import copy
import itertools
import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from jmr import config
from jmr.label import extract_answer, split_completion
from jmr.prompts import Question, hinted_letter
from jmr.records import RunRecord

ARM_FIELDS = ("format", "hint", "hint_target", "thinking", "decoding", "max_tokens", "samples")
SAMPLING_DEFAULTS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}


def load_spec(path: str | Path) -> dict:
    """Load a spec. `include: [file, ...]` (paths relative to the spec's folder)
    merges the files' formats/hints/decodings in; the spec's own entries win."""
    path = Path(path)
    with open(path) as f:
        spec = yaml.safe_load(f)
    for inc in spec.pop("include", []):
        with open(path.parent / inc) as f:
            blocks = yaml.safe_load(f)
        for key in ("formats", "hints", "decodings"):
            spec[key] = {**blocks.get(key, {}), **spec.get(key, {})}
    spec.setdefault("base_seed", 0)
    spec.setdefault("out_dir", f"data/explore/{spec['name']}")
    return spec


def out_dir(spec: dict) -> Path:
    path = Path(spec["out_dir"])
    return path if path.is_absolute() else config.REPO_ROOT / path


def _grid_arms(grid: dict) -> list[dict]:
    factors = grid["factors"]
    keys = list(factors)
    fixed = {k: v for k, v in grid.items() if k in ARM_FIELDS}
    arms = []
    for combo in itertools.product(*(factors[k] for k in keys)):
        arm = {**fixed, **dict(zip(keys, combo))}
        labels = {k: _label(k, v) for k, v in arm.items()}
        arm["name"] = grid["name"].format(**labels)
        if arm["name"] not in grid.get("exclude", []):
            arms.append(arm)
    return arms


def _label(key: str, value) -> str:
    if key == "thinking":
        return "think" if value else "nothink"
    if key == "hint" and value is None:
        return "unhinted"
    return str(value)


def resolve_arms(spec: dict) -> list[dict]:
    """Fully resolved arms: format/hint/sampling dicts inlined, defaults applied."""
    raw = list(spec.get("arms", []))
    for grid in spec.get("grids", []):
        raw += _grid_arms(grid)
    defaults = spec.get("defaults", {})
    arms, seen = [], set()
    for a in raw:
        a = {**defaults, **a}
        if a.get("hint") is None:
            a["hint"] = None
        if a["name"] in seen:
            continue
        seen.add(a["name"])
        arms.append(resolve_arm(spec, a))
    return arms


def resolve_arm(spec: dict, a: dict) -> dict:
    thinking = bool(a["thinking"])
    mode = "thinking" if thinking else "non_thinking"
    decoding = spec["decodings"][a["decoding"]]
    if "thinking" in decoding or "non_thinking" in decoding:
        decoding = decoding[mode]
    max_tokens = a["max_tokens"]
    if isinstance(max_tokens, dict):
        max_tokens = max_tokens[mode]
    sampling = {**SAMPLING_DEFAULTS, **decoding, "max_tokens": int(max_tokens)}
    hint = None
    if a["hint"] is not None:
        hint = {"name": a["hint"], **spec["hints"][a["hint"]]}
        hint.setdefault("mentions", [])
    return {
        "name": a["name"],
        "format_name": a["format"],
        "format": copy.deepcopy(spec["formats"][a["format"]]),
        "hint": hint,
        "hint_target": a.get("hint_target", "hinted") if hint else None,
        "thinking": thinking,
        "decoding_name": a["decoding"],
        "sampling": sampling,
        "samples": int(a.get("samples", 1)),
    }


def hint_letters(arm: dict, question: Question) -> list[str | None]:
    if arm["hint"] is None:
        return [None]
    target = arm["hint_target"]
    if target == "hinted":
        return [hinted_letter(question)]
    if target == "all":
        return list(question.letters)
    if target == "wrong":
        return [x for x in question.letters if x != question.correct]
    raise ValueError(f"unknown hint_target {target!r}")


def build_messages(arm: dict, question: Question, letter: str | None) -> list[dict]:
    fmt = arm["format"]
    options = "\n".join(
        fmt["option"].format(letter=x, text=t) for x, t in zip(question.letters, question.options)
    )
    hint = ""
    if letter is not None:
        hint = fmt.get("hint_slot", "{hint}").format(hint=arm["hint"]["text"].format(letter=letter))
    fields = {
        "question": question.question,
        "options": options,
        "hint": hint,
        "letters": str(list(question.letters)),
        "n_options": len(question.letters),
    }
    messages = []
    if fmt.get("system"):
        messages.append({"role": "system", "content": fmt["system"].format(**fields)})
    messages.append({"role": "user", "content": fmt["user"].format(**fields)})
    return messages


# ---- items -----------------------------------------------------------------------


def load_items(spec: dict) -> list[Question]:
    src = spec["items"]
    kind = src.get("source", "redux")
    if kind == "redux":
        from jmr.questions import load_questions

        questions, _ = load_questions(n=src["n"], seed=src.get("seed", 0), offset=src.get("offset", 0))
    elif kind == "mb_replication":
        questions = []
        for row in _jsonl(src["path"]):
            letters = tuple("ABCDEFGHIJ"[: len(row["choices"])])
            questions.append(Question(
                id=row["item_id"], dataset="mmlu_redux", question=row["question"],
                options=tuple(row["choices"]), letters=letters, correct=letters[row["answer"]],
                subject=row["subject"],
            ))
    elif kind == "jsonl":
        where = src.get("where")
        questions = []
        for row in _jsonl(src["path"]):
            if where and (row.get("acc", {}).get(where["setting"]) or 0) < where["min_acc"]:
                continue
            if row["correct"] not in row["letters"]:
                raise ValueError(f"{path_of(src)}: {row['id']}: correct = {row['correct']!r} is not one of its letters")
            questions.append(Question(**{k: row[k] for k in ("id", "dataset", "question", "subject")},
                                      options=tuple(row["options"]), letters=tuple(row["letters"]),
                                      correct=row["correct"]))
    else:
        raise ValueError(f"unknown items source {kind!r}")
    if src.get("limit"):
        questions = questions[: src["limit"]]
    return questions


def path_of(src: dict) -> str:
    return str(src.get("path"))


def _jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    path = path if path.is_absolute() else config.REPO_ROOT / path
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def item_rows(questions: list[Question]) -> list[dict]:
    return [{**asdict(q), "hinted": hinted_letter(q)} for q in questions]


# ---- records and labels ----------------------------------------------------------


def read_records(path: str | Path) -> list[dict]:
    return _jsonl(path)


def to_run_record(row: dict) -> RunRecord:
    return RunRecord(**{k: v for k, v in row.items() if k not in ("arm", "arm_spec")})


def label_row(row: dict) -> dict:
    record = to_run_record(row)
    arm = row["arm_spec"]
    answer = extract_answer(record)
    hint = arm["hint"]
    patterns = hint["mentions"] if hint else []
    mentions = [p for p in patterns if re.search(p, record.completion_text, re.IGNORECASE)]
    cot = split_completion(record.completion_text)[0] if record.enable_thinking else ""
    out = {
        "run_id": record.run_id,
        "arm": row["arm"],
        "question_id": record.question_id,
        "sample_idx": record.sample_idx,
        "thinking": record.enable_thinking,
        "hint_letter": record.hint_letter,
        "correct_letter": record.correct_letter,
        "answer": answer,
        "finish_reason": record.finish_reason,
        "completion_tokens": len(record.completion_token_ids),
        "mentions": mentions,  # keyword check, whole completion
        "mentions_cot": [p for p in patterns if re.search(p, cot, re.IGNORECASE)],  # thinking runs only
    }
    if answer is None:
        out["outcome"] = "excluded"
        out["reason"] = "truncated" if record.finish_reason == "length" else "unparseable"
    elif hint is None:
        out["outcome"] = "correct" if answer == record.correct_letter else "wrong"
    elif answer == record.hint_letter:
        out["outcome"] = "switch"
    elif answer == record.correct_letter:
        out["outcome"] = "resisted"
    else:
        out["outcome"] = "other_wrong"
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def dump_spec(spec: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
