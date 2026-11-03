"""Wave-2 arms: an extension of the jmr.explore spec format (no heavy imports).

Specs are configs/explore/w2_*.yaml and are run by scripts/explore/run_w2.py
(not run_arms.py). Everything in jmr.explore's format still applies (include,
items, defaults, formats, hints, decodings, arms; `grids` are not supported
here). ---- EXTENSION POINT: extra arm fields, all optional ----

  letter: hinted | correct | uniform | none
      Which letter fills {letter} in the hint, turns and prefill, and is saved as
      the run's hint_letter. hinted = the item's fixed wrong option B
      (jmr.prompts.hinted_letter); correct = the correct option; uniform = drawn
      uniformly over all options per (item, sample), the same draw in every arm
      of the spec (seed: spec `letter_seed`, default 0); none = no letter
      (hint_letter None). Default: hinted if the arm has a hint, else none.
      The letter may be recorded without being shown (e.g. pushback P1/P3,
      null control); labels.jsonl says so (letter_shown).
  turns: [{role: assistant|user, text: "..."} | {role: assistant, from_run: true}]
      Turns after the first user message (built as usual from format + hint).
      {letter} and {correct} are filled in. `from_run: true` (first turn only)
      inserts an on-policy assistant turn: the visible answer of a saved run of
      the same item, spliced in as its generated token IDs (jmr.explore.turns).
  turn1: {runs: path/to/runs.jsonl, arm: <arm name in that file>, require: correct}
      Where from_run turns come from. Sample k of an item uses that item's k-th
      qualifying run (by sample_idx, cycling), the same one in every arm. The
      saved run's first user message must equal this arm's (checked).
  prefill: "<think>\\nThe answer is ({letter}). Let me explain why."
      Thinking-mode assistant prefill; its IDs end the prompt (w2.prefill_*).
  format_by_dataset: {moralchoice: moralchoice}
      Format per Question.dataset; `format` is the fallback.
  labels: [grader, argued, concealment]   extra label sets (jmr.explore.w2_labels)
  mentions: [regex, ...]                  added to the hint's mention regexes
  est_tokens: 1600                        mean completion tokens, for --dry-run estimates

Item sources added (items.source):
  known_pool   {path: data/explore/known_items.jsonl, setting: A, min_acc: 0.9,
                max_acc: 1.0, limit: 40}   rows with Question fields + acc{setting}
  moral        {moralchoice: {n, seed, order_seed, path?, skip?}, redux_subjects:
                {subjects: [...], n_per_subject, seed}}   (jmr.moral)
Anything else goes to jmr.explore.spec.load_items.

Run IDs: "<spec>/<arm>/<item>/<letter or ->/<k>"; seeds as in jmr.explore.
Rows in runs.jsonl: RunRecord fields + arm + arm_spec + w2 (how the prompt was
built: kind, letter_shown, prefill_text/prefill_token_ids, turn1_run_id,
template_equal, assistant_span). The prompt IDs include any prefill and spliced
turns; completion IDs are only what the model generated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from jmr import config
from jmr.explore.spec import build_messages, load_items, load_spec, resolve_arm
from jmr.explore.turns import PromptPlan, plain_prompt, prefill_prompt, splice_assistant_turn, visible_response_ids
from jmr.prompts import Question, hinted_letter

W2_FIELDS = ("letter", "turns", "turn1", "prefill", "format_by_dataset", "labels", "mentions", "est_tokens")
LETTER_MODES = ("hinted", "correct", "uniform", "none")


def load_w2_spec(path: str | Path) -> dict:
    spec = load_spec(path)
    if spec.get("grids"):
        raise ValueError("w2 specs don't support grids; list the arms")
    return spec


def resolve_w2_arms(spec: dict) -> list[dict]:
    defaults = spec.get("defaults", {})
    arms, seen = [], set()
    for raw in spec.get("arms", []):
        a = {**defaults, **raw}
        a.setdefault("hint", None)
        if a["name"] in seen:
            raise ValueError(f"duplicate arm name {a['name']!r}")
        seen.add(a["name"])
        arm = resolve_arm(spec, a)
        letter = a.get("letter") or ("hinted" if arm["hint"] else "none")
        if letter not in LETTER_MODES:
            raise ValueError(f"{a['name']}: unknown letter mode {letter!r}")
        turns = copy.deepcopy(a.get("turns") or [])
        for i, t in enumerate(turns):
            if t.get("from_run") and (i != 0 or t["role"] != "assistant"):
                raise ValueError(f"{a['name']}: from_run is only allowed for the first turn (assistant)")
        if any(t.get("from_run") for t in turns) and not a.get("turn1"):
            raise ValueError(f"{a['name']}: from_run turn needs `turn1`")
        if a.get("prefill") and turns:
            raise ValueError(f"{a['name']}: prefill and turns together are not supported")
        if a.get("prefill") and not arm["thinking"]:
            raise ValueError(f"{a['name']}: prefill is for thinking mode")
        arm.update({
            "letter": letter, "turns": turns, "turn1": a.get("turn1"), "prefill": a.get("prefill"),
            "formats_by_dataset": {ds: copy.deepcopy(spec["formats"][f])
                                   for ds, f in (a.get("format_by_dataset") or {}).items()},
            "labels": list(a.get("labels") or []),
            "mention_patterns": (arm["hint"]["mentions"] if arm["hint"] else []) + list(a.get("mentions") or []),
            "est_tokens": a.get("est_tokens"),
        })
        arm["hint_target"] = None  # superseded by `letter`
        arms.append(arm)
    return arms


# ---- items -----------------------------------------------------------------------


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else config.REPO_ROOT / p


def _jsonl(path: str | Path) -> list[dict]:
    with open(_path(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def question_from_row(row: dict) -> Question:
    if not isinstance(row.get("correct"), str) or row["correct"] not in row["letters"]:
        raise ValueError(f"{row.get('id')}: `correct` is {row.get('correct')!r}, not one of its letters "
                         "(a known-items file that overwrote it with per-setting counts?)")
    return Question(id=row["id"], dataset=row["dataset"], question=row["question"], options=tuple(row["options"]),
                    letters=tuple(row["letters"]), correct=row["correct"], subject=row.get("subject", ""))


def load_w2_items(spec: dict) -> list[Question]:
    src = spec["items"]
    kind = src.get("source", "redux")
    if kind == "known_pool":
        qs = []
        for row in _jsonl(src["path"]):
            acc = (row.get("acc") or {}).get(src.get("setting", "A"))
            if acc is None or not (src.get("min_acc", 0.9) <= acc <= src.get("max_acc", 1.0)):
                continue
            qs.append(question_from_row(row))
    elif kind == "moral":
        from jmr.moral import load_moralchoice, load_redux_subjects

        qs = []
        if src.get("redux_subjects"):
            r = src["redux_subjects"]
            qs += load_redux_subjects(r["subjects"], r["n_per_subject"], r.get("seed", 0))
        if src.get("moralchoice"):
            m = src["moralchoice"]
            qs += load_moralchoice(m.get("n"), seed=m.get("seed", 0), order_seed=m.get("order_seed", 0),
                                   path=m.get("path"))[m.get("skip", 0):]
    else:
        return load_items(spec)
    if src.get("limit"):
        qs = qs[: src["limit"]]
    return qs


# ---- letters and prompts ---------------------------------------------------------


def uniform_letter(question: Question, k: int, seed: int = 0) -> str:
    h = int(hashlib.sha256(f"{seed}:{question.id}:{k}".encode()).hexdigest()[:16], 16)
    return random.Random(h).choice(list(question.letters))


def arm_letter(arm: dict, question: Question, k: int, letter_seed: int = 0) -> str | None:
    mode = arm["letter"]
    if mode == "none":
        return None
    if mode == "hinted":
        return hinted_letter(question)
    if mode == "correct":
        return question.correct
    return uniform_letter(question, k, letter_seed)


def letter_shown(arm: dict) -> bool:
    texts = [arm["hint"]["text"]] if arm["hint"] else []
    texts += [t.get("text", "") for t in arm["turns"]] + [arm["prefill"] or ""]
    return any("{letter}" in t for t in texts)


def user_messages(arm: dict, question: Question, letter: str | None) -> list[dict]:
    fmt = arm["formats_by_dataset"].get(question.dataset)
    a = arm if fmt is None else {**arm, "format": fmt}
    # build_messages only inserts the hint when a letter is given
    return build_messages(a, question, letter if arm["hint"] else None)


def _fill(text: str, letter: str | None, question: Question) -> str:
    return text.format(letter=letter, correct=question.correct)


@dataclass
class Turn1:
    run_id: str
    messages: list[dict]
    prompt_ids: list[int]
    completion_ids: list[int]
    enable_thinking: bool


def load_turn1(cfg: dict, *, cache: dict | None = None) -> dict[str, list[Turn1]]:
    """item -> qualifying saved runs (require: correct | any), by sample_idx."""
    from jmr.label import extract_answer
    from jmr.records import RunRecord

    key = (str(cfg["runs"]), cfg.get("arm"), cfg.get("require", "correct"))
    if cache is not None and key in cache:
        return cache[key]
    names = set(RunRecord.__dataclass_fields__)
    out: dict[str, list[tuple[int, Turn1]]] = {}
    for row in _jsonl(cfg["runs"]):
        if cfg.get("arm") and row.get("arm") != cfg["arm"]:
            continue
        rec = RunRecord(**{k: v for k, v in row.items() if k in names})
        if cfg.get("require", "correct") == "correct" and extract_answer(rec) != rec.correct_letter:
            continue
        out.setdefault(rec.question_id, []).append(
            (rec.sample_idx, Turn1(rec.run_id, rec.messages, rec.prompt_token_ids, rec.completion_token_ids,
                                   rec.enable_thinking)))
    res = {q: [t for _, t in sorted(v, key=lambda x: (x[0], x[1].run_id))] for q, v in out.items()}
    if cache is not None:
        cache[key] = res
    return res


def build_prompt(tokenizer, arm: dict, question: Question, letter: str | None, k: int,
                 turn1: dict[str, list[Turn1]] | None = None) -> PromptPlan:
    msgs = user_messages(arm, question, letter)
    shown = letter_shown(arm) and letter is not None
    if arm["prefill"]:
        plan = prefill_prompt(tokenizer, msgs, _fill(arm["prefill"], letter, question))
    elif arm["turns"] and arm["turns"][0].get("from_run"):
        runs = (turn1 or {}).get(question.id)
        if not runs:
            raise ValueError(f"{arm['name']}: no qualifying turn-1 run for {question.id}")
        t1 = runs[k % len(runs)]
        if t1.messages != msgs:
            raise ValueError(f"{arm['name']}: turn-1 run {t1.run_id} has a different first user message")
        resp = visible_response_ids(tokenizer, t1.completion_ids, thinking=t1.enable_thinking)
        following = [{"role": t["role"], "content": _fill(t["text"], letter, question)} for t in arm["turns"][1:]]
        plan = splice_assistant_turn(tokenizer, msgs, resp, following, enable_thinking=arm["thinking"],
                                     prefix_ids=t1.prompt_ids)
        if not plan.info["head_matches_template"]:
            raise ValueError(f"{t1.run_id}: saved prompt IDs differ from the template's for the same messages")
        plan.info.update({"turn1_run_id": t1.run_id, "turn1_reuse": len(runs) <= k})
    elif arm["turns"]:
        msgs = msgs + [{"role": t["role"], "content": _fill(t["text"], letter, question)} for t in arm["turns"]]
        plan = plain_prompt(tokenizer, msgs, enable_thinking=arm["thinking"])
    else:
        plan = plain_prompt(tokenizer, msgs, enable_thinking=arm["thinking"])
    plan.info["letter_shown"] = shown
    return plan


@dataclass
class W2Request:
    run_id: str
    run_set: str
    question: Question
    arm: dict
    letter: str | None
    sample_idx: int
    seed: int
    plan: PromptPlan


def run_seed(base_seed: int, run_id: str) -> int:
    """= jmr.gen.generate.run_seed (copied: that module imports vllm)."""
    return int(hashlib.sha256(f"{base_seed}:{run_id}".encode()).hexdigest()[:8], 16)


def plan_requests(spec: dict, arms: list[dict], questions: list[Question], tokenizer) -> list[W2Request]:
    cache: dict = {}
    out = []
    for arm in arms:
        turn1 = load_turn1(arm["turn1"], cache=cache) if arm["turn1"] else None
        for q in questions:
            for k in range(arm["samples"]):
                letter = arm_letter(arm, q, k, spec.get("letter_seed", 0))
                run_id = f"{spec['name']}/{arm['name']}/{q.id}/{letter or '-'}/{k}"
                plan = build_prompt(tokenizer, arm, q, letter, k, turn1)
                out.append(W2Request(run_id, spec["name"], q, arm, letter, k,
                                     run_seed(spec["base_seed"], run_id), plan))
    return out


def estimate(requests: list[W2Request], tok_s: float, default_tokens: dict) -> dict:
    """Runs and GPU minutes per arm at tok_s aggregate generated tokens/s."""
    per: dict[str, dict] = {}
    for r in requests:
        a = r.arm
        est = a.get("est_tokens") or default_tokens["thinking" if a["thinking"] else "non_thinking"]
        d = per.setdefault(a["name"], {"runs": 0, "tokens": 0, "prompt_tokens": 0})
        d["runs"] += 1
        d["tokens"] += est
        d["prompt_tokens"] += len(r.plan.prompt_ids)
    for d in per.values():
        d["gpu_min"] = round(d["tokens"] / tok_s / 60, 1)
    return per
