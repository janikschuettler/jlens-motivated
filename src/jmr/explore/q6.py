"""Point of no return (Q6): cut a saved run's reasoning at many points and
resample the rest, to see where the outcome stops depending on the resample.

No heavy imports (the gen script imports vllm; the analysis only numpy).

Per selected run:
- the span is the part of the generated text the cuts fall in. Thinking-mode
  MCQ runs: from the first generated reasoning token (after "<think>", or the
  start of the completion if "<think>" and a planted claim sit in the prompt) to
  the model's own "</think>". Hack runs (text <thinking> block, non-thinking
  template): from the first token inside <thinking> to the start of the line
  holding the first hack-related word (jmr.hack.first_mention), for hack runs,
  or to the end of the code, for no-hack runs; so cuts cover the reasoning and
  the code written before the hack becomes visible.
- cut candidates are sentence boundaries found on the saved token IDs' bytes
  (never by re-tokenizing text): right after a token that ends in a newline, or
  after a token ending in ".", "?" or "!" (optionally followed by quotes or
  brackets) when the next token starts with a space (before a newline, the cut
  goes after the newline token instead). The span start
  and end are always candidates.
- the cut for fraction f is the candidate nearest to start + f (end - start),
  ties to the earlier one; duplicates are dropped. f = 1 is the span end.
- each cut is resampled R times: prompt = saved prompt IDs + completion IDs
  before the cut, the run's own sampling settings, max_tokens reduced by the
  prefix length (so the total completion budget is the original one), a fresh
  seed per resample (resample_seed).
- the outcome of a resample is read from the prefix + continuation, with the
  same answer extractor as the original runs (jmr.label.extract_answer), or the
  same hack labeller (jmr.hack.label_run). "same" = the original outcome.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from jmr import config
from jmr.records import RunRecord

THINK_START, THINK_END = "<think>", "</think>"
SENT_END = re.compile(rb"[.?!][\"')\]*]*$")


# ---- loading and selection ---------------------------------------------------------


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else config.REPO_ROOT / p


def read_jsonl(path: str | Path, keep=None) -> list[dict]:
    out = []
    with open(_path(path)) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if keep is None or keep(row):
                    out.append(row)
    return out


def record_from_row(row: dict) -> RunRecord:
    names = set(RunRecord.__dataclass_fields__)
    return RunRecord(**{k: v for k, v in row.items() if k in names})


def group_of(label: dict, case: dict) -> str | None:
    """The run's group name in this case (from its labels.jsonl row), or None."""
    field_ = case.get("label_field", "outcome")
    for name, g in case["groups"].items():
        if label.get(field_) in g["values"]:
            return name
    return None


def select_runs(labels: list[dict], case: dict, seed: int = 0) -> dict[str, list[str]]:
    """Run IDs per group: seeded, spread over items (one run per item first, then
    a second run per item, ... until n). A group with contested_with: X keeps only
    items that also have at least one run of group X in the same arm(s); this
    keeps the comparison group on items where the outcome is in play."""
    item_key = case.get("item_field", "question_id")
    by_group: dict[str, dict[str, list[str]]] = {}
    for lab in labels:
        if case.get("arms") and lab.get("arm") not in case["arms"]:
            continue
        if lab.get("finish_reason", "stop") != "stop":
            continue
        g = group_of(lab, case)
        if g is not None:
            by_group.setdefault(g, {}).setdefault(lab[item_key], []).append(lab["run_id"])
    out = {}
    for name, g in case["groups"].items():
        items = by_group.get(name, {})
        if g.get("contested_with"):
            other = set(by_group.get(g["contested_with"], {}))
            items = {k: v for k, v in items.items() if k in other}
        rng = random.Random(f"{seed}:{case['name']}:{name}")
        keys = sorted(items)
        rng.shuffle(keys)
        pools = {k: sorted(items[k]) for k in keys}
        for k in keys:
            rng.shuffle(pools[k])
        picked, depth = [], 0
        while len(picked) < g["n"] and any(len(pools[k]) > depth for k in keys):
            for k in keys:
                if len(picked) >= g["n"]:
                    break
                if len(pools[k]) > depth:
                    picked.append(pools[k][depth])
            depth += 1
        out[name] = picked
    return out


# ---- spans and cuts ----------------------------------------------------------------


def offsets_of(pieces: list[bytes]) -> list[int]:
    offs = [0]
    for p in pieces:
        offs.append(offs[-1] + len(p))
    return offs


def mcq_span(ids: list[int], prompt_len: int, think_start_id: int, think_end_id: int) -> tuple[int, int] | None:
    """[start, end) in the full sequence: generated reasoning tokens, up to the
    model's own </think>. None if there is no </think> after the start."""
    starts = [i for i, x in enumerate(ids) if x == think_start_id]
    if not starts:
        return None
    start = max(starts[-1] + 1, prompt_len)
    ends = [i for i in range(start, len(ids)) if ids[i] == think_end_id]
    return (start, ends[0]) if ends else None


def hack_span(pieces: list[bytes], prompt_len: int, is_hack: bool) -> tuple[int, int] | None:
    """[start, end) in the full sequence for a hack-task run (pieces = token bytes
    of prompt + completion): from the first token inside the <thinking> block to
    the start of the line holding the first hack-related word after the block's
    start (hack runs), or to the last token (no-hack runs)."""
    from jmr.hack import first_mention

    comp = pieces[prompt_len:]
    offs = offsets_of(comp)
    data = b"".join(comp)
    s = data.find(b"<thinking>")
    if s < 0:
        return None
    cs = s + len(b"<thinking>")
    t0 = next(i for i in range(len(comp)) if offs[i] >= cs)
    if is_hack:
        text = data.decode("utf-8", errors="replace")
        # character offsets -> byte offsets: work on the decoded prefix
        m = first_mention(text, len(data[:cs].decode("utf-8", errors="replace")))
        if m is None:
            return None
        mb = len(text[:m].encode("utf-8"))
        line_start = data.rfind(b"\n", 0, mb) + 1
        end = next(i for i in range(len(comp)) if offs[i] >= line_start)
    else:
        end = len(comp) - 1  # the last token (usually <|im_end|>) is never a cut
    return (prompt_len + t0, prompt_len + end) if end > t0 else None


def sentence_candidates(pieces: list[bytes], start: int, end: int) -> list[int]:
    """Cut positions c in [start, end] (prefix = seq[:c]) at sentence or line
    boundaries; start and end always included."""
    out = {start, end}
    for c in range(start + 1, end):
        prev, nxt = pieces[c - 1], pieces[c]
        if prev.endswith(b"\n"):
            out.add(c)
        elif SENT_END.search(prev) and nxt[:1] == b" ":
            out.add(c)
    return sorted(out)


@dataclass
class Cut:
    idx: int  # order within the run
    f: float  # requested fraction
    pos: int  # prefix = seq[:pos]
    frac: float  # actual fraction of the span before the cut
    sentence: int  # number of sentence boundaries strictly before the cut (0 = span start)


def make_cuts(candidates: list[int], start: int, end: int, fractions: list[float]) -> list[Cut]:
    seen, out = set(), []
    rank = {c: i for i, c in enumerate(candidates)}
    for f in fractions:
        target = start + f * (end - start)
        c = min(candidates, key=lambda x: (abs(x - target), x))
        if c in seen:
            continue
        seen.add(c)
        out.append(Cut(len(out), f, c, round((c - start) / max(end - start, 1), 4), rank[c]))
    return out


def resample_seed(run_id: str, cut_pos: int, k: int) -> int:
    return int(hashlib.sha256(f"q6:{run_id}:{cut_pos}:{k}".encode()).hexdigest()[:8], 16)


# ---- planning ----------------------------------------------------------------------


@dataclass
class Plan:
    run_id: str
    case: str
    group: str
    question_id: str
    record: RunRecord
    original: dict  # original outcome fields
    span: tuple[int, int]
    cuts: list[Cut]
    marks: dict = field(default_factory=dict)  # positions of interest in the full sequence


def original_outcome(rec: RunRecord, label: dict, case: dict) -> dict:
    if case["outcome"] == "hack":
        return {"hack": bool(label["hack"]), "group": label["group"]}
    return {"answer": label["answer"], "correct": rec.correct_letter, "hinted": rec.hint_letter,
            "outcome": label.get("outcome")}


def first_match_pos(pieces: list[bytes], lo: int, hi: int, patterns: list[str]) -> int | None:
    """Index (full sequence) of the token holding the first byte of the first match
    of any pattern in the bytes of tokens [lo, hi); None if no match."""
    seg = pieces[lo:hi]
    offs = offsets_of(seg)
    data = b"".join(seg)
    hits = [m.start() for p in patterns if (m := re.search(p.encode(), data, re.IGNORECASE))]
    if not hits:
        return None
    b = min(hits)
    return lo + next(i for i in range(len(seg)) if offs[i] <= b < offs[i + 1])


def answer_mention_patterns(letter: str) -> list[str]:
    """The reasoning states a letter as the answer ("answer is B", "option B is
    correct", "(B)" + "correct"/"right"...). Coarse; used only for the 'first
    statement of the final answer' mark."""
    L = re.escape(letter)
    return [rf"answer (?:is|would be|should be|must be)\s*\(?\*{{0,2}}\s*(?:option\s*)?\(?{L}\b",
            rf"(?:option|choice)\s*\(?{L}\)?\s*(?:is|would be|must be|seems)\s*(?:the\s*)?(?:correct|right|answer)",
            rf"\({L}\)\s*is\s*(?:the\s*)?(?:correct|right|answer)",
            rf"(?:go with|choose|pick|select)\s*(?:option\s*)?\(?{L}\b"]


def plan_run(rec: RunRecord, label: dict, case: dict, group: str, tokenizer, fractions: list[float],
             mention_patterns: list[str] | None = None) -> Plan:
    from jmr.explore.turns import special_id
    from jmr.tokens import token_bytes

    seq = rec.prompt_token_ids + rec.completion_token_ids
    P = len(rec.prompt_token_ids)
    pieces = token_bytes(tokenizer, seq)
    orig = original_outcome(rec, label, case)
    if case["outcome"] == "hack":
        span = hack_span(pieces, P, orig["hack"])
    else:
        span = mcq_span(seq, P, special_id(tokenizer, THINK_START), special_id(tokenizer, THINK_END))
    if span is None:
        raise ValueError(f"{rec.run_id}: no span to cut")
    start, end = span
    cuts = make_cuts(sentence_candidates(pieces, start, end), start, end, fractions)
    marks = {"prompt_len": P, "span_start": start, "span_end": end, "seq_len": len(seq)}
    if case["outcome"] == "hack":
        from jmr.hack import token_windows

        w = token_windows(pieces[P:])
        marks["think_end"] = None if w.think_end is None else P + w.think_end
        marks["code_first"] = None if w.code_first is None else P + w.code_first
        marks["first_mention"] = None if w.mention is None else P + w.mention
    else:
        if mention_patterns:
            marks["first_mention"] = first_match_pos(pieces, start, end, mention_patterns)
        for key, letter in (("first_states_final", orig["answer"]), ("first_states_correct", rec.correct_letter),
                            ("first_states_hinted", rec.hint_letter)):
            if letter:
                marks[key] = first_match_pos(pieces, start, end, answer_mention_patterns(letter))
    return Plan(rec.run_id, case["name"], group, rec.question_id, rec, orig, span, cuts, marks)


def resample_requests(plan: Plan, n_resamples: int) -> list[dict]:
    rec = plan.record
    seq = rec.prompt_token_ids + rec.completion_token_ids
    P = len(rec.prompt_token_ids)
    budget = rec.sampling["max_tokens"]
    out = []
    for cut in plan.cuts:
        max_tokens = budget - (cut.pos - P)
        if max_tokens <= 0:
            raise ValueError(f"{rec.run_id}: cut {cut.pos} leaves no token budget")
        for k in range(n_resamples):
            out.append({"resample_id": f"{plan.run_id}/c{cut.idx}/r{k}", "run_id": plan.run_id, "cut": cut.idx,
                        "cut_pos": cut.pos, "k": k, "prompt_ids": seq[: cut.pos], "max_tokens": max_tokens,
                        "seed": resample_seed(plan.run_id, cut.pos, k), "sampling": rec.sampling})
    return out


# ---- outcomes ----------------------------------------------------------------------


def resample_outcome(plan: Plan, cut_pos: int, new_ids: list[int], finish_reason: str, tokenizer,
                     case: dict) -> dict:
    """Outcome of one resample: the original completion before the cut + the new
    tokens, labelled like the original runs."""
    from jmr.tokens import token_bytes

    rec = plan.record
    P = len(rec.prompt_token_ids)
    comp = rec.completion_token_ids[: cut_pos - P] + list(new_ids)
    text = b"".join(token_bytes(tokenizer, comp)).decode("utf-8", errors="replace")
    if case["outcome"] == "hack":
        from jmr.hack import label_run

        if finish_reason != "stop":
            return {"valid": False, "finish_reason": finish_reason}
        lab = label_run(text)
        valid = lab.group != "no_thinking"
        return {"valid": valid, "finish_reason": finish_reason, "hack": lab.hack, "group": lab.group,
                "same": valid and lab.hack == plan.original["hack"]}
    from jmr.label import extract_answer

    r = RunRecord(**{**rec.__dict__, "completion_token_ids": comp, "completion_text": text,
                     "finish_reason": finish_reason, "completion_logprobs": []})
    ans = extract_answer(r)
    return {"valid": ans is not None, "finish_reason": finish_reason, "answer": ans,
            "same": ans is not None and ans == plan.original["answer"]}


# ---- analysis ----------------------------------------------------------------------


def commitment_cut(p_same: list[float | None], threshold: float) -> int | None:
    """Smallest cut index k with p_same >= threshold at k and at every later cut
    (cuts with no valid resample are skipped). None if even the last cut with data
    is below the threshold (the outcome is still open at the end of the span)."""
    k_star = None
    for k in range(len(p_same) - 1, -1, -1):
        p = p_same[k]
        if p is None:
            continue
        if p >= threshold:
            k_star = k
        else:
            break
    return k_star


def refine_cuts(plan: Plan, p_same: list[float | None], threshold: float, pieces: list[bytes],
                max_new: int = 4, idx0: int = 100) -> list[Cut]:
    """Second pass: up to max_new sentence-boundary cuts strictly inside the
    commitment segment (between the last cut before the commitment cut and the
    commitment cut), evenly spread over the boundaries there. [] if the run
    commits at cut 0, never commits, or the segment has no inner boundary."""
    k = commitment_cut(p_same, threshold)
    if not k:
        return []
    a, b = plan.cuts[k - 1].pos, plan.cuts[k].pos
    start, end = plan.span
    allc = sentence_candidates(pieces, start, end)
    rank = {c: i for i, c in enumerate(allc)}
    inner = [c for c in allc if a < c < b]
    if len(inner) > max_new:
        inner = sorted({inner[round((i + 1) * (len(inner) + 1) / (max_new + 1)) - 1] for i in range(max_new)})
    return [Cut(idx0 + j, -1.0, c, round((c - start) / max(end - start, 1), 4), rank[c]) for j, c in enumerate(inner)]
