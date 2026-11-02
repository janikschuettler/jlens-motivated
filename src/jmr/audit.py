"""A human-readable audit file: prompts (with hints) and complete chains of thought,
grouped by label, with a summary on top. Shared by both envs; no heavy imports.

Chains of thought are never truncated. Which runs appear is a random sample of up
to `per_group` runs per label × condition (or all of them), fixed by a seed.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from jmr import config
from jmr.label import split_completion
from jmr.prompts import CONDITIONS, NUDGES
from jmr.records import RunRecord

# Labels in the order they are listed; any other label comes after these.
LABEL_ORDER = ["silent_switch", "acknowledged_switch", "resisted", "other_wrong", "correct", "wrong", "excluded"]


def _label_key(label: str) -> tuple[int, str]:
    return (LABEL_ORDER.index(label), label) if label in LABEL_ORDER else (len(LABEL_ORDER), label)


def fence(text: str, info: str = "") -> str:
    """A fenced block that can't be closed early by backticks inside the text."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{info}\n{text}\n{ticks}"


def select_rows(rows: list[dict], per_group: int | None, seed: int) -> list[dict]:
    """Up to per_group rows per (label, condition), chosen at random with a fixed
    seed; all rows if per_group is None. Returned sorted by label, condition, run ID."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in sorted(rows, key=lambda r: r["run_id"]):
        groups.setdefault((row["label"], row["condition"]), []).append(row)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(groups):
        group = groups[key]
        chosen += group if per_group is None or len(group) <= per_group else sorted(
            rng.sample(group, per_group), key=lambda r: r["run_id"]
        )
    cond_order = {c: i for i, c in enumerate(CONDITIONS)}
    return sorted(chosen, key=lambda r: (_label_key(r["label"]), cond_order.get(r["condition"], 99), r["run_id"]))


def _summary(rows: list[dict], shown: list[dict]) -> str:
    conditions = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    labels = sorted({r["label"] for r in rows}, key=_label_key)
    counts = Counter((r["label"], r["condition"]) for r in rows)
    shown_counts = Counter((r["label"], r["condition"]) for r in shown)
    nudged = [c for c in conditions if c in NUDGES]
    n_nudged = sum(1 for r in rows if r["condition"] in NUDGES)

    lines = ["| label | " + " | ".join(conditions) + " | all nudged | % of nudged |",
             "| --- |" + " ---: |" * (len(conditions) + 2)]
    for label in labels:
        cells = [str(counts[(label, c)]) for c in conditions]
        total = sum(counts[(label, c)] for c in nudged)
        pct = f"{100 * total / n_nudged:.0f}%" if total and n_nudged else ""
        lines.append(f"| **{label}** | " + " | ".join(cells) + f" | {total or ''} | {pct} |")
    lines.append("| *total* | " + " | ".join(str(sum(counts[(l, c)] for l in labels)) for c in conditions)
                 + f" | {n_nudged} | |")

    excluded = Counter(r.get("reason") for r in rows if r["label"] == "excluded")
    extra = ""
    if excluded:
        extra = "\n\nExcluded runs by reason: " + ", ".join(f"{reason} ({n})" for reason, n in excluded.most_common()) + "."

    shown_lines = ["| label | " + " | ".join(conditions) + " |", "| --- |" + " ---: |" * len(conditions)]
    for label in labels:
        shown_lines.append(f"| {label} | " + " | ".join(f"{shown_counts[(label, c)]} of {counts[(label, c)]}" for c in conditions) + " |")
    return "\n".join(lines) + extra + "\n\n**Shown below (examples per label × condition):**\n\n" + "\n".join(shown_lines)


def _example(index: int, total: int, record: RunRecord, row: dict) -> str:
    thinking, response = split_completion(record.completion_text)
    out = [f"### {row['label']} {index}/{total} · `{record.run_id}`", ""]
    facts = [
        f"**Condition:** {record.condition}",
        f"**Hinted option B:** {record.hint_letter or 'none'}" + (" (not in the prompt)" if record.condition == "null_control" else ""),
        f"**Correct option:** {record.correct_letter}",
        f"**Baseline accuracy on this item:** {row['baseline_correct']}/{row['baseline_n']} unhinted samples correct" if "baseline_correct" in row else "**Baseline accuracy on this item:** not recorded",
        f"**Final answer:** {row.get('answer') or 'none'}",
        f"**Label:** {row['label']}" + (f" ({row['reason']})" if row.get("reason") else ""),
    ]
    if record.condition in NUDGES:
        mentions = row.get("hint_mentions") or []
        facts.append("**Nudge mentioned in chain of thought (regex):** " + (", ".join(f"`{m}`" for m in mentions) if mentions else "no"))
    facts.append(f"**Completion:** {len(record.completion_token_ids)} tokens, finish reason `{record.finish_reason}`, temperature {record.sampling.get('temperature')}")
    out += [f"- {f}" for f in facts] + [""]
    for message in record.messages:
        role = message["role"] + (" (earlier turn, prefilled)" if message["role"] == "assistant" else "")
        out += [f"**Prompt, {role}:**", "", fence(message["content"]), ""]
    out += [f"**Chain of thought** ({len(thinking.strip())} characters, complete):", "", fence(thinking.strip()), ""]
    if response is None:
        out += ["**Response:** none (thinking never ended)", ""]
    else:
        out += ["**Response:**", "", fence(response.strip()), ""]
    return "\n".join(out)


def render(records: list[RunRecord], rows: list[dict], *, per_group: int | None, seed: int, note: str = "") -> str:
    by_id = {r.run_id: r for r in records}
    shown = select_rows(rows, per_group, seed)
    first = records[0]
    sampling = {k: first.sampling[k] for k in ("temperature", "top_p", "top_k", "max_tokens") if k in first.sampling}
    git = config.git_state()
    selection = "all runs" if per_group is None else f"up to {per_group} random runs per label × condition (seed {seed})"
    parts = [
        f"# Audit: run set `{first.run_set}`",
        "",
        f"- **Model:** {first.model.get('hf_id')} at `{str(first.model.get('revision'))[:8]}`, thinking {'on' if first.enable_thinking else 'off'}, sampling {sampling}",
        f"- **Runs:** {len(records)}; **shown:** {len(shown)} ({selection}). Chains of thought are complete, never truncated.",
        f"- **Code:** commit `{(git['commit'] or 'unknown')[:8]}`{' (uncommitted changes)' if git['dirty'] else ''}",
    ]
    if note:
        parts.append(f"- **Note:** {note}")
    parts += ["", "## Summary", "", _summary(rows, shown), ""]

    by_label: dict[str, list[dict]] = {}
    for row in shown:
        by_label.setdefault(row["label"], []).append(row)
    for label in sorted(by_label, key=_label_key):
        group = by_label[label]
        total = sum(1 for r in rows if r["label"] == label)
        parts += [f"## {label} ({len(group)} shown of {total})", ""]
        parts += [_example(i, len(group), by_id[row["run_id"]], row) for i, row in enumerate(group, 1)]
    return "\n".join(parts) + "\n"


def write_audit(path: Path, records: list[RunRecord], rows: list[dict], *, per_group: int | None = 5, seed: int = 0, note: str = "") -> Path:
    path.write_text(render(records, rows, per_group=per_group, seed=seed, note=note))
    return path
