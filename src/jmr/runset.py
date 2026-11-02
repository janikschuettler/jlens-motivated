"""Run-set files: items, labels and the audit file. Shared by both envs (no heavy imports)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from jmr import config
from jmr.audit import write_audit
from jmr.label import label_counts, label_runs
from jmr.prompts import Question, hinted_letter
from jmr.records import RunRecord, read_runs, runs_path


def items_path(run_set_dir: Path) -> Path:
    return run_set_dir / "items.jsonl"


def item_rows(questions: list[Question]) -> list[dict]:
    return [{**asdict(q), "hinted": hinted_letter(q)} for q in questions]


def write_items(run_set_dir: Path, rows: list[dict]) -> None:
    with open(items_path(run_set_dir), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_items(run_set_dir: Path) -> dict[str, dict]:
    with open(items_path(run_set_dir)) as f:
        return {row["id"]: row for row in map(json.loads, filter(str.strip, f))}


def baseline_stats(records: list[RunRecord], keep_fraction: float) -> dict[str, dict]:
    """Per item: how many baseline samples are correct (unparseable ones count as
    not correct), and whether the item is reliably known."""
    from jmr.label import extract_answer

    stats: dict[str, dict] = {}
    for r in records:
        if r.condition != "baseline":
            continue
        s = stats.setdefault(r.question_id, {"baseline_correct": 0, "baseline_n": 0})
        s["baseline_n"] += 1
        s["baseline_correct"] += extract_answer(r) == r.correct_letter
    for s in stats.values():
        s["kept"] = s["baseline_correct"] >= keep_fraction * s["baseline_n"]
    return stats


def label_run_set(run_set: str, *, audit_per_group: int | None = 5, audit_seed: int = 0) -> list[dict]:
    """Write labels.jsonl, labels_summary.json and audit.md for a run set."""
    out_dir = config.run_set_dir(run_set)
    records = read_runs(runs_path(out_dir))
    items = read_items(out_dir) if items_path(out_dir).exists() else None
    rows = label_runs(records, items)
    with open(out_dir / "labels.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    reasons: dict[str, int] = {}
    for row in rows:
        if row["label"] == "excluded":
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    summary = {"counts": label_counts(rows), "excluded_reasons": reasons, "mention_check": "regex only"}
    (out_dir / "labels_summary.json").write_text(json.dumps(summary, indent=2))
    write_audit(
        out_dir / "audit.md", records, rows, per_group=audit_per_group, seed=audit_seed,
        note="Hint mentions are a regex check only; read the chains of thought to judge them. "
        "In null_control there is no hint: a switch means the model answered B anyway.",
    )
    return rows
