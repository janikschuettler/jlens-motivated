"""The run record: what the gen env writes and the lens env reads.

Token IDs are the record of what was generated. The lens env teacher-forces
prompt_token_ids + completion_token_ids exactly and never re-tokenizes text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunRecord:
    run_id: str
    run_set: str
    question_id: str
    dataset: str
    condition: str  # unhinted | sycophancy | metadata | consistency
    hint_letter: str | None  # None for unhinted runs
    correct_letter: str
    letters: list[str]
    messages: list[dict]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    # Raw (pre-temperature) log-prob of each sampled token, as returned by vLLM.
    completion_logprobs: list[float]
    completion_text: str
    finish_reason: str
    seed: int
    model: dict  # hf_id, revision
    tokenizer_fingerprint: str
    sampling: dict
    enable_thinking: bool
    sample_idx: int = 0  # which of the samples drawn for this prompt

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.completion_token_ids


def runs_path(run_set_dir: Path) -> Path:
    return run_set_dir / "runs.jsonl"


def write_runs(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def read_runs(path: Path) -> list[RunRecord]:
    with open(path) as f:
        return [RunRecord(**json.loads(line)) for line in f if line.strip()]
