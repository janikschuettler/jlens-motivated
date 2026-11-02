"""Collecting pass/fail checks for the M0 scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jmr import config


class Checks:
    def __init__(self, name: str) -> None:
        self.name = name
        self.results: list[dict] = []
        self.info: dict = {}

    def check(self, label: str, passed: bool, detail=None) -> bool:
        self.results.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail is not None else ""))
        return passed

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)

    def save(self) -> Path:
        path = config.m0_dir() / f"{self.name}.json"
        payload = {
            "name": self.name,
            "all_passed": self.all_passed,
            "checks": self.results,
            "info": self.info,
            "git": config.git_state(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n{sum(r['passed'] for r in self.results)}/{len(self.results)} checks passed → {path}")
        return path

    def finish(self) -> None:
        self.save()
        sys.exit(0 if self.all_passed else 1)
