"""Paths, config loading and provenance. Shared by both envs: no heavy imports."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


@cache
def load(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def git_state() -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def snapshot(out_dir: Path) -> None:
    """Copy every config and the git state into out_dir, so outputs can be traced
    back to the settings that produced them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CONFIG_DIR, out_dir / "configs", dirs_exist_ok=True)
    (out_dir / "git.json").write_text(json.dumps(git_state(), indent=2))


def run_set_dir(run_set: str) -> Path:
    return DATA_DIR / "runs" / run_set


def m0_dir() -> Path:
    path = DATA_DIR / "m0"
    path.mkdir(parents=True, exist_ok=True)
    return path
