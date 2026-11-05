"""Build the findings dashboard, docs/dashboard/index.html.

Sources (the folders and files below are the source of truth):
  docs/dashboard/site.yaml        the main question, summary and strands (thread), terms, meta
  docs/experiments/<id>/README.md one experiment: YAML front matter (short structured fields)
                                  plus Markdown sections Question, Method, Result, Interpretation,
                                  Caveats (bullets) and Files
  docs/log.jsonl                  chronological record, one JSON object per line
This script assembles them into one in-memory structure, validates it against the schema below,
failing loudly on any missing or malformed field, then inlines the structure and the SVG
figures (drawn by scripts/docs/make_figures.py into the experiment folders) into one
self-contained HTML page, using the template scripts/docs/dashboard_template.html.

Python stdlib plus PyYAML (installed in envs/judge). Run from anywhere:
    ~/.local/bin/uv run --project envs/judge python scripts/docs/build_dashboard.py
    ... build_dashboard.py --check     # validate only, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "dashboard" / "site.yaml"
EXP_DIR = ROOT / "docs" / "experiments"
LOG = ROOT / "docs" / "log.jsonl"
TEMPLATE = ROOT / "scripts" / "docs" / "dashboard_template.html"
OUT = ROOT / "docs" / "dashboard" / "index.html"

STRAND_IDS = ("A", "B", "C", "T", "N")
STATUS = ("done", "running", "upcoming")
KIND = ("exploratory", "pre-registered", "definitive", "tooling")
VERDICT = ("supports", "against", "inconclusive", "instrument-limit", "context", "pending")
CONFIDENCE = ("high", "moderate", "low")
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")

# field -> (type check, may be null/empty)
STR, NUM, LIST, DICT, BOOL = "str", "num", "list", "dict", "bool"
EXP_FIELDS = {
    "id": (STR, False), "title": (STR, False), "strand": (STR, False), "also": (LIST, True),
    "status": (STR, False), "kind": (STR, False), "verdict": (STR, False),
    "question": (STR, False), "method": (STR, False), "result": (STR, False),
    "numbers": (LIST, True), "numbers_note": (STR, True), "interpretation": (STR, False), "confidence": (STR, True),
    "caveats": (LIST, True), "n": (DICT, False), "dates": (DICT, False), "timing": (DICT, False),
    "machine": (STR, False), "figure": (STR, True), "reports": (LIST, True), "data": (LIST, True),
    "commits": (LIST, True), "related": (LIST, True),
}
LOG_TYPES = ("decision", "start", "result", "dead_end", "surprise", "note")
LOG_BY = ("user", "claude")
LOG_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$")
FRONT_KEYS = ("id", "title", "strand", "also", "status", "kind", "verdict", "confidence", "dates", "timing",
              "machine", "n", "numbers", "numbers_note", "reports", "figure", "data", "commits", "related")
SECTIONS = {"Question": "question", "Method": "method", "Result": "result", "Interpretation": "interpretation"}
STRAND_FIELDS = ("id", "name", "question", "hypothesis", "answer", "confidence", "confidence_note", "next")


class RegistryError(Exception):
    pass


def _rel(folder: Path, p: str) -> str:
    """A path in a README, relative to its folder, as a repo-relative path."""
    return (folder / p).resolve().relative_to(ROOT).as_posix()


def parse_readme(path: Path) -> dict:
    """Turn one experiment README (front matter + sections) into a registry-style entry."""
    txt = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n(.*)\Z", txt, flags=re.S)
    if not m:
        raise RegistryError(f"{path}: no YAML front matter")
    fm = yaml.safe_load(m.group(1))
    if not isinstance(fm, dict):
        raise RegistryError(f"{path}: front matter is not a mapping")
    missing = [k for k in FRONT_KEYS if k not in fm]
    extra = [k for k in fm if k not in FRONT_KEYS]
    if missing or extra:
        raise RegistryError(f"{path}: front matter missing {missing}, unknown {extra}")
    if fm["id"] != path.parent.name:
        raise RegistryError(f"{path}: id {fm['id']!r} does not match its folder name")
    parts = re.split(r"^## (.+)$", m.group(2), flags=re.M)
    secs = {parts[i].strip(): parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}
    e = dict(fm)
    for head, key in SECTIONS.items():
        if head not in secs:
            raise RegistryError(f"{path}: missing section '## {head}'")
        e[key] = secs[head]
    cav = secs.get("Caveats", "")
    e["caveats"] = [ln[2:] for ln in cav.split("\n") if ln.startswith("- ")]
    if cav.strip() and len(e["caveats"]) != len([ln for ln in cav.split("\n") if ln.strip()]):
        raise RegistryError(f"{path}: every non-empty line of '## Caveats' must be a '- ' bullet")
    d = path.parent
    e["figure"] = Path(fm["figure"]).stem if fm["figure"] else None
    e["_figure_file"] = _rel(d, fm["figure"]) if fm["figure"] else None
    e["reports"] = [_rel(d, p) for p in fm["reports"] or []]
    e["data"] = [_rel(d, p) for p in fm["data"] or []]
    return e


def load_log() -> list[dict]:
    rows = []
    for i, ln in enumerate(LOG.read_text(encoding="utf-8").splitlines(), 1):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as exc:
                raise RegistryError(f"{LOG.name} line {i}: not valid JSON ({exc})")
    return rows


def load_registry() -> dict:
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    reg = {"meta": site.get("meta"), "thread": site.get("thread"), "terms": site.get("terms")}
    reg["experiments"] = [parse_readme(p) for p in sorted(EXP_DIR.glob("*/README.md"))]
    reg["log"] = load_log()
    return reg


def _type_ok(v, kind):
    return {STR: isinstance(v, str), NUM: isinstance(v, (int, float)) and not isinstance(v, bool),
            LIST: isinstance(v, list), DICT: isinstance(v, dict), BOOL: isinstance(v, bool)}[kind]


def _known_commits():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--all"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.split()


def validate(reg: dict) -> list[str]:
    """Raise RegistryError listing every problem; return warnings."""
    errs, warns = [], []
    for key in ("meta", "thread", "terms", "experiments", "log"):
        if key not in reg:
            errs.append(f"top level: missing '{key}'")
    if errs:
        raise RegistryError("\n".join(errs))

    meta = reg["meta"]
    for k in ("title", "updated", "default_setting"):
        if not isinstance(meta.get(k), str) or not meta.get(k):
            errs.append(f"meta.{k}: missing or not a non-empty string")
    if isinstance(meta.get("updated"), str) and not TIME_RE.match(meta["updated"]):
        errs.append("meta.updated: must be YYYY-MM-DD or YYYY-MM-DDTHH:MM")

    th = reg["thread"]
    for k in ("question", "summary"):
        if not isinstance(th.get(k), str) or not th.get(k):
            errs.append(f"thread.{k}: missing or empty")
    strands = th.get("strands") or []
    seen_s = [s.get("id") for s in strands]
    if sorted(seen_s) != sorted(STRAND_IDS):
        errs.append(f"thread.strands: ids must be exactly {STRAND_IDS}, got {seen_s}")
    for s in strands:
        for k in STRAND_FIELDS:
            if k not in s:
                errs.append(f"strand {s.get('id')}: missing '{k}'")
            elif k not in ("confidence_note", "next") and not s[k]:
                errs.append(f"strand {s.get('id')}: '{k}' is empty")
        if s.get("confidence") not in CONFIDENCE:
            errs.append(f"strand {s.get('id')}: confidence must be one of {CONFIDENCE}")

    for i, t in enumerate(reg["terms"]):
        if not t.get("term") or not t.get("def"):
            errs.append(f"terms[{i}]: needs 'term' and 'def'")

    commits = _known_commits()
    if commits is None:
        warns.append("git not available: commit hashes not checked")
    ids = [e.get("id") for e in reg["experiments"]]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errs.append(f"duplicate experiment ids: {sorted(dup)}")

    for e in reg["experiments"]:
        eid = e.get("id", "<no id>")
        where = f"experiment {eid}"
        for f, (kind, nullable) in EXP_FIELDS.items():
            if f not in e:
                errs.append(f"{where}: missing field '{f}'")
                continue
            v = e[f]
            if v is None:
                if not nullable:
                    errs.append(f"{where}: '{f}' must not be null")
                continue
            if not _type_ok(v, kind):
                errs.append(f"{where}: '{f}' must be {kind}")
            elif kind == STR and not nullable and not v.strip():
                errs.append(f"{where}: '{f}' is empty")
        extra = set(e) - set(EXP_FIELDS) - {"_figure_file"}
        if extra:
            errs.append(f"{where}: unknown fields {sorted(extra)}")
        if any(f not in e for f in EXP_FIELDS):
            continue
        if not ID_RE.match(eid):
            errs.append(f"{where}: id must be lowercase letters, digits and hyphens")
        if e["strand"] not in STRAND_IDS:
            errs.append(f"{where}: strand must be one of {STRAND_IDS}")
        for a in e["also"] or []:
            if a not in STRAND_IDS or a == e["strand"]:
                errs.append(f"{where}: bad 'also' entry {a!r}")
        if e["status"] not in STATUS:
            errs.append(f"{where}: status must be one of {STATUS}")
        if e["kind"] not in KIND:
            errs.append(f"{where}: kind must be one of {KIND}")
        if e["verdict"] not in VERDICT:
            errs.append(f"{where}: verdict must be one of {VERDICT}")
        if e["confidence"] is not None and e["confidence"] not in CONFIDENCE:
            errs.append(f"{where}: confidence must be null or one of {CONFIDENCE}")
        if e["numbers"] and e["numbers_note"] is not None and not e["numbers_note"].strip():
            errs.append(f"{where}: 'numbers_note' must be null or a non-empty string")
        for j, n in enumerate(e["numbers"] or []):
            if not isinstance(n, dict) or not n.get("label") or not n.get("value"):
                errs.append(f"{where}: numbers[{j}] needs 'label' and 'value'")
        for sub, keys in (("n", ("items", "runs")), ("dates", ("start", "end", "approx")),
                          ("timing", ("wall", "gpu_hours", "api_usd", "note"))):
            for k in keys:
                if k not in e[sub]:
                    errs.append(f"{where}: {sub}.{k} missing (use null if unknown)")
        d, tm = e["dates"], e["timing"]
        for k in ("start", "end"):
            if d.get(k) is not None and (not isinstance(d[k], str) or not TIME_RE.match(d[k])):
                errs.append(f"{where}: dates.{k} must be null, YYYY-MM-DD or YYYY-MM-DDTHH:MM")
        if not isinstance(d.get("approx"), bool):
            errs.append(f"{where}: dates.approx must be true or false")
        if d.get("start") and d.get("end") and d["end"][:10] < d["start"][:10]:
            errs.append(f"{where}: dates.end is before dates.start")
        for k in ("gpu_hours", "api_usd"):
            if tm.get(k) is not None and not _type_ok(tm[k], NUM):
                errs.append(f"{where}: timing.{k} must be a number or null")

        # status-dependent rules
        st = e["status"]
        if st == "done":
            if not d.get("start"):
                errs.append(f"{where}: done experiments need dates.start")
            if e["verdict"] == "pending":
                errs.append(f"{where}: a done experiment can't have verdict 'pending'")
            if not e["reports"] and not e["commits"]:
                errs.append(f"{where}: a done experiment needs a report or a commit")
            if e["confidence"] is None and e["verdict"] not in ("context",):
                errs.append(f"{where}: a done experiment needs a confidence (high/moderate/low)")
        elif st in ("running", "upcoming"):
            if e["verdict"] != "pending":
                errs.append(f"{where}: {st} experiments must have verdict 'pending'")
        if st == "running" and not d.get("start"):
            errs.append(f"{where}: running experiments need dates.start")
        if st == "upcoming" and d.get("start"):
            errs.append(f"{where}: upcoming experiments must have dates.start null")

        # files and commits
        if e["figure"] is not None and not (ROOT / e["_figure_file"]).is_file():
            errs.append(f"{where}: figure {e['_figure_file']} not found")
        for p in (e["reports"] or []) + (e["data"] or []):
            if not (ROOT / p).is_file():
                (warns if st != "done" else errs).append(f"{where}: file {p} not found")
        for c in e["commits"] or []:
            if not COMMIT_RE.match(c):
                errs.append(f"{where}: bad commit hash {c!r}")
            elif commits is not None and not any(h.startswith(c) for h in commits):
                errs.append(f"{where}: commit {c} not in this repository")
        for r in e["related"] or []:
            if r not in ids:
                errs.append(f"{where}: related id {r!r} not in the registry")

    for i, row in enumerate(reg["log"], 1):
        w = f"log entry {i}"
        if not isinstance(row, dict) or set(row) != {"ts", "type", "exp", "text", "by"}:
            errs.append(f"{w}: must have exactly the fields ts, type, exp, text, by")
            continue
        if not isinstance(row["ts"], str) or not LOG_TS_RE.match(row["ts"]):
            errs.append(f"{w}: bad ts {row['ts']!r}")
        if row["type"] not in LOG_TYPES:
            errs.append(f"{w}: type must be one of {LOG_TYPES}")
        if row["by"] not in LOG_BY:
            errs.append(f"{w}: by must be one of {LOG_BY}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            errs.append(f"{w}: empty text")
        if not isinstance(row["exp"], list) or any(x not in ids for x in row["exp"]):
            errs.append(f"{w}: exp must be a list of known experiment ids, got {row['exp']!r}")

    if errs:
        raise RegistryError("\n".join(errs))
    return warns


def figure_templates(reg: dict) -> str:
    used = {}
    for e in reg["experiments"]:
        if e["figure"]:
            if used.setdefault(e["figure"], e["_figure_file"]) != e["_figure_file"]:
                raise RegistryError(f"figure id {e['figure']} points at two different files")
    parts = []
    for fid in sorted(used):
        svg = (ROOT / used[fid]).read_text(encoding="utf-8")
        svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
        if "<script" in svg.lower():
            raise RegistryError(f"figure {fid}.svg contains a script; refusing to inline it")
        parts.append(f'<template id="fig-{fid}">{svg}</template>')
    return "\n".join(parts)


def build(reg: dict) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    figs = figure_templates(reg)
    reg = {**reg, "experiments": [{k: v for k, v in e.items() if not k.startswith("_")} for e in reg["experiments"]],
           "log": sorted(reg["log"], key=lambda r: r["ts"])}
    data = json.dumps(reg, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = reg["meta"]["title"].replace("&", "&amp;").replace("<", "&lt;")
    for marker in ("__TITLE__", "__REGISTRY_JSON__", "__FIG_TEMPLATES__"):
        if tpl.count(marker) != 1:
            raise RegistryError(f"template: expected exactly one {marker}")
    html = tpl.replace("__TITLE__", title).replace("__FIG_TEMPLATES__", figs)
    html = html.replace("__REGISTRY_JSON__", data)
    if not html.startswith("<!DOCTYPE html>"):
        raise RegistryError("output must start with <!DOCTYPE html>")
    return html


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="validate the registry only")
    args = ap.parse_args()
    try:
        reg = load_registry()
        if not isinstance(reg["meta"], dict) or not isinstance(reg["thread"], dict) or not isinstance(reg["terms"], list):
            raise RegistryError("site.yaml: needs meta (mapping), thread (mapping) and terms (list)")
        warns = validate(reg)
        for w in warns:
            print("warning:", w, file=sys.stderr)
        if args.check:
            print(f"sources OK: {len(reg['experiments'])} experiments, {len(reg['log'])} log entries")
            return 0
        html = build(reg)
    except RegistryError as exc:
        print("dashboard source check FAILED:\n" + str(exc), file=sys.stderr)
        return 1
    OUT.write_text(html, encoding="utf-8")
    counts = {s: sum(e["status"] == s for e in reg["experiments"]) for s in STATUS}
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB): "
          + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
