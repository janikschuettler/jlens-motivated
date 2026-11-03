"""Item-level aggregation and bootstrap intervals for exploration labels (numpy only).

Rates are pooled ratios (sum of events / sum of valid runs over items); intervals
are 95% percentile bootstrap intervals resampling items.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

N_BOOT = 10_000


def counts_by_item(rows: list[dict], event, valid=lambda r: r["outcome"] != "excluded") -> dict[str, tuple[int, int]]:
    """item -> (events, valid runs)."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if valid(r):
            out[r["question_id"]][1] += 1
            out[r["question_id"]][0] += bool(event(r))
    return {k: (v[0], v[1]) for k, v in out.items()}


def pooled_rate(counts: dict[str, tuple[int, int]], items: list[str] | None = None, *, seed: int = 0) -> dict:
    items = [i for i in (items if items is not None else counts) if i in counts and counts[i][1] > 0]
    if not items:
        return {"rate": None, "lo": None, "hi": None, "events": 0, "n": 0, "items": 0}
    ev = np.array([counts[i][0] for i in items], float)
    n = np.array([counts[i][1] for i in items], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), (N_BOOT, len(items)))
    boot = ev[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"rate": float(ev.sum() / n.sum()), "lo": float(lo), "hi": float(hi),
            "events": int(ev.sum()), "n": int(n.sum()), "items": len(items)}


def mean_diff(values: np.ndarray, *, seed: int = 0) -> dict:
    """Mean over items of a per-item value (e.g. a paired difference), with a bootstrap CI."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"diff": None, "lo": None, "hi": None, "items": 0}
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (N_BOOT, len(values)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(values.mean()), "lo": float(lo), "hi": float(hi), "items": int(len(values))}


def fmt_rate(r: dict, pct: bool = True) -> str:
    if r.get("rate") is None:
        return "–"
    return f"{100 * r['rate']:.0f}% [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"


def fmt_diff(d: dict) -> str:
    if d.get("diff") is None:
        return "–"
    return f"{100 * d['diff']:+.0f} [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]"
