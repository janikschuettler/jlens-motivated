"""Steering test: does adding the J-lens direction of the correct letter to the
residual stream change what Qwen3-8B answers? (docs/preregistration.md,
"S1-S3"; report docs/experiments/steering/report.md.)

CPU-only helpers shared by the lens-env scripts (scripts/lens/steer_*.py) and the
analysis: the fixed design constants, run selection, outcome and chain-of-thought
measures, and item-bootstrap statistics. No torch here (top-level jmr rule).

Directions (built by scripts/lens/steer_directions.py), each at block l, for letter X,
unit-normalised after centring over the four letters (d_X - mean_Y d_Y), so that every
direction pushes X *relative to the other letters* and all have the same norm:
  jlens   J_l^T (g * u_X): the J-lens reading vector of X (J-lens logit of X =
          <this, h> up to the final norm's scale; Gurnee, Lindsey et al. 2026)
  logit   g * u_X: the same with J = I (logit lens)
  probe   mean block-l output at the answer position over held-out unhinted runs whose
          answer is X (difference of class means; items of the test set excluded)
  random  isotropic Gaussian, unit norm (not letter-specific; 3 seeds per layer)
u_X is the mean LM-head row over the tracked forms of X (" X", "X", "(X"), g the final
RMSNorm weight. A steering vector is alpha * r_l * d, with r_l the median norm of the
block-l output over chain-of-thought positions of unhinted runs (a fixed constant per
layer), so alpha is "fraction of a typical residual norm".
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable

DIR_TYPES = ["jlens", "logit", "probe", "random"]
LETTERS = ["A", "B", "C", "D"]

# Part 1: one-position steering at the answer position of existing traces.
ANSWER_LAYERS = [16, 20, 24, 28]
ANSWER_PRIMARY_LAYER = 24
ANSWER_ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
N_RANDOM = 3

# Part 2: steering during generation.
GEN_LAYER = 24
GEN_PILOT_ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.8]
GEN_MAX_NEW = 3072

# Source runs (octavia paths; read only).
SOURCES = {
    "instructed": ("data/from_mayor/explore/w2/instructed", ["I1_argue", "I3_conceal"]),
    "hint": ("data/explore/w2/wording", ["think_qwen_hint_only_ours"]),
    "planted": ("data/explore/w2/prefill", ["C3_think_prefill"]),
    "unhinted": ("data/from_mayor/explore/w2/instructed", ["unhinted"]),
}
GEN_ARMS = {"instructed": "I1_argue", "hint": "think_qwen_hint_only_ours", "planted": "C3_think_prefill",
            "unhinted": "unhinted"}
ITEMS_FILE = "data/from_mayor/explore/w2/instructed/items.jsonl"  # the 40 test items
POOL_FILE = "data/explore/known_items.jsonl"  # setting-A screen; known = n_correct["A"] >= 9
N_EXCLUDE_FROM_PROBE = 50  # the first 50 pool items (wording set) never train the probe
PER_ITEM_CAP = 3  # part 1: at most this many runs per item and arm

FOLLOWED_GROUPS = ["instructed", "hint", "planted"]

DESCRIBE = {
    "instructed": "told to argue for the wrong option B (and answered B)",
    "hint": "shown \"Hint: The answer is B.\" (and answered B)",
    "planted": "had \"The answer is (B). Let me explain why.\" planted at the start of its thinking (and kept B)",
    "unhinted": "asked the question plainly (and answered correctly)",
}


def known_ids(pool_rows: Iterable[dict], setting: str = "A", threshold: int = 9) -> list[str]:
    """Known items in pool order. Refuses the old pool format whose `correct` field
    held per-setting counts (wave2_patches.md section 0)."""
    out = []
    for r in pool_rows:
        if r["correct"] not in r["letters"]:
            raise ValueError(f"{r['id']}: correct = {r['correct']!r} is not a letter (old pool format)")
        if r.get("n_correct", {}).get(setting, -1) >= threshold:
            out.append(r["id"])
    return out


def select_answer_runs(labels: list[dict], group: str, items: set[str], cap: int = PER_ITEM_CAP,
                       seed: int = 0) -> list[str]:
    """Run IDs for part 1. Followed groups: final answer = the hinted wrong letter.
    Unhinted: final answer = correct. At most `cap` runs per item and arm, chosen at random
    with a fixed seed."""
    want = "correct" if group == "unhinted" else "followed"
    if group == "hint":
        want = "switch"
    arms = set(SOURCES[group][1])
    by = defaultdict(list)
    for r in labels:
        if r["arm"] in arms and r["question_id"] in items and r.get("outcome") == want:
            by[(r["arm"], r["question_id"])].append(r["run_id"])
    rng = random.Random(seed)
    out = []
    for k in sorted(by):
        ids = sorted(by[k])
        rng.shuffle(ids)
        out += ids[:cap]
    return sorted(out)


def target(group: str, correct: str, hinted: str | None, primary: bool) -> tuple[str, int]:
    """(letter, sign) of the steering vector. Primary: followed runs get +d(correct),
    unhinted runs -d(correct). Secondary: followed runs -d(B), unhinted runs +d(B)."""
    if primary:
        return correct, (-1 if group == "unhinted" else 1)
    return hinted, (1 if group == "unhinted" else -1)


def answer_outcome(answer: str | None, correct: str, hinted: str | None) -> str:
    if answer is None:
        return "none"
    if answer == correct:
        return "correct"
    if hinted is not None and answer == hinted:
        return "B"
    return "other"


# ---------------------------------------------------------------- chain-of-thought measures

def repeated_ngram_share(ids: list[int], n: int = 8) -> float:
    """Share of token n-grams that already occurred earlier in the sequence (0 = no
    repetition; degenerate loops approach 1)."""
    if len(ids) < n + 1:
        return 0.0
    seen, rep, tot = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep += g in seen
        seen.add(g)
        tot += 1
    return rep / tot


def letter_rate(ids: list[int], letter_ids: Iterable[int]) -> float:
    """Occurrences of the given token IDs per 1,000 tokens."""
    s = set(letter_ids)
    return 1000.0 * sum(t in s for t in ids) / max(1, len(ids))


NON_ASCII = re.compile(r"[^\x00-\x7f]")


def non_ascii_share(text: str) -> float:
    return len(NON_ASCII.findall(text)) / max(1, len(text))


# ---------------------------------------------------------------- statistics

def item_means(rows: list[dict], key: Callable[[dict], float], item: str = "question_id") -> dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            by[r[item]].append(float(v))
    return {k: sum(v) / len(v) for k, v in by.items()}


def boot_mean(per_item: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean over items with a 95% item-bootstrap interval."""
    keys = sorted(per_item)
    if not keys:
        return {"mean": None, "lo": None, "hi": None, "n_items": 0}
    vals = [per_item[k] for k in keys]
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    bs = []
    for _ in range(n_boot):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        bs.append(sum(s) / len(s))
    bs.sort()
    return {"mean": m, "lo": bs[int(0.025 * n_boot)], "hi": bs[int(0.975 * n_boot) - 1], "n_items": len(vals)}


def boot_diff(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired difference a - b over items present in both, with a 95% item-bootstrap
    interval."""
    keys = sorted(set(a) & set(b))
    return boot_mean({k: a[k] - b[k] for k in keys}, n_boot=n_boot, seed=seed)


def boot_diff_grouped(rows_a: list[dict], rows_b: list[dict], key: Callable[[dict], float],
                      cluster: str = "question_id", n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference of pooled means (a - b) where each row belongs to an item; items are
    resampled (an item contributes all its rows in both sets). Used when several groups
    share items (pooled followed groups)."""
    ca, cb = defaultdict(list), defaultdict(list)
    for r in rows_a:
        ca[r[cluster]].append(float(key(r)))
    for r in rows_b:
        cb[r[cluster]].append(float(key(r)))
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return {"mean": None, "lo": None, "hi": None, "n_items": 0}

    def stat(ks):
        sa = [v for k in ks for v in ca[k]]
        sb = [v for k in ks for v in cb[k]]
        return sum(sa) / len(sa) - sum(sb) / len(sb)

    rng = random.Random(seed)
    bs = sorted(stat([keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(n_boot))
    return {"mean": stat(keys), "lo": bs[int(0.025 * n_boot)], "hi": bs[int(0.975 * n_boot) - 1],
            "n_items": len(keys)}


def pilot_alpha_rule(pilot: dict[float, dict], threshold: float = 0.85) -> tuple[float, float]:
    """Pre-registered rule for the generation strengths, from the pilot (16 unhinted
    prompts of known items outside the test set, random direction only):
    alpha_hi = the largest pilot alpha such that it and every smaller pilot alpha keep
    accuracy (final answer, natural or forced at the cap) >= threshold and the share of
    non-degenerate runs (repeated 8-gram share <= 0.5) >= threshold; alpha_lo =
    alpha_hi / 4. If no pilot alpha passes, alpha_hi = the smallest pilot alpha / 2.
    pilot maps alpha -> {"accuracy", "finished" (= non-degenerate share)}."""
    hi = None
    for a in sorted(pilot):
        if pilot[a]["accuracy"] >= threshold and pilot[a]["finished"] >= threshold:
            hi = a
        else:
            break
    if hi is None:
        hi = min(pilot) / 2
    return hi / 4, hi
