"""Loader for batch readouts (scripts/lens/readout_runs.py, format in
jmr.lens.batch). Numpy only, so it runs in either env.

    r = Readouts("data/readouts/temp_check", labels="data/runs/temp_check/labels.jsonl")
    idx = r.select(condition="known_evidence", label="resisted")
    x = r.at_anchor("jlens", "correct", "pre_answer", runs=idx)        # [runs, layers] log-prob
    y = r.cot_bins("logit", "hinted", n_bins=20, what="rank", runs=idx)  # [runs, layers, bins]

Lenses: "jlens", "logit" (layers = workspace layers) and "model" (the model's
own distribution, block 35; one layer). Roles: "correct", "hinted" (the run's hint letter, else the item's hinted
option if items.jsonl was given), "answer"
(the run's parsed final answer), "other_wrong_max" (best of the wrong options
other than the hinted one: highest log-prob, lowest rank), "other_wrong_mean",
or a literal letter "A".."D". Forms: "space" (" A"), "bare" ("A"), "paren"
("(A"), or "any" (log-sum-exp of the forms' log-probs; min of their ranks).
what: "lp" (log-prob, nats), "rank" (1 = top; min-rank tie rule), "logrank"
(log10 rank). Missing letters or anchors give NaN.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

LENS_INDEX = {"jlens": 0, "logit": 1}


class Readouts:
    def __init__(self, out_dir: str | Path, labels: str | Path | None = None,
                 items: str | Path | None = None) -> None:
        """labels: a labels.jsonl to join by run_id (adds meta["label"]).
        items: an items.jsonl; adds meta["item_hinted"], the item's hinted
        option, which role "hinted" uses for runs without a hint (unhinted)."""
        self.dir = Path(out_dir)
        with open(self.dir / "index.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        seen: dict[str, dict] = {}
        for m in rows:  # a resumed run may repeat a line; keep the last
            seen[m["run_id"]] = m
        self.meta = list(seen.values())
        if labels is not None:
            with open(labels) as f:
                lab = {d["run_id"]: d for d in map(json.loads, f) if d}
            for m in self.meta:
                m["label"] = lab.get(m["run_id"], {}).get("label")
        if items is not None:
            with open(items) as f:
                hinted = {d["id"]: d.get("hinted") for d in map(json.loads, f) if d}
            for m in self.meta:
                m["item_hinted"] = hinted.get(m["question_id"])
        first = self.meta[0]["tracked"]
        self.letters, self.forms = first["letters"], first["forms"]
        self._load = lru_cache(maxsize=256)(self._load_uncached)
        self.layers = self.load(0)["layers"].tolist()

    def __len__(self) -> int:
        return len(self.meta)

    # ------------------------------------------------------------ basics

    def _load_uncached(self, i: int) -> dict:
        # Everything but the block outputs (the largest array; resid() reads them).
        with np.load(self.dir / self.meta[i]["file"]) as z:
            return {k: z[k] for k in z.files if k not in ("meta_json", "resid")}

    def load(self, i: int) -> dict:
        return self._load(i)

    def select(self, **filters) -> list[int]:
        """Indices of runs whose meta (or meta['extras']) matches every filter.
        A filter value may be a list (any of)."""
        out = []
        for i, m in enumerate(self.meta):
            ok = True
            for k, v in filters.items():
                have = m.get(k, m.get("extras", {}).get(k))
                if (have not in v) if isinstance(v, (list, tuple, set)) else (have != v):
                    ok = False
                    break
            if ok:
                out.append(i)
        return out

    def _letters_for(self, m: dict, role: str) -> list[str]:
        if role in self.letters:
            return [role]
        if role == "correct":
            return [m["correct_letter"]] if m.get("correct_letter") else []
        if role == "hinted":
            b = m.get("hint_letter") or m.get("item_hinted")
            return [b] if b else []
        if role == "answer":
            return [m["answer"]] if m.get("answer") else []
        if role.startswith("other_wrong"):
            b = m.get("hint_letter") or m.get("item_hinted")
            return [x for x in m["letters"] if x not in (m.get("correct_letter"), b)]
        raise ValueError(f"unknown role {role!r}")

    def series(self, i: int, lens: str, letter: str, *, form: str = "space", what: str = "lp") -> np.ndarray:
        """[layers, positions] for one letter (positions from positions_lo on).
        For lens "model", layers has length 1."""
        a = self.load(i)
        if lens == "model":
            lp, rank = a["model_lp"][None], a["model_rank"][None]
        else:
            lp, rank = a["lp"][LENS_INDEX[lens]], a["rank"][LENS_INDEX[lens]]
        forms = self.forms if form == "any" else [form]
        cols = [self.letters.index(letter) * len(self.forms) + self.forms.index(f) for f in forms]
        if what == "lp":
            x = lp[..., cols].astype(np.float64)
            return np.logaddexp.reduce(x, axis=-1) if len(cols) > 1 else x[..., 0]
        r = rank[..., cols].min(axis=-1).astype(np.float64)
        return np.log10(r) if what == "logrank" else r

    def role_series(self, i: int, lens: str, role: str, *, form: str = "space", what: str = "lp") -> np.ndarray | None:
        letters = self._letters_for(self.meta[i], role)
        if not letters:
            return None
        xs = np.stack([self.series(i, lens, x, form=form, what=what) for x in letters])
        if role == "other_wrong_mean":
            return xs.mean(0)
        if role == "other_wrong_max" or len(letters) > 1:
            return xs.max(0) if what == "lp" else xs.min(0)
        return xs[0]

    def _n_layers(self, lens: str) -> int:
        return 1 if lens == "model" else len(self.layers)

    def position(self, i: int, anchor: str) -> int:
        """Absolute position of a named anchor (-1 if absent)."""
        return int(self.meta[i]["named_positions"].get(anchor, -1))

    # ------------------------------------------------------- aggregates

    def at_anchor(self, lens: str, role: str, anchor: str, *, form: str = "space", what: str = "lp",
                  runs: list[int] | None = None) -> np.ndarray:
        """[runs, layers] at a named anchor (last_prompt, think_start, think_end,
        pre_first_stated, pre_answer, cot_00..cot_15)."""
        runs = range(len(self)) if runs is None else runs
        out = np.full((len(runs), self._n_layers(lens)), np.nan)
        for j, i in enumerate(runs):
            pos = self.position(i, anchor)
            s = self.role_series(i, lens, role, form=form, what=what)
            if pos >= 0 and s is not None:
                out[j] = s[:, pos - int(self.load(i)["positions_lo"])]
        return out

    def cot_bins(self, lens: str, role: str, *, n_bins: int = 20, form: str = "space", what: str = "lp",
                 runs: list[int] | None = None) -> np.ndarray:
        """[runs, layers, n_bins]: mean over the positions of the reasoning span
        (inside <think>…</think>; see jmr.lens.batch.plan_run), binned by
        normalised position (p - start) / (end - start)."""
        runs = range(len(self)) if runs is None else runs
        out = np.full((len(runs), self._n_layers(lens), n_bins), np.nan)
        for j, i in enumerate(runs):
            span = self.meta[i]["cot_span"]
            s = self.role_series(i, lens, role, form=form, what=what)
            if span is None or s is None:
                continue
            start, end = span
            lo = int(self.load(i)["positions_lo"])
            x = s[:, start - lo : end - lo]
            b = np.minimum((np.arange(end - start) * n_bins) // (end - start), n_bins - 1)
            counts = np.bincount(b, minlength=n_bins)
            for li in range(x.shape[0]):
                sums = np.bincount(b, weights=x[li], minlength=n_bins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[j, li] = np.where(counts > 0, sums / counts, np.nan)
        return out

    def top_tokens(self, i: int, lens: str, anchor: str, layer: int | None = None) -> list[tuple[int, float]]:
        """Stored top-25 (ID, log-prob) at an anchor; layer ignored for "model"."""
        a = self.load(i)
        names = self.meta[i]["top_anchor_names"]
        if anchor not in names:
            return []
        k = names.index(anchor)
        if lens == "model":
            ids, lp = a["model_top_ids"][k], a["model_top_lp"][k]
        else:
            li = self.layers.index(layer)
            ids, lp = a["top_ids"][LENS_INDEX[lens], li, k], a["top_lp"][LENS_INDEX[lens], li, k]
        return [(int(t), float(v)) for t, v in zip(ids, lp) if t >= 0]

    def resid(self, i: int, anchor: str) -> np.ndarray | None:
        """Block outputs [36, d_model] (fp16) at a residual anchor, or None if
        the anchor is absent in this run."""
        names = self.meta[i]["resid_anchor_names"]
        k = names.index(anchor)
        if self.load(i)["resid_pos"][k] < 0:
            return None
        with np.load(self.dir / self.meta[i]["file"]) as z:
            return z["resid"][:, k]
