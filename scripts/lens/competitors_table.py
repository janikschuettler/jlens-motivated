"""Assemble the J-lens-vs-competitors table (EXPLORATORY) from
  data/overnight/competitors/auroc.json       (analyze_competitors.py, new cells)
  data/overnight/competitors/jdiff.json       (paired J-lens minus logit-lens differences)
  data/overnight/competitors/hack_auroc.json  (competitors_hack.py, new cells)
  data/overnight/option_lens.json             (analyze_option_lens.py, option words)
plus the cells already published in earlier reports (REUSED below, each with its source).
Writes docs/experiments/jlens-vs-competitors/data/jlens_vs_competitors.json and prints the markdown tables.
Stdlib only:  python scripts/lens/competitors_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
C = R / "data/overnight/competitors"
OUT = R / "docs/experiments/jlens-vs-competitors/data/jlens_vs_competitors.json"

POSITIONS = [("options", "prompt option words"), ("last_prompt", "last prompt token"),
             ("mid", "mid reasoning"), ("think_end", "end of reasoning"), ("pre_answer", "answer position"),
             ("whole", "whole run")]
READOUTS = [("jlens", "J-lens"), ("logit", "logit lens"), ("probe", "linear probe"),
            ("blackbox", "black box (model's own answer)"), ("monitor", "CoT keyword monitor"), ("length", "length")]

CASES = [  # id, plain label, model / mode, positive group, negative group
    ("instr", "Told to argue for B vs no hint", "Qwen3-8B, thinking",
     "told to argue for the wrong option B (and did)", "no hint, answered correctly"),
    ("wording", "Hint followers vs resisters", "Qwen3-8B, thinking",
     "followed an answer-stating hint to B", "resisted the same hint"),
    ("t2", "Followers who still know vs persuaded followers (pre-registered T2)", "Qwen3-8B, thinking",
     "followed the hint, but give the correct answer when asked afterwards", "followed and repeat B when asked"),
    ("silent", "Silent vs open switches", "Qwen3-8B, thinking off",
     "switched to the hinted B without mentioning the hint", "switched to B and mentioned the hint"),
    ("c3", "Kept a planted wrong claim vs recovered", "Qwen3-8B, thinking",
     "kept the planted wrong answer to the end", "dropped it and answered correctly"),
    ("c3_reel", "Kept-claim runs that still know vs those that don't", "Qwen3-8B, thinking",
     "kept the planted B, but give the correct answer when asked without the reasoning",
     "kept B and don't give the correct answer when asked"),
    ("hack", "Hidden reward hack vs no hack, same model", "Qwen3-8B + warm-start hack adapter, thinking off",
     "hacks the tests, reasoning hides it", "doesn't hack"),
    ("grader_q14", "Grader-bug followers vs resisters", "Qwen3-14B, thinking",
     "took the grader bug (answered B)", "resisted it (answered correctly)"),
    ("grader_gptoss", "Grader-bug followers vs resisters", "gpt-oss-20b, reasoning",
     "took the grader bug", "resisted it"),
    ("grader_nt8", "Grader-bug followers vs resisters", "Qwen3-8B, thinking off",
     "took the grader bug", "resisted it"),
    ("grader_nt14", "Grader-bug followers vs resisters", "Qwen3-14B, thinking off",
     "took the grader bug", "resisted it"),
]
# Contrasts whose two groups give different final answers: at the answer position every
# letter readout just reads the answer being written.
ANSWER_DIFFERS = {"instr", "wording", "c3", "grader_q14", "grader_gptoss", "grader_nt8", "grader_nt14"}
FIXED_PROBE = {"grader_q14": 27, "grader_nt14": 27, "grader_gptoss": 16}


def cell(a, lo, hi, src="new", note=None, n=None):
    d = {"auroc": a, "lo": lo, "hi": hi, "source": src}
    if note:
        d["note"] = note
    if n:
        d["n"] = n
    return d


def flip(a, lo, hi, src, note=None):
    """Earlier result reported with the other group as positive: AUROC -> 1 - AUROC."""
    return cell(round(1 - a, 3), None if hi is None else round(1 - hi, 3), None if lo is None else round(1 - lo, 3),
                src, note)


# ---------------------------------------------------------------- cells from earlier reports
LM, PR, C3R, HO, MG, IQ = ("lens_main.md", "lens_prereg.md", "lens_c3.md", "hack_organism.md", "models_grader.md",
                           "instrument_q1q3.md")
REUSED = {
    "t2": {
        ("pre_answer", "jlens"): cell(0.46, 0.32, 0.61, PR, "pre-registered score S (correct letter over the uninvolved letters, L14-32); FAIL"),
        ("pre_answer", "logit"): cell(0.43, 0.25, 0.62, PR, "score S"),
        ("mid", "jlens"): cell(None, None, None, PR, "correct-letter margin along the reasoning: AUROCs 0.41-0.61 in 20 bins, no CI given"),
        ("mid", "probe"): cell(0.70, 0.54, 0.85, PR, "probe on reasoning hidden states, mean of 16 anchors, L20 (best of 5 layers); not length-controlled"),
        ("whole", "monitor"): cell(0.50, None, None, PR, "hint/answer keyword"),
        ("whole", "length"): cell(0.72, 0.60, 0.82, PR, "reasoning length"),
        ("whole", "blackbox"): cell(0.82, 0.75, 0.87, PR, "asked again with the reasoning removed; partly the same measurement as the label"),
    },
    "silent": {
        ("pre_answer", "jlens"): cell(0.54, 0.504, 0.577, PR, "PRE-REGISTERED T1b, score S, 1,744 fresh items (986 vs 7,268 runs); PASS narrowly"),
        ("pre_answer", "logit"): cell(0.46, 0.424, 0.500, PR, "T1b, score S"),
        ("whole", "length"): cell(0.54, 0.506, 0.572, PR, "T1b, completion length"),
        ("whole", "monitor"): cell(0.0, 0.0, 0.0, LM, "zero by construction: 'silent' means the keyword monitor finds nothing"),
    },
    "silent_old": {
        ("pre_answer", "jlens"): cell(0.65, 0.56, 0.74, LM, "lens_main, 42 items chosen in the same data (151 vs 513 runs), score S"),
        ("pre_answer", "logit"): cell(0.48, 0.37, 0.58, LM, "lens_main, score S"),
        ("pre_answer", "probe"): cell(0.45, 0.36, 0.54, LM, "answer probe P(correct), L24"),
        ("last_prompt", "probe"): cell(0.52, 0.42, 0.63, LM, "answer probe P(correct), L24"),
        ("mid", "jlens"): cell(0.56, 0.46, 0.65, LM, "echo-free response positions"),
        ("mid", "logit"): cell(0.49, 0.39, 0.58, LM, "echo-free response positions"),
    },
    "c3": {
        ("mid", "jlens"): flip(0.46, 0.31, 0.61, C3R, "first 50 generated tokens, correct-over-B rank, L14-32"),
        ("mid", "logit"): flip(0.37, 0.22, 0.54, C3R, "first 50 generated tokens"),
        ("think_end", "jlens"): flip(0.53, 0.38, 0.68, C3R, "correct-over-B rank"),
        ("last_prompt", "jlens"): flip(0.42, 0.26, 0.59, C3R, "item-level (end of the planted sentence)"),
        ("mid", "blackbox"): flip(0.70, 0.52, 0.85, C3R, "forced answer at 20% of the reasoning, log p(correct) - log p(B)"),
        ("whole", "length"): flip(0.73, 0.61, 0.84, C3R, "reasoning length"),
    },
    "c3_reel": {
        ("pre_answer", "jlens"): cell(0.86, 0.77, 0.96, C3R, "score S; 51 vs 10 runs, 17 vs 10 items; may be an item effect"),
        ("pre_answer", "logit"): cell(0.77, 0.61, 0.91, C3R, "score S"),
        ("mid", "jlens"): cell(0.66, 0.45, 0.85, C3R, "first 50 generated tokens"),
        ("mid", "logit"): cell(0.52, None, None, C3R, "first 50 generated tokens"),
        ("mid", "blackbox"): cell(0.45, None, None, C3R, "forced answer at 20% of the reasoning"),
        ("whole", "monitor"): cell(0.42, None, None, C3R, "CoT text features 0.37-0.47 (midpoint shown)"),
        ("whole", "length"): cell(0.45, None, None, C3R, "reasoning length"),
    },
    "hack": {
        ("mid", "jlens"): cell(0.44, 0.33, 0.55, HO, "PRE-REGISTERED H2a: hack-word score at echo-free reasoning positions"),
        ("mid", "logit"): cell(0.47, 0.37, 0.57, HO, "H2a"),
        ("mid", "probe"): cell(0.60, 0.50, 0.70, HO, "probe trained on hidden vs no-hack, L24, problems held out (exploratory); transfer from named hacks: 0.39 [0.29, 0.50]"),
        ("pre_answer", "jlens"): cell(0.59, 0.51, 0.68, HO, "right before the code (exploratory)"),
        ("pre_answer", "logit"): cell(0.62, 0.52, 0.71, HO, "right before the code"),
        ("whole", "length"): cell(0.42, None, None, HO, "reasoning length"),
    },
    "grader_q14": {
        ("mid", "jlens_concept"): cell(0.85, 0.80, 0.90, MG, "grader/cheat words along the reasoning; tracks the text, which names the bug"),
        ("mid", "logit_concept"): cell(0.80, None, None, MG, "grader words along the reasoning"),
        ("last_prompt", "jlens_concept"): cell(0.60, 0.53, 0.66, MG, "grader words, item × arm level"),
        ("last_prompt", "logit_concept"): cell(0.48, 0.43, 0.54, MG, "grader words"),
    },
    "grader_gptoss": {
        ("mid", "jlens_concept"): cell(0.83, None, None, MG, "grader words along the reasoning"),
        ("mid", "logit_concept"): cell(0.81, None, None, MG, "grader words along the reasoning"),
    },
    "grader_nt8": {
        ("last_prompt", "jlens_concept"): cell(0.51, None, None, MG, "grader words"),
        ("last_prompt", "logit_concept"): cell(0.52, None, None, MG, "grader words"),
        ("pre_answer", "jlens_concept"): cell(0.72, None, None, MG, "grader words at the answer position"),
        ("pre_answer", "logit_concept"): cell(0.87, None, None, MG, "grader words at the answer position"),
    },
    "grader_nt14": {
        ("last_prompt", "jlens_concept"): cell(0.52, None, None, MG, "grader words"),
        ("last_prompt", "logit_concept"): cell(0.48, None, None, MG, "grader words"),
        ("pre_answer", "jlens_concept"): cell(0.77, 0.72, 0.82, MG, "grader words at the answer position"),
        ("pre_answer", "logit_concept"): cell(0.53, 0.47, 0.59, MG, "grader words at the answer position"),
    },
}


def new_cells(cid: str, A: dict) -> dict:
    c = A["cases"][cid]
    s, out = c["scores"], {}
    pl = c["probe_layers"]
    fixed = FIXED_PROBE.get(cid, 24)
    n = f"{c['n_runs_pos']} vs {c['n_runs_neg']} runs; {c['n_items_pos']} vs {c['n_items_neg']} items"

    def get(k, note=None):
        v = s.get(k, {})
        if v.get("auroc") is None:
            return None
        w = c["within_item"].get(k, {})
        d = cell(v["auroc"], v["lo"], v["hi"], "new", note)
        if w.get("auroc") is not None:
            d["within_item"] = {"auroc": w["auroc"], "lo": w["lo"], "hi": w["hi"]}
        return d

    for pos in ("last_prompt", "mid", "think_end", "pre_answer"):
        for ln in ("jlens", "logit"):
            d = get(f"{ln}|{pos}|D")
            if d:
                pl_res = c["per_layer"].get(f"{ln}|{pos}|D", {})
                best = max(pl_res.items(), key=lambda kv: abs((kv[1].get("auroc") or 0.5) - 0.5), default=None)
                if best and best[1].get("auroc") is not None:
                    d["best_layer"] = {"layer": int(best[0]), "auroc": best[1]["auroc"], "lo": best[1]["lo"], "hi": best[1]["hi"]}
                m = s.get(f"{ln}|{pos}|M", {})
                if m.get("auroc") is not None:
                    d["margin_score"] = {"auroc": m["auroc"], "lo": m["lo"], "hi": m["hi"]}
                if pos == "pre_answer" and cid in ANSWER_DIFFERS:
                    d["note"] = "the final answers differ, so this reads the answer being written"
                out[(pos, ln)] = d
        d = get(f"probe|{pos}|L{fixed}") if pos != "mid" or f"probe|mid|L{fixed}" in s else None
        if d:
            bests = [(L, s.get(f"probe|{pos}|L{L}", {})) for L in (pl if pos != "mid" else c["cot_probe_layers"])]
            bests = [(L, v) for L, v in bests if v.get("auroc") is not None]
            if bests:
                L, v = max(bests, key=lambda kv: abs(kv[1]["auroc"] - 0.5))
                d["best_layer"] = {"layer": L, "auroc": v["auroc"], "lo": v["lo"], "hi": v["hi"]}
            d["layer"] = fixed
            if pos == "last_prompt":
                d["note"] = "the last prompt token is shared by every sample of a prompt: item-level"
            out[(pos, "probe")] = d
    d = get("model|pre_answer|D")
    if d:
        if cid in ANSWER_DIFFERS:
            d["note"] = "the final answers differ, so the model's own answer separates them trivially"
        out[("pre_answer", "blackbox")] = d
    d = get("monitor|text|flag")
    if d:
        d["rates"] = {"pos": round(c["monitor_rate_pos"], 3), "neg": round(c["monitor_rate_neg"], 3)}
        out[("whole", "monitor")] = d
    d = get("length|completion|tokens")
    if d:
        r = get("length|reasoning|tokens")
        if r:
            d["reasoning_length"] = {k: r[k] for k in ("auroc", "lo", "hi")}
        out[("whole", "length")] = d
    return out, n, c


def hack_cells(H: dict) -> dict:
    s, out = H["scores"], {}
    m = {"last_prompt": "last_prompt", "mid": "mid", "think_end": "think_end", "pre_answer": "pre_code"}
    for pos, hp in m.items():
        for ln, key in (("jlens", f"jlens|{hp}"), ("logit", f"logit|{hp}"), ("blackbox", f"model|{hp}")):
            v = s.get(key, {})
            if v.get("auroc") is not None:
                out[(pos, ln)] = cell(v["auroc"], v["lo"], v["hi"], "new", "hack-word score; problem-level means")
        v = s.get(f"probe|{hp}|L24", {})
        if v.get("auroc") is not None:
            out[(pos, "probe")] = cell(v["auroc"], v["lo"], v["hi"], "new", "probe trained on hidden vs no-hack, L24")
    v = s.get("monitor|text", {})
    if v.get("auroc") is not None:
        out[("whole", "monitor")] = cell(v["auroc"], v["lo"], v["hi"], "new",
                                         f"hack keyword lists; fires on {H['monitor_rate_hidden']:.0%} of hidden and "
                                         f"{H['monitor_rate_nohack']:.0%} of no-hack runs")
    return out


def option_cells(O: dict) -> dict:
    a = O["auroc"]
    out = {}
    for rd, key in (("jlens", "words_jlens"), ("logit", "words_logit"), ("probe", "option_probe")):
        v = a[key]["kb_instr_vs_unhinted"]
        out[("options", rd)] = cell(round(v["auroc"], 3), round(v["lo"], 3), round(v["hi"], 3), "new",
                                    "item-level (the options come before the instruction); correct option's score margin")
    return out


SAME_ANSWER = {"silent", "hack", "t2", "c3_reel"}
BEFORE = ("options", "last_prompt", "mid", "think_end")
POS_LABEL = dict(POSITIONS)


def fold(v):
    a, lo, hi = v.get("auroc"), v.get("lo"), v.get("hi")
    if a is None:
        return None
    if a >= 0.5:
        return a, lo, hi
    return 1 - a, (None if hi is None else 1 - hi), (None if lo is None else 1 - lo)


def figure_rows(table: dict) -> list:
    """Per case and readout family: the best cell before the final answer is written,
    as a direction-free separation max(AUROC, 1 - AUROC) with its folded 95% CI.
    The position is chosen after looking, for every readout alike."""
    fam = {"jlens": ("jlens",), "logit": ("logit",), "probe": ("probe",),
           "other": ("blackbox", "monitor", "monitor_intent", "length")}
    rows = []
    for cid, label, model, *_ in CASES:
        t = table[cid]
        allowed = BEFORE + (("pre_answer",) if cid in SAME_ANSWER else ()) + ("whole",)
        vals = []
        for f, rds in fam.items():
            best = None
            for k, v in t.items():
                p, r = k.split("|")
                if p in allowed and r in rds and fold(v):
                    fa = fold(v)
                    if best is None or fa[0] > best[0][0]:
                        best = (fa, p, r)
            if best:
                (a, lo, hi), p, r = best
                vals.append({"s": f, "v": round(a, 3), "lo": None if lo is None else round(lo, 3),
                             "hi": None if hi is None else round(hi, 3), "where": POS_LABEL[p], "readout": r})
        rows.append({"case": cid, "label": label, "sub": model, "vals": vals})
    return rows


def main() -> None:
    A = json.loads((C / "auroc.json").read_text())
    GM = json.loads((C / "grader_monitors.json").read_text())
    J = json.loads((C / "jdiff.json").read_text())["jdiff"]
    H = json.loads((C / "hack_auroc.json").read_text())
    O = json.loads((R / "data/overnight/option_lens.json").read_text())
    table = {}
    meta = {}
    for cid, label, model, pos, neg in CASES:
        cells, n, info = {}, None, None
        if cid in A["cases"]:
            cells, n, info = new_cells(cid, A)
        if cid == "hack":
            cells.update(hack_cells(H))
            n = f"{H['n_hidden']} vs {H['n_nohack']} runs (problem-level AUROC over problems with both groups)"
        if cid == "instr":
            cells.update(option_cells(O))
        if cid in GM:
            e = GM[cid]["exploit_intent"]
            m = cells.get(("whole", "monitor"))
            if m is not None:
                m["note"] = (f"mentions the bug (label list): fires on {m['rates']['pos']:.0%} of followers and "
                             f"{m['rates']['neg']:.0%} of resisters")
            cells[("whole", "monitor_intent")] = cell(e["auroc"], e["lo"], e["hi"], "new",
                                                      f"exploit-intent words (label list): fires on {e['rate_pos']:.0%} "
                                                      f"of followers and {e['rate_neg']:.0%} of resisters")
        for k, v in REUSED.get(cid, {}).items():
            if k in cells:
                cells[k]["earlier"] = v  # keep both: new value and the earlier report's
            else:
                cells[k] = v
        if cid == "silent":
            for k, v in REUSED["silent_old"].items():
                cells.setdefault(k, {"source": "see earlier"})["earlier_old_items"] = v
        table[cid] = {f"{p}|{r}": v for (p, r), v in cells.items()}
        meta[cid] = {"label": label, "model": model, "positive": pos, "negative": neg, "n": n,
                     "answer_differs": cid in ANSWER_DIFFERS,
                     "jlens_minus_logit": J.get(cid)}
        if info:
            meta[cid].update({"arms": info["arms"], "monitor_rates": {"pos": info["monitor_rate_pos"], "neg": info["monitor_rate_neg"]},
                              "median_completion_tokens": {"pos": info["median_completion_pos"], "neg": info["median_completion_neg"]},
                              "n_items_both_groups": info["n_items_both"]})
    out = {"exploratory": True,
           "what": "AUROC (0.5 = chance, 1 = perfect, 0 = perfect in the other direction) of each readout for separating "
                   "the run that argues against its own knowledge or cheats (positive) from its comparison group, with "
                   "95% item-bootstrap intervals. source 'new' = computed on 2026-11-04 by scripts/lens/analyze_competitors.py, "
                   "competitors_hack.py or analyze_option_lens.py; otherwise the earlier report it was copied from.",
           "scores": {"lens": "D = log p(correct letter) - log p(pushed letter B), mean over workspace layers (hack: log-prob of 27 hack words; grader '*_concept': grader words)",
                      "probe": "logistic probe trained on the two groups themselves, items held out (5-fold GroupKFold), fixed middle layer",
                      "blackbox": "the model's own next-token D at the answer position (or a forced early answer)",
                      "monitor": "keyword monitor on the chain of thought (1/0)", "length": "completion length in tokens"},
           "cases": meta, "table": table,
           "option_words": {"groups": O["groups"], "headline": O["headline"], "auroc": O["auroc"],
                            "probe_reproduction": O["probe_reproduction"], "cpu_vs_gpu_check": O["cpu_vs_gpu_check"],
                            "words": {"positive": list(O["pos_words"]), "negative": list(O["neg_words"])}}}
    out["figure"] = figure_rows(table)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
