"""Analysis of the steering test (CPU; numpy + jmr only, either env).

  uv run --project envs/lens python scripts/lens/analyze_steer.py --answer data/steer/answer \
      --gen data/steer/gen --out data/steer/analysis.json --docs-out docs/overnight/data/steering.json

Part 1 (steering at the answer position): per group x layer x direction x alpha, the
item-level share of runs whose answer letter moves the intended way, and the
pre-registered summary (mean over the alpha grid, J-lens minus each other direction,
paired item bootstrap). Part 2 (steering during generation): final answer (natural, or
forced at the token cap), chain-of-thought measures and side effects, per group x
direction x alpha, and the pre-registered contrasts.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

from jmr import config, steer
from jmr.explore.w2_labels import REFUSAL_RE, asserted_letters, correct_option_mentions, w2_label
from jmr.records import RunRecord

DIRS = steer.DIR_TYPES


def softmax4(lp):
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    s = sum(e)
    return [x / s for x in e]


# ------------------------------------------------------------------ part 1

def part1(path: Path) -> dict:
    V = [tuple(v) for v in json.loads((path / "variants.json").read_text())]
    rows = [json.loads(l) for l in open(path / "results.jsonl")]
    L = steer.LETTERS
    recs = []  # one per run x variant
    base_ok = []
    for r in rows:
        c, b = L.index(r["correct"]), (L.index(r["hinted"]) if r["hinted"] else None)
        base = r["res"][0]
        base_ok.append(L[max(range(4), key=lambda k: base[k])] == r["answer"])
        for (layer, t, primary, alpha), res in zip(V, r["res"]):
            if t == "none":
                continue
            lp, top = res[:4], res[4]
            am = max(range(4), key=lambda k: lp[k])
            g = r["group"]
            if primary:
                moved = (am == c) if g != "unhinted" else (am != c)
            else:
                moved = (am != b) if g != "unhinted" else (am == b)
            p = softmax4(lp)
            recs.append({"question_id": r["question_id"], "group": g, "layer": layer, "dir": t, "primary": primary,
                         "alpha": alpha, "moved": float(moved), "p_correct": p[c],
                         "p_B": p[b] if b is not None else float("nan"),
                         "letter_mass": sum(math.exp(x) for x in lp), "top_is_letter": float(sum(math.exp(x) for x in lp) > 0.5)})
    groups = sorted({r["group"] for r in rows})
    out = {"n_runs": {g: sum(1 for r in rows if r["group"] == g) for g in groups},
           "n_items": {g: len({r["question_id"] for r in rows if r["group"] == g}) for g in groups},
           "unsteered_argmax_is_answer": sum(base_ok) / len(base_ok),
           "median_abs_tf_vllm": sorted(abs(r["tf_vs_vllm_answer_lp"]) for r in rows)[len(rows) // 2],
           "curves": {}, "summary": {}}
    by = defaultdict(list)
    for x in recs:
        by[(x["group"], x["layer"], x["dir"], x["primary"], x["alpha"])].append(x)
    pooled_name = "followed_pooled"
    for x in recs:
        if x["group"] in steer.FOLLOWED_GROUPS:
            by[(pooled_name, x["layer"], x["dir"], x["primary"], x["alpha"])].append(x)
    for (g, l, t, pr, a), xs in sorted(by.items(), key=lambda kv: str(kv[0])):
        key = f"{g}|L{l}|{t}|{'primary' if pr else 'secondary'}"
        cur = out["curves"].setdefault(key, [])
        mv = steer.boot_mean(steer.item_means(xs, lambda r: r["moved"]))
        cur.append({"alpha": a, "moved": mv["mean"], "lo": mv["lo"], "hi": mv["hi"], "n_items": mv["n_items"],
                    "p_correct": _mean(xs, "p_correct"), "p_B": _mean(xs, "p_B"),
                    "letter_mass": _mean(xs, "letter_mass"), "top_is_letter": _mean(xs, "top_is_letter")})
    # Pre-registered summary: mean of `moved` over the alpha grid, per item; J-lens minus others.
    for g in groups + [pooled_name]:
        for l in steer.ANSWER_LAYERS:
            for pr in (True, False):
                auc = {}
                for t in DIRS:
                    xs = [x for (gg, ll, tt, pp, a), v in by.items() if gg == g and ll == l and tt == t and pp == pr
                          for x in v]
                    auc[t] = steer.item_means(xs, lambda r: r["moved"])
                key = f"{g}|L{l}|{'primary' if pr else 'secondary'}"
                out["summary"][key] = {"mean_moved": {t: steer.boot_mean(auc[t]) for t in DIRS},
                                       "jlens_minus": {t: steer.boot_diff(auc["jlens"], auc[t]) for t in DIRS
                                                       if t != "jlens"}}
    return out


def part1_other(path: Path) -> dict:
    """Exploratory: pushing toward the other unchosen wrong letter (block 24). 'moved' =
    the answer letter is now that letter. Compare with the primary curves (toward the
    correct letter) of the same runs."""
    V = [tuple(v) for v in json.loads((path / "variants_other.json").read_text())]
    rows = [json.loads(l) for l in open(path / "results_other.jsonl")]
    L = steer.LETTERS
    by = defaultdict(list)
    for r in rows:
        o = L.index(r["other"])
        for (layer, t, _, alpha), res in zip(V, r["res"]):
            if t == "none":
                continue
            am = max(range(4), key=lambda k: res[k])
            x = {"question_id": r["question_id"], "moved": float(am == o),
                 "letter_mass": sum(math.exp(v) for v in res[:4])}
            by[(r["group"], t, alpha)].append(x)
            if r["group"] in steer.FOLLOWED_GROUPS:
                by[("followed_pooled", t, alpha)].append(x)
    out = {"curves": {}, "summary": {}}
    for (g, t, a), xs in sorted(by.items(), key=lambda kv: str(kv[0])):
        mv = steer.boot_mean(steer.item_means(xs, lambda r: r["moved"]))
        out["curves"].setdefault(f"{g}|L24|{t}|other", []).append(
            {"alpha": a, "moved": mv["mean"], "lo": mv["lo"], "hi": mv["hi"], "n_items": mv["n_items"],
             "letter_mass": _mean(xs, "letter_mass")})
    for g in sorted({k[0] for k in by}):
        out["summary"][g] = {t: steer.boot_mean(steer.item_means(
            [x for (gg, tt, a), v in by.items() if gg == g and tt == t for x in v], lambda r: r["moved"]))
            for t in ("jlens", "logit", "probe")}
    return out


def _mean(xs, k):
    v = [x[k] for x in xs if not (isinstance(x[k], float) and math.isnan(x[k]))]
    return sum(v) / len(v) if v else None


# ------------------------------------------------------------------ part 2

def label_gen(rows: list[dict], options: dict[str, dict]) -> list[dict]:
    names = {f.name for f in fields(RunRecord)}
    tr = config.load("lens")["tracked_tokens"]["ids"]
    out = []
    for row in rows:
        rec = RunRecord(**{k: v for k, v in row.items() if k in names})
        lab = w2_label(row, rec)
        j = row["steer"]
        item = options[rec.question_id]
        ctext = item["options"][item["letters"].index(rec.correct_letter)]
        w2 = row.get("w2", {})
        prefill = w2.get("prefill_text", "").replace("<think>", "", 1) if w2.get("prefill_starts_thinking") else ""
        text = prefill + rec.completion_text
        cot = text.split("</think>")[0].replace("<think>", "", 1)
        natural = lab["answer"]
        forced = (j.get("forced") or {}).get("letter")
        final = natural if natural is not None else (forced if rec.finish_reason == "length" else None)
        steered_letter = j["letter"]
        sl_ids = [tr[steered_letter][f] for f in ("space", "bare", "paren")] if steered_letter else []
        cm = correct_option_mentions(cot, rec.correct_letter, ctext)
        out.append({
            "job_id": row["job_id"], "group": j["group"], "dir": j["dir"], "alpha": j["alpha"], "sample": j["sample"],
            "question_id": rec.question_id, "correct": rec.correct_letter, "hinted": rec.hint_letter,
            "natural": natural, "forced": forced, "final": final, "finish": rec.finish_reason,
            "outcome": steer.answer_outcome(final, rec.correct_letter, rec.hint_letter),
            # truncated runs count as "no answer" here (orchestrator's condition, 2026-11-04 03:25)
            "outcome_natural": steer.answer_outcome(natural if rec.finish_reason == "stop" else None,
                                                    rec.correct_letter, rec.hint_letter),
            "n_tokens": len(rec.completion_token_ids),
            "cot_mentions_correct": float(not cm["cot_silent"]),
            "cot_asserts_correct": float(rec.correct_letter in asserted_letters(cot, rec.letters)),
            "rep8": steer.repeated_ngram_share(rec.completion_token_ids, 8),
            "steered_letter_per_1k": steer.letter_rate(rec.completion_token_ids, sl_ids) if sl_ids else None,
            "refusal": float(any(re.search(p, rec.completion_text) for p in REFUSAL_RE)),
            "non_ascii": steer.non_ascii_share(rec.completion_text),
        })
    return out


MEASURES = ["final_correct", "final_B", "no_answer", "natural_correct", "natural_B", "truncated", "natural_finish", "cot_mentions_correct", "cot_asserts_correct",
            "degenerate", "rep8", "refusal", "non_ascii", "n_tokens"]


def measure(r, k):
    if k == "final_correct":
        return float(r["outcome"] == "correct")
    if k == "final_B":
        return float(r["outcome"] == "B") if r["hinted"] else None
    if k == "natural_correct":
        return float(r["outcome_natural"] == "correct")
    if k == "natural_B":
        return float(r["outcome_natural"] == "B") if r["hinted"] else None
    if k == "truncated":
        return float(r["finish"] == "length")
    if k == "no_answer":
        return float(r["outcome"] == "none")
    if k == "natural_finish":
        return float(r["finish"] == "stop" and r["natural"] is not None)
    if k == "degenerate":
        return float(r["rep8"] > 0.5)
    return r[k]


def part2(rows: list[dict], name: str) -> dict:
    out = {"cells": {}, "contrasts": {}}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["group"], r["dir"], r["alpha"])].append(r)
        if r["group"] in steer.FOLLOWED_GROUPS:
            cells[("followed_pooled", r["dir"], r["alpha"])].append(r)
    for (g, t, a), xs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        out["cells"][f"{g}|{t}|{a:g}"] = {"n_runs": len(xs), **{k: steer.boot_mean(steer.item_means(
            xs, lambda r, k=k: measure(r, k))) for k in MEASURES}}
    alphas = sorted({r["alpha"] for r in rows if r["alpha"] > 0})
    groups = sorted({r["group"] for r in rows}) + ["followed_pooled"]
    for g in groups:
        for m in ("final_correct", "natural_correct", "natural_B", "truncated"):
            base = steer.item_means(cells.get((g, "none", 0.0), []), lambda r, m=m: measure(r, m))
            for a in alphas:
                per = {t: steer.item_means(cells.get((g, t, a), []), lambda r, m=m: measure(r, m)) for t in DIRS}
                key = f"{g}|{a:g}" + ("" if m == "final_correct" else f"|{m}")
                out["contrasts"][key] = {
                    "vs_unsteered": {t: steer.boot_diff(per[t], base) for t in DIRS},
                    "jlens_minus": {t: steer.boot_diff(per["jlens"], per[t]) for t in DIRS if t != "jlens"},
                }
    out["alphas"] = alphas
    return out


def originals(cap: int = 2048) -> dict:
    """The same 40 questions' original runs (vLLM, 8,192-token cap) per group's arm: share
    correct, share B, share longer than `cap` tokens, and share correct if runs longer
    than the cap had counted as no answer (what the cap alone would do)."""
    items = {json.loads(l)["id"] for l in open(steer.ITEMS_FILE)}
    out = {}
    for g, arm in steer.GEN_ARMS.items():
        d = Path(steer.SOURCES[g][0])
        lab = {json.loads(l)["run_id"]: json.loads(l) for l in open(d / "labels.jsonl")}
        rows = []
        for line in open(d / "runs.jsonl"):
            r = json.loads(line)
            if r.get("arm") != arm or r["question_id"] not in items:
                continue
            lb = lab[r["run_id"]]
            n = len(r["completion_token_ids"])
            ok = lb.get("answer") == r["correct_letter"]
            isb = r["hint_letter"] is not None and lb.get("answer") == r["hint_letter"]
            rows.append({"question_id": r["question_id"], "correct": float(ok), "B": float(isb),
                         "over_cap": float(n > cap), "correct_capped": float(ok and n <= cap),
                         "B_capped": float(isb and n <= cap)})
        out[g] = {"n_runs": len(rows), **{k: steer.boot_mean(steer.item_means(rows, lambda r, k=k: r[k]))
                                          for k in ("correct", "B", "over_cap", "correct_capped", "B_capped")}}
    return out


def headline(res: dict) -> dict:
    """Compact numbers for the report, the registry and the figure."""
    h = {}
    if "part1" in res:
        P = res["part1"]["summary"]
        h["part1"] = {}
        for key, v in P.items():
            g, l, which = key.split("|")
            h["part1"].setdefault(which, {}).setdefault(g, {})[l] = {
                "mean_moved": v["mean_moved"], "jlens_minus": v["jlens_minus"]}
    if "part2_main" in res:
        C = res["part2_main"]["cells"]
        h["part2"] = {"alphas": res["part2_main"]["alphas"], "cells": {}, "contrasts": res["part2_main"]["contrasts"]}
        for key, v in C.items():
            g, t, a = key.split("|")
            h["part2"]["cells"].setdefault(g, {}).setdefault(a, {})[t] = {
                k: v[k] for k in ("final_correct", "final_B", "no_answer", "natural_correct", "natural_B", "truncated",
                                  "natural_finish", "cot_asserts_correct",
                                  "cot_mentions_correct", "degenerate", "n_tokens")} | {"n_runs": v["n_runs"]}
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", default="data/steer/answer")
    ap.add_argument("--gen", default="data/steer/gen")
    ap.add_argument("--out", default="data/steer/analysis.json")
    ap.add_argument("--docs-out", default=None)
    a = ap.parse_args()
    res = {"model": config.load("model")["revision"]}
    meta_p = Path("data/steer/directions_meta.json")
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        res["directions"] = {k: meta[k] for k in ("r", "cos", "read_acc", "probe_cv_item_acc", "n_probe_runs",
                                                  "n_probe_items", "norm_convention")}
    if (Path(a.answer) / "results.jsonl").exists():
        res["part1"] = part1(Path(a.answer))
    if (Path(a.answer) / "results_other.jsonl").exists():
        res["part1_other"] = part1_other(Path(a.answer))
    options = {}
    for l in open(steer.POOL_FILE):
        r = json.loads(l)
        options[r["id"]] = r
    g = Path(a.gen)
    for name in ("pilot", "main"):
        p = g / f"runs_{name}.jsonl"
        if p.exists():
            rows = label_gen([json.loads(l) for l in open(p)], options)
            res[f"part2_{name}"] = part2(rows, name)
            Path(a.out).with_name(f"labels_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            if name == "pilot":
                pil = {}
                for al in steer.GEN_PILOT_ALPHAS:
                    xs = [r for r in rows if r["alpha"] == al]
                    if xs:
                        pil[al] = {"accuracy": sum(r["outcome"] == "correct" for r in xs) / len(xs),
                                   "finished": 1 - sum(r["rep8"] > 0.5 for r in xs) / len(xs),
                                   "natural_finish": sum(r["natural"] is not None for r in xs) / len(xs), "n": len(xs)}
                res["pilot_rule"] = {"per_alpha": pil, "alpha_lo_hi": steer.pilot_alpha_rule(pil) if pil else None}
    res["originals_8192"] = originals(2048)
    res["headline"] = headline(res)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print("wrote", a.out)
    if a.docs_out:
        Path(a.docs_out).write_text(json.dumps(res, indent=1, default=str))
        print("wrote", a.docs_out)


if __name__ == "__main__":
    main()
