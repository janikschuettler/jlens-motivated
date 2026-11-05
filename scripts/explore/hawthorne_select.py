"""Choose the runs and read windows for the Hawthorne lens readout (lens env for the tokenizer; CPU).

  JMR_MODEL_PROFILE=qwen3-14b uv run --project envs/lens python scripts/explore/hawthorne_select.py \
      --dir data/hawthorne/q14 --out data/hawthorne/q14/lens_select.jsonl

Rules (pre-registered in docs/overnight/preregistration.md, "Hawthorne"):
- Labels are the regex ones (2026-11-05 pre-registration, docs/preregistration.md section TA14): aware = AWARE_RE matches the reasoning; unaware = no AWARE_RE match and no BROAD_RE match.
- Realistic version only ("real" variant: no test-like names or URLs in the prompt); finished runs
  (with </think>) that the judge labelled.
- Aware = judge evaluation awareness Yes or Maybe; unaware = evaluation awareness No and
  hypothetical-scenario recognition No (labels as in scripts/explore/hawthorne_analyze.py).
- Cut of an aware run = the start of the sentence that holds the earliest awareness mention in the
  reasoning: the earliest of (a) a verbatim judge evidence quote for evaluation awareness, (b) a match
  of the keyword pattern AWARE_RE. Runs where neither is found are excluded (counted). Runs whose cut
  leaves fewer than MIN_WINDOW reasoning positions are excluded (counted).
- Prompts qualify with >= 1 located aware run and >= 1 unaware run.
- Window: from the <think> token up to the cut (aware), or up to the same number of reasoning tokens
  as the median cut of the prompt's aware runs, capped at the run's own </think> (unaware: matched
  window, so both groups are read over the same stretch of reasoning).
"""

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from jmr import config
from jmr.tokens import token_bytes

MIN_WINDOW = 16
_spec = importlib.util.spec_from_file_location("ha", Path(__file__).with_name("hawthorne_analyze.py"))
ha = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ha)
BROAD_RE = re.compile(r"\btest(s|ed|ing)?\b|hypothetical|simulat|evaluat|scenario|exercise|demonstrat|mock|fictional", re.I)
SENT_END = re.compile(r"[.!?]\s|\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    mcfg = config.load("model")
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    think_s, think_e = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    cand = defaultdict(lambda: {"aware": [], "unaware": []})
    counts = defaultdict(int)
    for l in open(d / "runs.jsonl"):
        r = json.loads(l)
        if r["variant"] != "real":
            continue
        C = r["completion_token_ids"]
        if think_e not in C:
            counts["unfinished"] += 1
            continue
        rtext = r["completion_text"][: r["completion_text"].index("</think>")]
        aware = bool(ha.AWARE_RE.search(rtext))
        unaware = (not aware) and not BROAD_RE.search(rtext)  # no keyword match and no looser test word
        if not (aware or unaware):
            counts["neither"] += 1
            continue
        if C[0] != think_s:
            raise ValueError(f"{r['run_id']}: completion doesn't start with <think> (rule 1)")
        bts = token_bytes(tok, C)
        text_b = b"".join(bts)
        if text_b.endswith(b"<|im_end|>"):  # the saved text omits the stop token
            text_b = text_b[:-len(b"<|im_end|>")]
        if text_b.decode("utf-8", errors="replace") != r["completion_text"] and \
                text_b.decode("utf-8", errors="replace").replace("�", "") != r["completion_text"].replace("�", ""):
            counts["text_mismatch"] += 1  # excluded and counted; > 2% stops the script
            continue
        starts = np.cumsum([0] + [len(b) for b in bts])  # byte offset of each token
        te = C.index(think_e)
        rec = {"run_id": r["run_id"], "question_id": r["question_id"], "P": len(r["prompt_token_ids"]),
               "te": te, "dataset": r["dataset"]}
        if unaware:
            cand[r["question_id"]]["unaware"].append(rec)
            continue
        reasoning_end_char = r["completion_text"].index("</think>")
        text = r["completion_text"][:reasoning_end_char]
        hits = []
        m = ha.AWARE_RE.search(text)
        if m:
            hits.append(m.start())
        if not hits:
            counts["aware_not_located"] += 1
            continue
        c = min(hits)
        ends = [e.end() for e in SENT_END.finditer(text, 0, c)]
        c0 = ends[-1] if ends else 0
        cb = len(text[:c0].encode("utf-8"))
        tok_idx = int(np.searchsorted(starts, cb, side="right") - 1)  # token holding the sentence start
        if tok_idx < MIN_WINDOW:
            counts["aware_cut_too_early"] += 1
            continue
        rec["cut"] = tok_idx
        rec["cut_frac"] = tok_idx / te
        cand[r["question_id"]]["aware"].append(rec)
    sel = []
    for q, g in sorted(cand.items()):
        if not g["aware"] or not g["unaware"]:
            continue
        med = int(np.median([x["cut"] for x in g["aware"]]))
        for x in g["aware"]:
            sel.append({"run_id": x["run_id"], "question_id": q, "dataset": x["dataset"], "group": "aware",
                        "lo": x["P"], "hi": x["P"] + x["cut"], "cut_tokens": x["cut"], "cut_frac": x["cut_frac"]})
        for x in g["unaware"]:
            hi = min(med, x["te"])
            sel.append({"run_id": x["run_id"], "question_id": q, "dataset": x["dataset"], "group": "unaware",
                        "lo": x["P"], "hi": x["P"] + hi, "cut_tokens": hi, "cut_frac": hi / x["te"]})
    n_seen = sum(counts.values()) + sum(len(g["aware"]) + len(g["unaware"]) for g in cand.values())
    if counts["text_mismatch"] > 0.02 * max(n_seen, 1):
        raise SystemExit(f"{counts['text_mismatch']} runs whose token bytes don't give the saved text (rule 1)")
    Path(a.out).write_text("".join(json.dumps(s) + "\n" for s in sel))
    summ = {"counts": dict(counts), "n_selected": len(sel),
            "n_prompts": len({s["question_id"] for s in sel}),
            "n_aware": sum(s["group"] == "aware" for s in sel), "n_unaware": sum(s["group"] == "unaware" for s in sel),
            "by_dataset": {ds: sum(s["dataset"] == ds for s in sel) for ds in ("actions", "triggers")},
            "median_cut_tokens_aware": float(np.median([s["cut_tokens"] for s in sel if s["group"] == "aware"])) if sel else None,
            "median_cut_frac_aware": float(np.median([s["cut_frac"] for s in sel if s["group"] == "aware"])) if sel else None}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
