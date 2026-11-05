"""probe-control test (a): choose runs, labels and windows (CPU; needs only the tokenizer).

  python3 -I? no: python scripts/explore/pc_select.py --runs data/probe_control/runs.jsonl --out data/probe_control/pc_select.jsonl

Pre-registered as PC in docs/preregistration.md. Runs: arms A_V0 and A_stakes (the 1,336 grader-bug
runs of Qwen3-14B). Identical-prompt group = (question_id, arm). Label "says testing me" = TEST_ME
(evalaware_models_analyze.py) in the thinking span (before </think>). Cut of an aware run = token
holding the start of the sentence with its first match (as hawthorne_select.py). Windows: reasoning
positions P .. P+cut (P = prompt length; completion token 0 is <think>). An unaware sibling is read
up to the largest cut of the aware runs in its group (capped at its own </think>) and carries the
list of cuts, so every aware-vs-unaware pair is scored on the same token index. Only mixed groups
(>= 1 aware run with cut >= MIN_WINDOW and >= 1 unaware run) are kept.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evalaware_models_analyze import TEST_ME  # noqa: E402
from jmr.tokens import token_bytes  # noqa: E402
import re

MIN_WINDOW = 16
SENT_END = re.compile(r"[.!?]\s|\n")
ARMS = ("A_V0", "A_stakes")
REV = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", revision=REV)
    te_id = tok.convert_tokens_to_ids("</think>")
    cnt = defaultdict(int)
    groups = defaultdict(lambda: {"aware": [], "unaware": []})
    for l in open(a.runs):
        r = json.loads(l)
        if r["condition"] not in ARMS:
            continue
        cnt["runs"] += 1
        assert r["model"]["revision"] == REV
        C = r["completion_token_ids"]
        P = len(r["prompt_token_ids"])
        te = C.index(te_id) if te_id in C else len(C)
        text = r["completion_text"].split("</think>")[0]
        hits = [m.start() for p in TEST_ME for m in p.finditer(text)]
        rec = {"run_id": r["run_id"], "question_id": r["question_id"], "arm": r["condition"], "P": P, "te": te}
        key = f'{r["question_id"]}|{r["condition"]}'
        if not hits:
            groups[key]["unaware"].append(rec)
            continue
        cnt["aware"] += 1
        bts = token_bytes(tok, C)
        text_b = b"".join(bts)
        if text_b.endswith(b"<|im_end|>"):
            text_b = text_b[: -len(b"<|im_end|>")]
        if text_b.decode("utf-8", errors="replace").replace("�", "") != r["completion_text"].replace("�", ""):
            cnt["text_mismatch"] += 1
            continue
        starts = np.cumsum([0] + [len(b) for b in bts])
        c = min(hits)
        ends = [e.end() for e in SENT_END.finditer(text, 0, c)]
        c0 = ends[-1] if ends else 0
        cb = len(text[:c0].encode("utf-8"))
        tok_idx = int(np.searchsorted(starts, cb, side="right") - 1)
        if tok_idx < MIN_WINDOW:
            cnt["aware_cut_too_early"] += 1
            continue
        rec["cut"] = tok_idx
        rec["cut_frac"] = tok_idx / max(te, 1)
        groups[key]["aware"].append(rec)
    if cnt["text_mismatch"] > 0.02 * cnt["runs"]:
        raise SystemExit(f"{cnt['text_mismatch']} text mismatches (rule 1)")
    sel, n_pairs, n_short = [], 0, 0
    for key, g in sorted(groups.items()):
        if not g["aware"] or not g["unaware"]:
            continue
        cuts = sorted(x["cut"] for x in g["aware"])
        for x in g["aware"]:
            sel.append({"run_id": x["run_id"], "question_id": key, "group": "aware", "lo": x["P"],
                        "hi": x["P"] + x["cut"], "cuts": [x["cut"]], "te": x["te"], "cut_frac": x["cut_frac"]})
        for x in g["unaware"]:
            cc = [min(c, x["te"]) for c in cuts]
            n_short += sum(c > x["te"] for c in cuts)
            sel.append({"run_id": x["run_id"], "question_id": key, "group": "unaware", "lo": x["P"],
                        "hi": x["P"] + max(cc), "cuts": cc, "src_cuts": cuts, "te": x["te"]})
        n_pairs += len(g["aware"]) * len(g["unaware"])
    Path(a.out).write_text("".join(json.dumps(s) + "\n" for s in sel))
    fr = [s["cut_frac"] for s in sel if s["group"] == "aware"]
    summ = {"counts": dict(cnt), "n_selected": len(sel), "n_groups": len({s["question_id"] for s in sel}),
            "n_questions": len({s["question_id"].split("|")[0] for s in sel}),
            "n_aware": sum(s["group"] == "aware" for s in sel), "n_unaware": sum(s["group"] == "unaware" for s in sel),
            "n_pairs": n_pairs, "pairs_where_sibling_ended_before_cut": n_short,
            "median_cut_frac_aware": float(np.median(fr)), "median_cut_tokens_aware": float(np.median([s["hi"] - s["lo"] for s in sel if s["group"] == "aware"]))}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
