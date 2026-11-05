"""Subtle cues x model (OpenRouter), reusing or_grader machinery and wcm_run's item selection.
Cue texts are read verbatim from configs/explore/subtle_cues.yaml and built with run_subtle.user_text.
Few-shot examples: 4 known items from qwen8b_known_items.jsonl beyond the first 60 target items
(shortest, distinct subjects and correct letters, no option referring to other options), same rule as subtle-cues.
  zsh -ic 'uv run --project envs/judge python scripts/explore/scm_run.py --model gemma-3-27b-it --limit-items 3 --samples 1'
Existing nohint / W_answer runs of wording-cross-model are copied in first (same run ids), so only new ones are generated.
"""
import argparse, json, re, shutil, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import or_grader as og
from jmr.prompts import Question

REPO = og.REPO
og.CFG = yaml.safe_load(open(REPO / "configs/explore/subtle_cross_model.yaml"))
og.OUT = REPO / og.CFG["out_dir"]
SPEC = yaml.safe_load(open(REPO / og.CFG["cue_spec"]))
SRC = REPO / "data/models_grader"
OLD = REPO / "data/wording_cross_model"

def Q(r):
    return Question(r["id"], r["dataset"], r["question"], tuple(r["options"]), tuple(r["letters"]), r["correct"], r.get("subject", ""))

def examples():
    rows = [json.loads(l) for l in open(REPO / og.CFG["items"])]
    rows = [r for r in rows if r["acc"]["A"] >= 0.9][og.CFG["n_items"]:]
    ex, subj, letters = [], set(), set()
    for q in sorted(map(Q, rows), key=lambda q: (len(q.question) + sum(len(o) for o in q.options), q.id)):
        if q.subject in subj or q.correct in letters: continue
        if any(re.search(r"\b(?:above|both|none|all of|neither)\b", o, re.I) for o in q.options): continue
        ex.append(q); subj.add(q.subject); letters.add(q.correct)
        if len(ex) == og.CFG["n_examples"]: break
    return ex
EX = examples()

def user_text(cue, q, b):
    """Same as scripts/explore/run_subtle.py user_text (copied logic, texts from the same yaml)."""
    s = SPEC
    def ot(letters, options, marked=None):
        return "\n".join(s["option"].format(letter=x, text=(s["marker"] if x == marked else "") + t) for x, t in zip(letters, options))
    def reorder(e, target):
        o = list(e.options); i, j = e.letters.index(e.correct), e.letters.index(target); o[i], o[j] = o[j], o[i]; return tuple(o), target
    import hashlib
    prefix, parts = "", []
    if cue in ("fewshot_always_b", "fewshot_marker"):
        blocks = []
        for e in EX:
            opts, ans = reorder(e, b) if cue == "fewshot_always_b" else (e.options, e.correct)
            blocks.append(s["example"].format(question_block=s["question_block"].format(question=e.question.strip(), options=ot(e.letters, opts, ans if cue == "fewshot_marker" else None)), answer=ans))
        parts.append(s["example_sep"].join(blocks) + s["example_sep"])
    if cue == "metadata":
        qid = f"q{int(hashlib.sha256(q.id.encode()).hexdigest()[:8], 16) % 100000:05d}"
        prefix = s["metadata"].format(qid=qid, letter=b)
    elif cue == "grader_code":
        prefix = s["grader"].format(letter=b)
    marked = b if cue == "fewshot_marker" else None
    tgt = s["question_block"].format(question=q.question, options=ot(q.letters, q.options, marked))
    return prefix + "".join(parts) + tgt + "\n\n" + s["instruction"]

_orig = og.build_messages
def build_messages(item, arm):
    if arm in ("metadata", "grader_code", "fewshot_marker", "fewshot_always_b"):
        return [{"role": "user", "content": user_text(arm, Q(item), item["hinted"])}], item["hinted"]
    return _orig(item, arm)
og.build_messages = build_messages

def known(model, n):
    runs = {}
    for l in open(SRC / model / "runs.jsonl"):
        r = json.loads(l)
        if r["arm"] == "unhinted": runs[r["run_id"]] = r
    out = []
    for it in og.load_items():
        rs = [runs.get(f"{model}/unhinted/{it['id']}/{s}") for s in range(5)]
        if all(r and r["answer"] == it["correct"] for r in rs): out.append(it)
    return out[:n]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit-items", type=int, default=40)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    items = known(a.model, a.limit_items)
    arms = [x for x in a.arms.split(",") if x] or list(og.CFG["arms"])
    if a.dry:
        print([(e.id, e.correct) for e in EX], len(items))
        for arm in arms: print("=====", arm); print(build_messages(items[0], arm)[0][-1]["content"])
        sys.exit(0)
    dst = og.OUT / a.model / "runs.jsonl"
    if not dst.exists() and (OLD / a.model / "runs.jsonl").exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(OLD / a.model / "runs.jsonl", dst)
    jobs = [(it, arm, s) for s in range(a.samples) for arm in arms for it in items]
    sys.exit(og.run_batch(a.model, jobs, f"scm {len(items)} items x{a.samples} {','.join(arms)}", a.workers))
