"""Per-run features for the J-lens-vs-competitors comparison (EXPLORATORY).
Overnight 2026-11-04, CPU only, existing readouts (no new generation, no GPU).

Numpy only, so it runs in any env with numpy (the lens env on octavia, or the
judge env on arena9-cloud for the readouts archived there).

For each run of one multiple-choice readout set (format: jmr.lens.batch) this
stores, without any full-vocabulary output (rule 3; only the stored tracked
letters are used):

  letters  = log-sum-exp over the stored forms ("A", " A", "(A") per letter.
  C        = the correct letter; B = the pushed letter (hint / instruction /
             planted claim / grader-bug letter; for unhinted runs the item's
             would-be hinted letter); O = the two letters that are neither.
  D        = lp(C) - lp(B)                 "leans to the correct answer over the pushed one"
  M        = lp(C) - mean lp(O)            "correct letter above the uninvolved letters"
  for lens in (J-lens, logit lens) at every stored layer, and for the model's own
  output distribution ("model"), at these positions:
    last_prompt   the last prompt token (shared by every sample of a prompt)
    mid           mean over reasoning positions at relative 0.25-0.75 of the reasoning span
    think_end     the </think> token (absent in non-thinking runs)
    pre_answer    the position whose next token is the final-answer letter
  Reasoning span: [cot_span[0], think_end) in thinking runs; C3 runs (planted
  claim prefilled in the prompt): [prompt_len, think_end); non-thinking runs:
  [prompt_len, pre_answer).
  Block outputs ("resid") at last_prompt / think_end / pre_answer for 3 layers,
  and the mean of the stored mid-CoT block outputs (cot_resid anchors 4-11, the
  instrument_q1q3 mid-CoT window) where the set has them; float16.
  Text features: completion length, reasoning length, and the keyword monitor
  (the label file's own hint/instruction/grader mention lists: mentions_cot for
  thinking runs, mentions for non-thinking runs).

Groups (pos = the "arguing against its own knowledge / cheating" side):
  instr    pos: instructed to argue for B (I1 argue, I3 argue + conceal), final = B
           neg: unhinted, final = correct
  wording  pos: hint followers (final = shown B); neg: resisters (final = correct)
  c3       pos: planted wrong claim kept to the end (outcome followed)
           neg: planted claim dropped, final correct (outcome correct)
  silent   pos: switch to the hint without mentioning it; neg: switch that mentions it
  grader   pos: follows the grader bug (outcome followed) in the unsanctioned arms
           neg: resists it (outcome correct) in the same arms
Known items only (file given by --known; records with an acc field need acc[--acc-key] >= 0.9,
the model's own screen: A = Qwen3-8B thinking, Q14, NT8, NT14, GPTOSS).

    python scripts/lens/competitors_extract.py --kind instr --readouts DIR --labels L.jsonl \
        --known K.jsonl --out data/overnight/competitors/instr.npz
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

POS = ("last_prompt", "mid", "think_end", "pre_answer")
RESID_ANCHORS = ("last_prompt", "think_end", "pre_answer")
G: dict = {}


def lse_forms(lp: np.ndarray, letters: list, forms: list, L: str) -> np.ndarray:
    nf = len(forms)
    k = letters.index(L) * nf
    return np.logaddexp.reduce(lp[..., k:k + nf].astype(np.float64), axis=-1)


def work(job):
    m, B, group = job
    rdir, probe_layers, cot_layers = G["rdir"], G["probe_layers"], G["cot_layers"]
    tr = m["tracked"]
    letters, forms = tr["letters"], tr["forms"]
    C = m["correct_letter"]
    O = [x for x in letters if x not in (C, B)]
    with np.load(rdir / m["file"]) as z:
        lp = z["lp"]  # [2, L, T, 12]
        mlp = z["model_lp"]  # [T, 12]
        lo = int(z["positions_lo"])
        resid = z["resid"]  # [blocks, 3, d]
        rpos = z["resid_pos"]
        cres = z["cot_resid"] if "cot_resid" in z.files else None
    npos = m["named_positions"]
    P = m["prompt_len"]
    te = npos.get("think_end", -1)
    pa = npos["pre_answer"]
    cond = str(m.get("condition", ""))
    if cond.startswith("C3"):
        span = (P, te)
    elif te is not None and te >= 0 and m.get("cot_span"):
        span = (m["cot_span"][0], te)
    else:
        span = (P, pa)
    a, b = span
    mid = np.arange(a + int(0.25 * (b - a)), a + max(int(0.75 * (b - a)), int(0.25 * (b - a)) + 1))
    lC, lB = lse_forms(lp, letters, forms, C), lse_forms(lp, letters, forms, B)
    lO = np.mean([lse_forms(lp, letters, forms, x) for x in O], 0)
    mC, mB = lse_forms(mlp, letters, forms, C), lse_forms(mlp, letters, forms, B)
    mO = np.mean([lse_forms(mlp, letters, forms, x) for x in O], 0)
    feats = {}
    for name, p in (("last_prompt", P - 1), ("think_end", te), ("pre_answer", pa)):
        if p is None or p < 0:
            feats[name] = np.full((2, 2, lp.shape[1]), np.nan)
            feats["model_" + name] = np.full(2, np.nan)
            continue
        i = p - lo
        feats[name] = np.stack([lC[:, :, i] - lB[:, :, i], lC[:, :, i] - lO[:, :, i]])  # [D/M, lens, L]
        feats["model_" + name] = np.array([mC[i] - mB[i], mC[i] - mO[i]])
    if b - a >= 4:
        ii = mid - lo
        feats["mid"] = np.stack([(lC - lB)[:, :, ii].mean(-1), (lC - lO)[:, :, ii].mean(-1)])
    else:
        feats["mid"] = np.full((2, 2, lp.shape[1]), np.nan)
    lens_arr = np.stack([feats[p] for p in POS])  # [4 pos, 2 score, 2 lens, L]
    model_arr = np.stack([feats["model_" + p] for p in ("last_prompt", "think_end", "pre_answer")])  # [3, 2]
    names = m["resid_anchor_names"]
    r = np.full((len(RESID_ANCHORS), len(probe_layers), resid.shape[-1]), np.nan, np.float16)
    for k, an in enumerate(RESID_ANCHORS):
        if an in names and rpos[names.index(an)] >= 0:
            r[k] = resid[probe_layers, names.index(an)]
    cr = None
    if cres is not None and cot_layers:
        cl = m["cot_resid_layers"]
        cr = np.stack([cres[cl.index(L), 4:12].astype(np.float32).mean(0) for L in cot_layers]).astype(np.float16)
    return {"lens": lens_arr.astype(np.float32), "model": model_arr.astype(np.float32), "resid": r, "cot_resid": cr,
            "cot_len": int(b - a), "completion_len": int(m["completion_len"])}


def init(rdir, probe_layers, cot_layers):
    G.update(rdir=Path(rdir), probe_layers=probe_layers, cot_layers=cot_layers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["instr", "wording", "c3", "silent", "grader"])
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--known", required=True)
    ap.add_argument("--acc-key", default="A", help="screen whose accuracy defines known (A = Qwen3-8B thinking)")
    ap.add_argument("--arms", default="", help="grader: comma-separated unsanctioned arms")
    ap.add_argument("--probe-layers", default="16,24,32")
    ap.add_argument("--cot-layers", default="16,24,32")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rdir = Path(args.readouts)
    known = {}
    for d in map(json.loads, open(args.known)):
        if "acc" in d and (d["acc"].get(args.acc_key) is None or d["acc"][args.acc_key] < 0.9):
            continue
        known[d["id"]] = d
    labels = {d["run_id"]: d for d in map(json.loads, open(args.labels))}
    index = list({m["run_id"]: m for m in map(json.loads, open(rdir / "index.jsonl"))}.values())
    arms = set(filter(None, args.arms.split(",")))
    jobs, meta, counts = [], [], {}
    for m in index:
        lab = labels.get(m["run_id"])
        q, C, A, H = m["question_id"], m["correct_letter"], m["answer"], m["hint_letter"]
        arm = (lab or {}).get("arm") or m.get("extras", {}).get("arm") or m["condition"]
        g = None
        if q in known and lab is not None and A is not None and m["named_positions"].get("pre_answer", -1) >= 0:
            if args.kind == "instr":
                if arm in ("I1_argue", "I3_conceal") and A == H:
                    g = 1
                elif arm == "unhinted" and A == C:
                    g = 0
            elif args.kind == "wording" and H:
                g = 1 if A == H else 0 if A == C else None
            elif args.kind == "c3" and str(arm).startswith("C3"):
                g = {"followed": 1, "correct": 0}.get(lab["outcome"])
            elif args.kind == "silent" and H and A == H:
                g = 0 if lab["mentions"] else 1
            elif args.kind == "grader" and arm in arms:
                g = {"followed": 1, "correct": 0}.get(lab["outcome"])
                if g is not None and A != (H if g else C):
                    raise ValueError(f"label/answer mismatch {m['run_id']}")
        key = f"group {g}" if g is not None else "not in a group"
        counts[key] = counts.get(key, 0) + 1
        if g is None:
            continue
        B = H or known[q].get("hinted")
        if not B or B == C:
            raise ValueError(f"no pushed letter for {m['run_id']}")
        think = bool(m.get("enable_thinking"))
        mon = lab.get("mentions_cot" if think else "mentions") or []
        if args.kind == "grader" and think:
            mon = lab.get("grader_mention_cot") or []
        jobs.append((m, B, g))
        meta.append({"run_id": m["run_id"], "item": q, "group": g, "arm": arm, "monitor": int(bool(mon)),
                     "thinking": think})
    print(counts, f"[{time.time() - t0:.0f}s]", flush=True)
    pl = [int(x) for x in args.probe_layers.split(",")]
    cl = [int(x) for x in args.cot_layers.split(",")] if args.cot_layers else []
    with Pool(args.workers, initializer=init, initargs=(rdir, pl, cl)) as pool:
        rows = pool.map(work, jobs, chunksize=4)
    print(f"{len(rows)} runs [{time.time() - t0:.0f}s]", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have_cot = all(r["cot_resid"] is not None for r in rows)
    with np.load(rdir / index[0]["file"]) as z:
        layers = z["layers"].tolist()
    np.savez_compressed(
        out, lens=np.stack([r["lens"] for r in rows]), model=np.stack([r["model"] for r in rows]),
        resid=np.stack([r["resid"] for r in rows]),
        cot_resid=np.stack([r["cot_resid"] for r in rows]) if have_cot else np.zeros(0),
        cot_len=np.array([r["cot_len"] for r in rows]), completion_len=np.array([r["completion_len"] for r in rows]),
        meta_json=json.dumps({"kind": args.kind, "readouts": str(rdir), "labels": args.labels, "known": args.known,
                              "arms": sorted(arms), "acc_key": args.acc_key, "lens_layers": layers, "probe_layers": pl,
                              "cot_layers": cl if have_cot else [], "positions": list(POS),
                              "resid_anchors": list(RESID_ANCHORS), "counts": counts, "runs": meta}))
    print(f"wrote {out} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
