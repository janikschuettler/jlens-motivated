"""Instrument check, Q1/Q3 (lens env, CPU): can any readout see which option is
correct during the reasoning, on runs where the model knows the answer
(sensitivity) and not on runs where it doesn't (specificity)? Exploratory;
derived numbers only.

Inputs (octavia): readouts from scripts/lens/readout_runs.py --cot-resid
[--content-tokens configs/content_tokens.json] of the runs chosen by
scripts/lens/instrument_select.py (group in meta["extras"]["instr_group"]),
the screen items (option texts) and the runs files (completion token IDs, for
the CoT text baseline).

Positions (18): the 16 evenly spaced CoT anchors cot_00..cot_15 (relative
position k/15 of the reasoning span inside <think>...</think>), </think> and
pre_answer (the position whose next token is the final-answer letter).

Readouts, each giving one score per option (A-D) at each position:
  letter_{jlens,logit,model}  log-sum-exp over the option letter's forms
      " X", "X", "(X" of the lens log-probs (model = the model's own block-35
      distribution);
  content_{jlens,logit,model} log-sum-exp over the option's content-word tokens
      (configs/content_tokens.json; only items that have them);
  for both: "_bin" = the mean over the positions of the CoT bin k of 16
      (bin k = relative positions [k/16, (k+1)/16)), instead of the single
      anchor position; at </think> and pre_answer the anchor value is used;
      "_bincal" = "_bin" minus the option's mean score over the 64 stored
      prompt positions before the reasoning (removes token-frequency offsets,
      e.g. " much" vs " aspiration");
  probe       4-class multinomial logistic regression (standardised features,
      C = 0.05) on the block output at layer l and that position; label = the
      correct letter; trained on unhinted correct runs on known items only;
      5-fold GroupKFold by item over all items: every run is scored by the
      model that held out its item (honest-error items never have training
      runs);
  probe_pooled  the same, one probe per layer trained on all 16 CoT anchors;
  text_count / text_last  CoT text baseline: the option mentioned most often /
      most recently in the CoT before the position (letter patterns as in
      analyze_main, the option's text, its content word if it has one).
"Points to X" = X has the highest score (ties share the point; no mention
yet = 1/4 each). Rates are means over items of per-item means (each item
counts once), 95% CIs from 2,000 item resamples. Chance = 0.25.

Groups: unhinted (held-out train_unhinted + w2_unhinted), kb_instr (I1/I3,
final = the instructed letter B), i2_says_b, i2_knows (I2 final = correct while
arguing B), honest_error (screen A, items 2-8/10, final wrong), unstable_correct
(same items, final correct), and if read out: wording_switch/resisted,
c3_kept_b/recovered.

Fixed before looking at any result: headline window = mid-CoT anchors
cot_04..cot_11 (relative positions 0.27-0.73); per readout, the layer is the
one with the highest held-out unhinted rate in that window (chosen on the
reference group only, not on the test groups).

uv run --project envs/lens python scripts/lens/analyze_instrument.py --out data/readouts/instrument_q1q3.json
"""

import argparse
import bisect
import json
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

N_BOOT, SEED, N_COT, N_FOLDS, PROBE_C = 2000, 0, 16, 5, 0.05
LETTERS = ["A", "B", "C", "D"]
POS_NAMES = [f"cot_{k:02d}" for k in range(N_COT)] + ["think_end", "pre_answer"]
MID = list(range(4, 12))
LATE = list(range(12, 16))
READOUT_DIRS = ["readouts/instr_ct_core", "readouts/instr_nc_core", "readouts/instr_ct_later",
                "readouts/instr_nc_later"]
OPT_DIR = "readouts/instr_options"  # cot_resid_runs.py --options-only
RESID_ONLY_DIRS = ["readouts/instr_train_extra"]  # scripts/lens/cot_resid_runs.py: probe training only
RUN_FILES = ["instrument/ct_core", "instrument/nc_core", "instrument/ct_later", "instrument/nc_later"]
ITEMS = "explore/gap_factorial/screen/items.jsonl"
GROUP_MAP = {"train_unhinted": "unhinted", "w2_unhinted": "unhinted", "kb_I1": "kb_instr", "kb_I3": "kb_instr"}
TRAIN_GROUPS = {"train_unhinted", "w2_unhinted"}
REPORT_GROUPS = ["unhinted", "kb_instr", "i2_says_b", "i2_knows", "honest_error", "unstable_correct",
                 "wording_switch", "wording_resisted", "c3_kept_b", "c3_recovered"]


def mention_patterns(letter: str, option: str, word: str | None) -> list[re.Pattern]:
    e = re.escape(letter)
    pats = [rf"\({e}\)", rf"(?<![A-Za-z]){e}\)", rf"(?i:option|choice|answer|letter)\s*\(?{e}(?![A-Za-z])",
            rf"\*\*{e}(?![A-Za-z])", rf"\b{e}:\s"]
    out = [re.compile(p.encode()) for p in pats]
    opt = option.strip().rstrip(".").strip()
    if len(opt) >= 3:
        out.append(re.compile(re.escape(opt.encode()), re.IGNORECASE))
    if word:
        out.append(re.compile(rb"\b" + re.escape(word.encode()) + rb"\b", re.IGNORECASE))
    return out


# ------------------------------------------------------------ per-run pass

_TOK = None
_ITEMS = None


def _init(items):
    global _TOK, _ITEMS
    from transformers import AutoTokenizer
    from jmr import config
    m = config.load("model")
    _TOK = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    _ITEMS = items


def lse(x, axis):
    return np.logaddexp.reduce(x, axis=axis)


def extract(job):
    """One run -> scores [readout][layers, 18, 4], bin scores, residuals [10, 18, d], text counts."""
    from jmr.tokens import token_bytes
    path, meta, ids = job
    z = np.load(path)
    lo = int(z["positions_lo"])
    np_ = meta["named_positions"]
    span = meta["cot_span"]
    letters = meta["letters"]
    cpos = z["cot_resid_pos"].astype(int)
    pos = list(cpos) + [np_.get("think_end", -1), np_.get("pre_answer", -1)]
    if span is None or any(p < 0 for p in pos):
        return None
    start, end = span
    out = {"meta": meta, "pos": pos, "span": span}
    tr = meta["tracked"]
    nf = len(tr["forms"])
    lp = z["lp"].astype(np.float32)  # [2, L, T, 12]
    mlp = z["model_lp"].astype(np.float32)[None]  # [1, T, 12]
    t_idx = np.array(pos) - lo
    nb = end - start
    b = np.minimum((np.arange(nb) * N_COT) // nb, N_COT - 1)

    def letter_scores(arr):  # [S, T, 12] -> [S, T, 4]
        cols = [[tr["letters"].index(L) * nf + f for f in range(nf)] for L in letters]
        return np.stack([lse(arr[..., c], -1) for c in cols], -1)

    def bins(sc):  # sc [S, T, 4] -> [S, 16, 4] bin means over the CoT span
        x = sc[:, start - lo:end - lo]
        cnt = np.bincount(b, minlength=N_COT)
        res = np.stack([np.stack([np.bincount(b, weights=x[s, :, o], minlength=N_COT) / np.maximum(cnt, 1)
                                  for o in range(4)], -1) for s in range(x.shape[0])])
        res[:, cnt == 0] = np.nan
        return res

    sc = {}
    for name in ("letter", "content"):
        if name == "letter":
            full = {"jlens": letter_scores(lp[0]), "logit": letter_scores(lp[1]), "model": letter_scores(mlp)}
        else:
            if "x_lp" not in z.files:
                continue
            xlp = z["x_lp"].astype(np.float32)
            xm = z["x_model_lp"].astype(np.float32)[None]
            xo = z["x_opt"].astype(int)

            def cs(a):
                return np.stack([lse(a[..., xo == o], -1) for o in range(4)], -1)
            full = {"jlens": cs(xlp[0]), "logit": cs(xlp[1]), "model": cs(xm)}
        for lens, s in full.items():
            at = s[:, t_idx]  # [S, 18, 4]
            sc[f"{name}_{lens}"] = at.astype(np.float32)
            bb = bins(s)
            sc[f"{name}_{lens}_bin"] = np.concatenate([bb, at[:, N_COT:]], 1).astype(np.float32)
            # calibrated: minus the option's mean score over the stored prompt tail (before any
            # reasoning), which removes per-token frequency offsets
            base = s[:, :meta["prompt_len"] - lo].mean(1, keepdims=True)  # [S, 1, 4]
            sc[f"{name}_{lens}_bincal"] = (sc[f"{name}_{lens}_bin"] - base).astype(np.float32)
    out["scores"] = sc
    # residuals at the workspace layers: CoT anchors + think_end / pre_answer
    layers = z["layers"].tolist()
    rp = list(meta["resid_anchor_names"])
    rz = z["resid"]
    extra = np.stack([rz[layers, rp.index(a)] for a in ("think_end", "pre_answer")], 1)  # [L, 2, d]
    out["resid"] = np.concatenate([z["cot_resid"], extra], 1)  # [L, 18, d] fp16
    if not np.isfinite(out["resid"]).all():
        return None
    # text baseline
    item = _ITEMS[meta["question_id"]]
    tb = token_bytes(_TOK, ids[start:end])
    offs = np.cumsum([0] + [len(x) for x in tb])
    text = b"".join(tb)
    words = meta.get("content_tokens") or {}
    counts = np.zeros((18, 4))
    last = np.full((18, 4), 0.25)
    ends_by_opt = []
    for o, L in enumerate(letters):
        e = []
        for p in mention_patterns(L, item["options"][o], (words.get(L) or {}).get("word")):
            e += [m.end() for m in p.finditer(text)]
        # token index (absolute position) holding the mention's last byte
        ends_by_opt.append(sorted(start + bisect.bisect_left(offs, x) - 1 for x in e))
    for k, p in enumerate(pos):
        lastpos = []
        for o in range(4):
            c = [x for x in ends_by_opt[o] if x <= p]
            counts[k, o] = len(c)
            lastpos.append(max(c) if c else -1)
        if max(lastpos) >= 0:
            last[k] = (np.array(lastpos) == max(lastpos)).astype(float)
    out["text_count"] = counts
    out["text_last"] = last
    return out


def extract_resid(job):
    """Resid-only run (cot_resid_runs.py) -> meta + residuals [10, 18, d], or None."""
    path, meta = job
    z = np.load(path)
    rp = list(meta["resid_anchor_names"])
    if meta["cot_span"] is None or (z["cot_resid_pos"] < 0).any() or (z["resid_pos"] < 0).any():
        return None
    layers = z["layers"].tolist()
    extra = np.stack([z["resid"][layers, rp.index(a)] for a in ("think_end", "pre_answer")], 1)
    return {"meta": meta, "resid": np.concatenate([z["cot_resid"], extra], 1)}


# --------------------------------------------------------------- helpers


def point(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """scores [..., 4], target index [...] -> share of the point on target (ties share)."""
    s = np.nan_to_num(scores, nan=-np.inf)
    mx = s.max(-1, keepdims=True)
    top = (s == mx).astype(float)
    allnan = ~np.isfinite(mx[..., 0])
    top[allnan] = 1.0
    sh = top / top.sum(-1, keepdims=True)
    r = np.take_along_axis(sh, target[..., None], -1)[..., 0]
    r[np.isnan(scores).all(-1)] = np.nan
    return r


def boot(vals: np.ndarray, items: list, rng) -> dict:
    """vals [runs, ...] -> item-level mean and 95% CI (item resampling)."""
    keys = sorted(set(items))
    if not keys:
        return {"n_items": 0}
    groups = defaultdict(list)
    for v, q in zip(vals, items):
        groups[q].append(v)
    with np.errstate(all="ignore"):
        x = np.stack([np.nanmean(np.stack(groups[k]), 0) for k in keys])
        idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
        bs = np.nanmean(x[idx], axis=1)
        return {"n_items": len(keys), "n_runs": len(items), "mean": np.round(np.nanmean(x, 0), 4).tolist(),
                "lo": np.round(np.nanpercentile(bs, 2.5, axis=0), 4).tolist(),
                "hi": np.round(np.nanpercentile(bs, 97.5, axis=0), 4).tolist()}


def auroc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = allv.argsort().argsort().astype(float) + 1
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_boot(score, items, pos_idx, neg_idx, rng):
    pi, ni = defaultdict(list), defaultdict(list)
    for i in pos_idx:
        pi[items[i]].append(score[i])
    for i in neg_idx:
        ni[items[i]].append(score[i])
    if not pi or not ni:
        return None
    keys = sorted(set(pi) | set(ni))
    a = auroc(np.array([v for k in pi for v in pi[k]]), np.array([v for k in ni for v in ni[k]]))
    bs = []
    for _ in range(1000):
        s = rng.choice(keys, size=len(keys))
        p = np.array([v for k in s for v in pi.get(k, [])])
        n = np.array([v for k in s for v in ni.get(k, [])])
        if len(p) and len(n):
            bs.append(auroc(p, n))
    return {"auroc": round(a, 4), "lo": round(float(np.nanpercentile(bs, 2.5)), 4),
            "hi": round(float(np.nanpercentile(bs, 97.5)), 4), "n_pos": len(pos_idx), "n_neg": len(neg_idx),
            "n_items_pos": len(pi), "n_items_neg": len(ni)}


# ------------------------------------------------------------------ probe


def _fit_predict(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, Xte = args
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=PROBE_C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    P = np.zeros((len(Xte), 4))
    P[:, clf.classes_] = clf.predict_proba(sc.transform(Xte))
    return P


def run_probes(R, y, train_mask, folds, n_jobs):
    """R [runs, L, 18, d] fp16 -> held-out P [runs, L, 18, 4] (per layer x position) and
    P_pooled (one probe per layer on all 16 CoT anchors)."""
    n, nL, nP, d = R.shape
    P = np.full((n, nL, nP, 4), np.nan)
    Pp = np.full((n, nL, nP, 4), np.nan)
    jobs, where = [], []
    for f in range(N_FOLDS):
        te = np.where(folds == f)[0]
        tr = np.where((folds != f) & train_mask)[0]
        for li in range(nL):
            for k in range(nP):
                jobs.append((R[tr, li, k].astype(np.float32), y[tr], R[te, li, k].astype(np.float32)))
                where.append(("pk", te, li, k))
            Xtr = R[tr, li, :N_COT].reshape(-1, d).astype(np.float32)
            ytr = np.repeat(y[tr], N_COT)
            jobs.append((Xtr, ytr, R[te, li].reshape(-1, d).astype(np.float32)))
            where.append(("pool", te, li, None))
    with Pool(n_jobs) as pool:
        res = pool.map(_fit_predict, jobs, chunksize=1)
    for (kind, te, li, k), p in zip(where, res):
        if kind == "pk":
            P[te, li, k] = p
        else:
            Pp[te, li] = p.reshape(len(te), nP, 4)
    return P, Pp


# ----------------------------------------------------------- match probe

_G = {}  # arrays shared with forked workers (set before the Pool is created)


def _zs(x, mu, sd):
    return (x - mu) / sd


def _fit_match(job):
    """Option-matching probe: score_i = w . (z(h) * z~(o_i)), z~ = option z-scores centred over the
    item's 4 options; binary logistic on (run, option) rows, label = option is correct. kind
    "option_only": features z~(o_i) alone (the prompt's option representation, no CoT state)."""
    from sklearn.linear_model import LogisticRegression
    kind, f, lo, k = job
    H, O, y, tr_mask, folds = _G["H"], _G["O"], _G["y"], _G["train"], _G["folds"]
    tr = np.where((folds != f) & tr_mask)[0]
    te = np.where(folds == f)[0]
    Otr = O[tr, lo].astype(np.float32)  # [m, 4, d]
    mu_o, sd_o = Otr.reshape(-1, Otr.shape[-1]).mean(0), Otr.reshape(-1, Otr.shape[-1]).std(0) + 1e-3

    def feats(idx):
        o = _zs(O[idx, lo].astype(np.float32), mu_o, sd_o)
        o = o - o.mean(1, keepdims=True)
        if kind == "option_only":
            return o
        h = H[idx, lo, k].astype(np.float32)
        return _zs(h, mu_h, sd_h)[:, None, :] * o

    if kind != "option_only":
        Htr = H[tr, lo, k].astype(np.float32)
        mu_h, sd_h = Htr.mean(0), Htr.std(0) + 1e-3
    Ftr = feats(tr)
    m, d = Ftr.shape[0], Ftr.shape[-1]
    lab = np.zeros((m, 4), int)
    lab[np.arange(m), y[tr]] = 1
    sc_mu, sc_sd = Ftr.reshape(-1, d).mean(0), Ftr.reshape(-1, d).std(0) + 1e-6
    clf = LogisticRegression(C=PROBE_C, max_iter=3000).fit(((Ftr - sc_mu) / sc_sd).reshape(-1, d), lab.ravel())
    Fte = (feats(te) - sc_mu) / sc_sd
    z = clf.decision_function(Fte.reshape(-1, d)).reshape(len(te), 4)
    z = np.exp(z - z.max(1, keepdims=True))
    return kind, te, lo, k, z / z.sum(1, keepdims=True)


def run_match_probes(H, O, y, train_mask, folds, n_jobs):
    """H [runs, L_opt, 18, d], O [runs, L_opt, 4, d] -> P_match [runs, L_opt, 18, 4], P_opt [runs, L_opt, 4]."""
    _G.update(H=H, O=O, y=y, train=train_mask, folds=folds)
    n, nL = H.shape[0], H.shape[1]
    jobs = [("match", f, lo, k) for f in range(N_FOLDS) for lo in range(nL) for k in range(H.shape[2])]
    jobs += [("option_only", f, lo, None) for f in range(N_FOLDS) for lo in range(nL)]
    Pm = np.full((n, nL, H.shape[2], 4), np.nan)
    Po = np.full((n, nL, 4), np.nan)
    with Pool(n_jobs) as pool:
        for kind, te, lo, k, p in pool.imap_unordered(_fit_match, jobs):
            if kind == "match":
                Pm[te, lo, k] = p
            else:
                Po[te, lo] = p
    return Pm, Po


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--docs-out", default=None,
                    help="also write a trimmed copy for docs/experiments/instrument-q1q3/data (curves as means only)")
    args = ap.parse_args()
    D = Path(args.data)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(D / ITEMS)}

    ids_by_run = {}
    for rf in RUN_FILES:
        p = D / rf / "runs.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                ids_by_run[r["run_id"]] = r["prompt_token_ids"] + r["completion_token_ids"]
    jobs = []
    for rd in READOUT_DIRS:
        if not (D / rd / "index.jsonl").exists():
            print(f"no readouts in {rd}", flush=True)
            continue
        idx = list({m["run_id"]: m for m in map(json.loads, open(D / rd / "index.jsonl"))}.values())
        for m in idx:
            jobs.append((str(D / rd / m["file"]), m, ids_by_run[m["run_id"]]))
    print(f"{len(jobs)} runs to extract", flush=True)
    with Pool(args.jobs, initializer=_init, initargs=(items,)) as pool:
        ext = pool.map(extract, jobs, chunksize=4)
    dropped = sum(e is None for e in ext)
    ext = [e for e in ext if e is not None]
    print(f"extracted {len(ext)} ({dropped} dropped: no CoT span / anchors) in {time.time() - t0:.0f}s", flush=True)

    meta = [e["meta"] for e in ext]
    raw_group = [m["extras"]["instr_group"] for m in meta]
    group = np.array([GROUP_MAP.get(g, g) for g in raw_group])
    item = [m["question_id"] for m in meta]
    C = np.array([m["letters"].index(m["correct_letter"]) for m in meta])
    A = np.array([m["letters"].index(m["answer"]) for m in meta])
    has_content = np.array(["content_jlens" in e["scores"] for e in ext])
    cot_len = np.array([e["span"][1] - e["span"][0] for e in ext])
    train_mask = np.isin(np.array(raw_group), list(TRAIN_GROUPS))
    # sanity: training runs answer correctly
    assert (C[train_mask] == A[train_mask]).all()

    # folds by item (GroupKFold-like, seeded)
    uitems = sorted(set(item))
    perm = rng.permutation(len(uitems))
    fold_of = {q: int(perm[i] % N_FOLDS) for i, q in enumerate(uitems)}
    folds = np.array([fold_of[q] for q in item])

    # extra probe-training runs (resid only; unhinted correct on known items), not reported
    seen = {m["run_id"] for m in meta}
    xjobs = []
    for rd in RESID_ONLY_DIRS:
        if (D / rd / "index.jsonl").exists():
            for m in {m["run_id"]: m for m in map(json.loads, open(D / rd / "index.jsonl"))}.values():
                if m["run_id"] not in seen and m["extras"]["instr_group"] == "train_extra":
                    xjobs.append((str(D / rd / m["file"]), m))
    with Pool(args.jobs) as pool:
        xext = [e for e in pool.map(extract_resid, xjobs, chunksize=8) if e is not None]
    for e in xext:
        m = e["meta"]
        if m["answer"] != m["correct_letter"] or m["question_id"] not in fold_of:
            raise ValueError(f"extra training run {m['run_id']}: not correct or item outside the folds")
    print(f"{len(xext)} extra resid-only training runs", flush=True)
    R = np.stack([e["resid"] for e in ext] + [e["resid"] for e in xext])  # [n + x, L, 18, d] fp16
    Cx = np.concatenate([C, [m["letters"].index(m["correct_letter"]) for m in (e["meta"] for e in xext)]]).astype(int)
    tmx = np.concatenate([train_mask, np.ones(len(xext), bool)])
    fx = np.concatenate([folds, [fold_of[e["meta"]["question_id"]] for e in xext]]).astype(int)
    layers = meta[0]["cot_resid_layers"]
    t1 = time.time()
    P, Pp = run_probes(R, Cx, tmx, fx, args.jobs)
    P, Pp = P[:len(ext)], Pp[:len(ext)]
    print(f"probes done in {time.time() - t1:.0f}s", flush=True)
    res_train = {"n_train_runs_full": int(train_mask.sum()), "n_train_runs_extra": len(xext)}
    # option-matching probe (needs the option-span block outputs, cot_resid_runs.py --options-only)
    Pm = Po = None
    if (D / OPT_DIR / "index.jsonl").exists():
        oidx = {m["run_id"]: m for m in map(json.loads, open(D / OPT_DIR / "index.jsonl"))}
        all_meta = meta + [e["meta"] for e in xext]
        have = np.array([m["run_id"] in oidx for m in all_meta])
        res_train["n_runs_with_option_resid"] = int(have.sum())
        if have[:len(ext)].all():
            z0 = np.load(D / OPT_DIR / oidx[all_meta[0]["run_id"]]["file"])
            opt_layers = z0["layers"].tolist()
            O = np.zeros((len(all_meta), len(opt_layers), 4, R.shape[-1]), np.float16)
            for j, m in enumerate(all_meta):
                if have[j]:
                    O[j] = np.load(D / OPT_DIR / oidx[m["run_id"]]["file"])["opt_resid"]
            sel_l = [layers.index(l) for l in opt_layers]
            keep = np.where(have)[0]
            t2 = time.time()
            pm, po = run_match_probes(R[keep][:, sel_l], O[keep], Cx[keep], tmx[keep], fx[keep], args.jobs)
            Pm, Po = pm[:len(ext)], po[:len(ext)]
            res_train["option_layers"] = opt_layers
            print(f"match probes done in {time.time() - t2:.0f}s", flush=True)
        else:
            print("WARNING: option residuals missing for some reported runs; match probe skipped", flush=True)
    del R

    # scores per readout: [n, S, 18, 4]
    readouts = {}
    for key in ext[0]["scores"]:
        if key.startswith("letter"):
            readouts[key] = np.stack([e["scores"][key] for e in ext])
    ck = [k for k in ext[np.argmax(has_content)]["scores"] if k.startswith("content")] if has_content.any() else []
    for key in ck:
        shape = next(e["scores"][key].shape for e in ext if key in e["scores"])
        readouts[key] = np.stack([e["scores"].get(key, np.full(shape, np.nan, np.float32)) for e in ext])
    readouts["probe"] = P
    readouts["probe_pooled"] = Pp
    if Pm is not None:
        readouts["match_probe"] = Pm
        readouts["option_only"] = np.repeat(Po[:, :, None], 18, 2)
    readouts["text_count"] = np.stack([e["text_count"] for e in ext])[:, None]
    readouts["text_last"] = np.stack([e["text_last"] for e in ext])[:, None]

    res = {"layers": layers, "positions": POS_NAMES, "mid_window": [POS_NAMES[k] for k in MID],
           "late_window": [POS_NAMES[k] for k in LATE], "n_runs": len(ext), "n_dropped": dropped,
           "groups": {}, "readouts": {}, "headline": {}, "auroc": {}, "length": {}, "probe_training": res_train}
    for g in REPORT_GROUPS:
        sel = np.where(group == g)[0]
        if len(sel):
            res["groups"][g] = {"n_runs": int(len(sel)), "n_items": len(set(item[i] for i in sel)),
                                "n_runs_content": int(has_content[sel].sum()),
                                "n_items_content": len(set(item[i] for i in sel if has_content[i])),
                                "cot_len_median": float(np.median(cot_len[sel])),
                                "cot_len_iqr": [float(np.percentile(cot_len[sel], 25)),
                                                float(np.percentile(cot_len[sel], 75))]}

    subsets = {"all": np.ones(len(ext), bool), "content_items": has_content}
    for rname, sc in readouts.items():
        pc = point(sc, np.broadcast_to(C[:, None, None], sc.shape[:3]).copy())  # [n, S, 18]
        pa = point(sc, np.broadcast_to(A[:, None, None], sc.shape[:3]).copy())
        res["readouts"][rname] = {}
        for sub, smask in subsets.items():
            if rname.startswith("content") and sub == "all":
                continue
            out = {}
            for g in REPORT_GROUPS:
                sel = np.where((group == g) & smask & np.isfinite(pc[:, 0, 0]))[0]
                if not len(sel):
                    continue
                its = [item[i] for i in sel]
                out[g] = {"point_correct": boot(pc[sel], its, rng), "point_answer": boot(pa[sel], its, rng)}
                mid = np.nanmean(pc[sel][:, :, MID], -1)  # [runs, S]
                out[g]["mid_point_correct"] = boot(mid, its, rng)
                out[g]["late_point_correct"] = boot(np.nanmean(pc[sel][:, :, LATE], -1), its, rng)
                out[g]["mid_point_answer"] = boot(np.nanmean(pa[sel][:, :, MID], -1), its, rng)
            res["readouts"][rname][sub] = out
            # headline: layer with the best unhinted mid-CoT rate
            if "unhinted" in out:
                u = np.array(out["unhinted"]["mid_point_correct"]["mean"])
                li = int(np.nanargmax(u))
                lay = (res_train.get("option_layers") if rname in ("match_probe", "option_only")
                       else layers if sc.shape[1] == len(layers) else None)
                h = {"layer_index": li, "layer": lay[li] if lay else ("model" if "model" in rname else None)}
                for g, o in out.items():
                    h[g] = {w: {"mean": o[w]["mean"][li], "lo": o[w]["lo"][li], "hi": o[w]["hi"][li],
                                "n_items": o[w]["n_items"], "n_runs": o[w]["n_runs"]}
                            for w in ("mid_point_correct", "late_point_correct", "mid_point_answer")}
                    for k in (N_COT, N_COT + 1):
                        pcg = o["point_correct"]
                        h[g][POS_NAMES[k]] = {"mean": pcg["mean"][li][k], "lo": pcg["lo"][li][k],
                                              "hi": pcg["hi"][li][k]}
                res["headline"].setdefault(rname, {})[sub] = h
                # AUROC: score margin (correct minus mean other) mid-CoT, kb_instr vs honest_error,
                # i2_knows vs honest_error, kb_instr vs unhinted
                if rname.startswith("text"):
                    continue
                s = sc[:, li] if sc.shape[1] > 1 else sc[:, 0]
                if rname.startswith("probe"):
                    marg = np.log(np.take_along_axis(s, C[:, None, None], -1)[..., 0] + 1e-9)
                else:
                    corr = np.take_along_axis(s, C[:, None, None], -1)[..., 0]
                    marg = corr - (np.nansum(s, -1) - corr) / 3
                mm = np.nanmean(marg[:, MID], -1)
                for a_, b_ in (("kb_instr", "honest_error"), ("i2_knows", "honest_error"),
                               ("unhinted", "honest_error"), ("kb_instr", "unhinted")):
                    pi = np.where((group == a_) & smask)[0]
                    ni = np.where((group == b_) & smask)[0]
                    r_ = auroc_boot(mm, item, pi, ni, rng)
                    if r_:
                        res["auroc"].setdefault(rname, {}).setdefault(sub, {})[f"{a_}_vs_{b_}"] = r_

    # length control: probe (best layer) mid-CoT point_correct within global CoT-length tertiles
    cuts = np.percentile(cot_len, [100 / 3, 200 / 3])
    tert = np.digitize(cot_len, cuts)
    for rname in ("probe", "probe_pooled", "letter_jlens_bin", "text_last"):
        sc = readouts[rname]
        h = res["headline"][rname]["all"]
        li = h["layer_index"] if sc.shape[1] > 1 else 0
        pc = point(sc[:, li], np.broadcast_to(C[:, None], sc.shape[0:1] + (18,)).copy())
        mid = np.nanmean(pc[:, MID], -1)
        res["length"][rname] = {"cuts": cuts.tolist()}
        for g in REPORT_GROUPS:
            for t in range(3):
                sel = np.where((group == g) & (tert == t))[0]
                if len(sel):
                    res["length"][rname][f"{g}/t{t}"] = boot(mid[sel], [item[i] for i in sel], rng)
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"→ {args.out} ({res['elapsed_s']}s)")
    if args.docs_out:
        trim = {k: v for k, v in res.items() if k != "readouts"}
        trim["curves"] = {}
        for rname, subs in res["readouts"].items():
            for sub, out in subs.items():
                trim["curves"].setdefault(rname, {})[sub] = {
                    g: {"point_correct_mean": o["point_correct"]["mean"],
                        "point_answer_mean": o["point_answer"]["mean"],
                        "n_items": o["point_correct"]["n_items"], "n_runs": o["point_correct"]["n_runs"]}
                    for g, o in out.items()}
        trim["note"] = ("Derived numbers of scripts/lens/analyze_instrument.py; curves are item-level means "
                        "[layer][position] (positions: cot_00..cot_15, think_end, pre_answer); full CIs per "
                        "layer x position are in the octavia copy data/readouts/instrument_q1q3.json.")
        Path(args.docs_out).write_text(json.dumps(trim, separators=(",", ":")))
        print(f"→ {args.docs_out}")


if __name__ == "__main__":
    main()
