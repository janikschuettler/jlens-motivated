"""J-lens and logit lens at the prompt's option words (lens env, CPU only).

Question: the option-text probe of analyze_instrument.py (instrument_q1q3.md)
reads the correct option from the prompt's representation of each option (the
mean block output over the option's text, layers 16/20/24/28, stored by
cot_resid_runs.py --options-only in data/readouts/instr_options). Does a fixed
vocabulary readout of the *same* vectors, the J-lens or the logit lens, see
it too?

Inputs are the stored span means, so lens and probe see identical vectors. A
span mean is not a real position: the lens readout of a mean of residuals is
not the readout at any token. Read it as "the lens applied to the probe's
input".

Readouts (fixed before looking at any lens result):
  J-lens       unembed(h @ J[l].T), unembed = final RMSNorm + LM head (bf16), as
               jmr.lens.readout (jlens transport + HFLensModel.unembed).
  logit lens   unembed(h).
  Full softmax in memory only; stored: tracked-token log-probs and ranks, and
  the top-25 per (run, lens, layer, option) in a separate npz (CLAUDE.md rule 3).

Option scores (fixed before looking):
  (a) "correctness words": per option i, log-sum-exp of the lens log-probs at
      option i's span of POS_WORDS minus the same for NEG_WORDS; the option
      with the highest score is the one the readout points to.
  (b) "answer letters": per letter X, the mean over the 4 option spans of the
      lens log-prob of X (log-sum-exp over " X", "X", "(X"); argmax letter.
  Every word must be a single token (checked; the run stops otherwise).

Metrics, as analyze_instrument.py: pointing rate = share of the point on the
correct option (ties share), item-level means, 2,000 item resamples; AUROC of
the correct-option margin (score of correct minus mean of the other three)
for kb_instr / i2_knows / unhinted vs honest_error, and kb_instr vs unhinted
(1,000 item resamples). Headline layer per readout = best unhinted pointing
rate (chosen on the unhinted group only).

Check first: the option-text probe is refitted with analyze_instrument.py's
exact data, folds and settings; its unhinted pointing rate must reproduce the
stored result (data/readouts/instrument_q1q3.json). The CPU unembed + J is
checked against stored GPU readouts (tracked log-probs and top-25 at the
stored block-output anchors of a few instr_ct_core runs).

Diagnostic: the option-text probe refitted on all training runs; its weight
direction mapped back to raw block-output space (undoing both standardisations;
the per-item centring makes it a direction in h) and pushed through the
readouts' linear part, W_U diag(g) J[l] d (J-lens) and W_U diag(g) d (logit),
g = final-norm gain. Top/bottom 15 tokens.

CUDA_VISIBLE_DEVICES="" uv run --project envs/lens python scripts/lens/analyze_option_lens.py \
    --out data/readouts/option_lens.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import analyze_instrument as AI  # noqa: E402  (same folder; reuse groups, folds, probe, bootstrap)

POS_WORDS = [" correct", " right", " true", " yes", " Correct", " True", "正确"]
NEG_WORDS = [" wrong", " incorrect", " false", " no", " Wrong", " False", "错误"]
LETTER_IDS = {"A": [32, 362, 4346], "B": [33, 425, 5349], "C": [34, 356, 3025], "D": [35, 422, 5432]}
CONTRASTS = (("kb_instr", "honest_error"), ("i2_knows", "honest_error"), ("unhinted", "honest_error"),
             ("kb_instr", "unhinted"))
N_THREADS = 32


def load_meta(D: Path):
    """Reported runs exactly as analyze_instrument.extract keeps them, and the extra
    training runs as extract_resid keeps them (only the small arrays are read)."""
    ext = []
    for rd in AI.READOUT_DIRS:
        p = D / rd / "index.jsonl"
        if not p.exists():
            continue
        for m in {m["run_id"]: m for m in map(json.loads, open(p))}.values():
            with np.load(D / rd / m["file"]) as z:
                cpos = z["cot_resid_pos"].astype(int)
            np_ = m["named_positions"]
            pos = list(cpos) + [np_.get("think_end", -1), np_.get("pre_answer", -1)]
            if m["cot_span"] is None or any(q < 0 for q in pos):
                continue
            ext.append(m)
    xext = []
    seen = {m["run_id"] for m in ext}
    for rd in AI.RESID_ONLY_DIRS:
        for m in {m["run_id"]: m for m in map(json.loads, open(D / rd / "index.jsonl"))}.values():
            if m["run_id"] in seen or m["extras"]["instr_group"] != "train_extra":
                continue
            with np.load(D / rd / m["file"]) as z:
                bad = m["cot_span"] is None or (z["cot_resid_pos"] < 0).any() or (z["resid_pos"] < 0).any()
            if not bad:
                xext.append(m)
    return ext, xext


def single_ids(tok, words):
    out = []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) != 1 or tok.decode(ids) != w:
            raise ValueError(f"{w!r} is not a single token: {ids}")
        out.append(ids[0])
    return out


def lens_logprobs(lens_model, lens, H, layer, use_j, track, chunk=256):
    """H [n, d] float32 -> tracked log-probs [n, len(track)], ranks, top-25 ids / lps."""
    import torch
    sel = torch.tensor(track)
    lp_o, rk_o, ti_o, tl_o = [], [], [], []
    for s in range(0, len(H), chunk):
        h = torch.from_numpy(H[s:s + chunk]).float()
        if use_j:
            h = lens.transport(h, layer)
        lp = torch.log_softmax(lens_model.unembed(h).float(), dim=-1)
        t = lp[:, sel]
        rk = 1 + (lp[:, None, :] > t[:, :, None]).sum(-1)
        v, i = lp.topk(25, dim=-1)
        lp_o.append(t.numpy()); rk_o.append(rk.numpy()); ti_o.append(i.numpy().astype(np.int32))
        tl_o.append(v.numpy().astype(np.float16))
    return np.concatenate(lp_o), np.concatenate(rk_o), np.concatenate(ti_o), np.concatenate(tl_o)


def verify(D, lens_model, lens, n_runs=3):
    """CPU readout vs stored GPU readout at the stored block-output anchors."""
    rows = [json.loads(l) for l in open(D / "readouts/instr_ct_core/index.jsonl")][:n_runs]
    dl, top_overlap, top1 = [], [], []
    for m in rows:
        with np.load(D / "readouts/instr_ct_core" / m["file"]) as z:
            lo, layers = int(z["positions_lo"]), z["layers"].tolist()
            resid, rpos = z["resid"].astype(np.float32), z["resid_pos"]
            lp, top_ids, top_pos = z["lp"].astype(np.float32), z["top_ids"], z["top_pos"].tolist()
            tracked = z["tracked_ids"].tolist()
        for k, pos in enumerate(rpos):
            for lens_i, use_j in ((0, True), (1, False)):
                for li, layer in enumerate(layers):
                    if layer not in (16, 20, 24, 28, 32):
                        continue
                    t, _, ti, _ = lens_logprobs(lens_model, lens, resid[layer, k][None], layer, use_j, tracked)
                    dl.extend(np.abs(t[0] - lp[lens_i, li, pos - lo]).tolist())
                    if pos in top_pos:
                        st = top_ids[lens_i, li, top_pos.index(pos)]
                        top_overlap.append(len(set(st.tolist()) & set(ti[0].tolist())) / 25)
                        top1.append(int(st[0] == ti[0][0]))
    dl = np.array(dl)
    return {"n_runs": len(rows), "n_values": int(len(dl)), "tracked_lp_absdiff_median": float(np.median(dl)),
            "tracked_lp_absdiff_p99": float(np.percentile(dl, 99)), "tracked_lp_absdiff_max": float(dl.max()),
            "top25_overlap_mean": float(np.mean(top_overlap)), "top25_overlap_min": float(np.min(top_overlap)),
            "top1_agree": float(np.mean(top1)), "n_top_checks": len(top1)}


def fit_full_option_probe(O, y, train, lo):
    """Option-text probe on all training runs at one option layer; returns the raw-space
    direction d (score_i = d . (o_i - mean_j o_j) + const)."""
    from sklearn.linear_model import LogisticRegression
    tr = np.where(train)[0]
    Otr = O[tr, lo].astype(np.float32)
    d = Otr.shape[-1]
    sd_o = Otr.reshape(-1, d).std(0) + 1e-3
    mu_o = Otr.reshape(-1, d).mean(0)
    F = (Otr - mu_o) / sd_o
    F = F - F.mean(1, keepdims=True)
    lab = np.zeros((len(tr), 4), int)
    lab[np.arange(len(tr)), y[tr]] = 1
    sc_mu, sc_sd = F.reshape(-1, d).mean(0), F.reshape(-1, d).std(0) + 1e-6
    clf = LogisticRegression(C=AI.PROBE_C, max_iter=3000).fit(((F - sc_mu) / sc_sd).reshape(-1, d), lab.ravel())
    return clf.coef_[0] / (sc_sd * sd_o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    D = Path(args.data)
    t0 = time.time()
    rng = np.random.default_rng(AI.SEED)

    # ---------------------------------------------------------- runs, groups, folds
    ext, xext = load_meta(D)
    print(f"{len(ext)} reported runs, {len(xext)} extra training runs ({time.time() - t0:.0f}s)", flush=True)
    raw_group = [m["extras"]["instr_group"] for m in ext]
    group = np.array([AI.GROUP_MAP.get(g, g) for g in raw_group])
    item = [m["question_id"] for m in ext]
    C = np.array([m["letters"].index(m["correct_letter"]) for m in ext])
    A = np.array([m["letters"].index(m["answer"]) for m in ext])
    train_mask = np.isin(np.array(raw_group), list(AI.TRAIN_GROUPS))
    assert (C[train_mask] == A[train_mask]).all()
    uitems = sorted(set(item))
    perm = rng.permutation(len(uitems))
    fold_of = {q: int(perm[i] % AI.N_FOLDS) for i, q in enumerate(uitems)}
    folds = np.array([fold_of[q] for q in item])
    for m in xext:
        if m["answer"] != m["correct_letter"] or m["question_id"] not in fold_of:
            raise ValueError(f"extra training run {m['run_id']}: not correct or item outside the folds")
    all_meta = ext + xext
    Cx = np.concatenate([C, [m["letters"].index(m["correct_letter"]) for m in xext]]).astype(int)
    tmx = np.concatenate([train_mask, np.ones(len(xext), bool)])
    fx = np.concatenate([folds, [fold_of[m["question_id"]] for m in xext]]).astype(int)

    oidx = {m["run_id"]: m for m in map(json.loads, open(D / AI.OPT_DIR / "index.jsonl"))}
    have = np.array([m["run_id"] in oidx for m in all_meta])
    if not have[:len(ext)].all():
        raise SystemExit(f"option block outputs missing for {int((~have[:len(ext)]).sum())} reported runs")
    opt_layers = np.load(D / AI.OPT_DIR / oidx[all_meta[0]["run_id"]]["file"])["layers"].tolist()
    O = np.zeros((len(all_meta), len(opt_layers), 4, 4096), np.float16)
    for j, m in enumerate(all_meta):
        if have[j]:
            O[j] = np.load(D / AI.OPT_DIR / oidx[m["run_id"]]["file"])["opt_resid"]
    keep = np.where(have)[0]
    print(f"option block outputs: {len(keep)} runs, layers {opt_layers} ({time.time() - t0:.0f}s)", flush=True)
    # how item-level are the option vectors? (identical bytes across an item's runs)
    by_item = {}
    for j in range(len(ext)):
        by_item.setdefault(item[j], set()).add(hashlib.md5(O[j].tobytes()).hexdigest())
    n_distinct = [len(v) for v in by_item.values()]

    # ---------------------------------------------------------- probe reproduction
    t1 = time.time()
    AI._G.update(H=None, O=O[keep], y=Cx[keep], train=tmx[keep], folds=fx[keep])
    jobs = [("option_only", f, lo, None) for f in range(AI.N_FOLDS) for lo in range(len(opt_layers))]
    Po = np.full((len(keep), len(opt_layers), 4), np.nan)
    with Pool(min(args.jobs, len(jobs))) as pool:
        for _, te, lo, _, p in pool.imap_unordered(AI._fit_match, jobs):
            Po[te, lo] = p
    Po = Po[:len(ext)]  # keep[:len(ext)] == arange(len(ext))
    print(f"option-text probe refitted in {time.time() - t1:.0f}s", flush=True)

    # ---------------------------------------------------------- lens readouts
    import torch
    torch.set_num_threads(N_THREADS)
    from jmr.lens.load import lens_file, load_lens, load_model
    t2 = time.time()
    hf_model, tok, lens_model = load_model(device="cpu")
    lens = load_lens(lens_file())
    print(f"model + lens loaded on CPU in {time.time() - t2:.0f}s", flush=True)
    pos_ids, neg_ids = single_ids(tok, POS_WORDS), single_ids(tok, NEG_WORDS)
    letter_ids = [i for L in "ABCD" for i in LETTER_IDS[L]]
    for L in "ABCD":  # the tracked letter IDs of configs/lens.yaml
        assert single_ids(tok, [L, " " + L, "(" + L]) == LETTER_IDS[L]
    track = letter_ids + pos_ids + neg_ids
    t3 = time.time()
    check = verify(D, lens_model, lens)
    print(f"CPU vs stored GPU readout: {check} ({time.time() - t3:.0f}s)", flush=True)
    if check["tracked_lp_absdiff_p99"] > 0.15 or check["top25_overlap_mean"] < 0.9:
        raise SystemExit("CPU readout disagrees with the stored GPU readout beyond tolerance (rule 1)")

    n = len(ext)
    H = O[:n].astype(np.float32)  # [n, L, 4, d]
    TLP = np.zeros((2, n, len(opt_layers), 4, len(track)), np.float32)
    TRK = np.zeros((2, n, len(opt_layers), 4, len(track)), np.int32)
    TOPI = np.zeros((2, n, len(opt_layers), 4, 25), np.int32)
    TOPL = np.zeros((2, n, len(opt_layers), 4, 25), np.float16)
    t4 = time.time()
    for lens_i, use_j in ((0, True), (1, False)):
        for li, layer in enumerate(opt_layers):
            flat = H[:, li].reshape(-1, H.shape[-1])
            a, b, c, d = lens_logprobs(lens_model, lens, flat, layer, use_j, track)
            TLP[lens_i, :, li] = a.reshape(n, 4, -1)
            TRK[lens_i, :, li] = b.reshape(n, 4, -1)
            TOPI[lens_i, :, li] = c.reshape(n, 4, 25)
            TOPL[lens_i, :, li] = d.reshape(n, 4, 25)
            print(f"  lens {lens_i} layer {layer} done ({time.time() - t4:.0f}s)", flush=True)
    np.savez_compressed(Path(args.out).with_suffix(".top25.npz"), top_ids=TOPI, top_lp=TOPL,
                        tracked_ids=np.array(track), tracked_lp=TLP.astype(np.float16), tracked_rank=TRK,
                        run_ids=np.array([m["run_id"] for m in ext]), layers=np.array(opt_layers))

    lse = lambda x: np.logaddexp.reduce(x, axis=-1)  # noqa: E731
    nl = len(letter_ids)
    s_pos = lse(TLP[..., nl:nl + len(pos_ids)])
    s_neg = lse(TLP[..., nl + len(pos_ids):])
    scores = {}
    for lens_i, ln in ((0, "jlens"), (1, "logit")):
        scores[f"words_{ln}"] = (s_pos - s_neg)[lens_i]  # [n, L, 4 options]
        let = np.stack([lse(TLP[lens_i, ..., 3 * k:3 * k + 3]) for k in range(4)], -1)  # [n, L, 4 spans, 4 letters]
        scores[f"letters_{ln}"] = let.mean(2)  # mean over the spans -> [n, L, 4 letters]
        scores[f"poswords_{ln}"] = s_pos[lens_i]  # secondary: the halves of (a)
        scores[f"negwords_{ln}"] = -s_neg[lens_i]
    scores["option_probe"] = Po

    # ---------------------------------------------------------- metrics
    res = {"script": "scripts/lens/analyze_option_lens.py", "layers": opt_layers, "n_runs": n,
           "pos_words": dict(zip(POS_WORDS, pos_ids)), "neg_words": dict(zip(NEG_WORDS, neg_ids)),
           "cpu_vs_gpu_check": check,
           "option_vectors_distinct_per_item": {"median": float(np.median(n_distinct)), "max": int(max(n_distinct)),
                                                "n_items": len(n_distinct),
                                                "share_items_identical": float(np.mean(np.array(n_distinct) == 1))},
           "groups": {}, "readouts": {}, "headline": {}, "auroc": {}}
    for g in AI.REPORT_GROUPS:
        sel = np.where(group == g)[0]
        if len(sel):
            res["groups"][g] = {"n_runs": int(len(sel)), "n_items": len(set(item[i] for i in sel))}
    for rname, sc in scores.items():
        pc = AI.point(sc, np.broadcast_to(C[:, None], sc.shape[:2]).copy())  # [n, L]
        out = {}
        for g in AI.REPORT_GROUPS:
            sel = np.where(group == g)[0]
            if len(sel):
                out[g] = AI.boot(pc[sel], [item[i] for i in sel], rng)
        res["readouts"][rname] = out
        li = int(np.nanargmax(np.array(out["unhinted"]["mean"])))
        res["headline"][rname] = {"layer": opt_layers[li], **{
            g: {"mean": o["mean"][li], "lo": o["lo"][li], "hi": o["hi"][li], "n_items": o["n_items"],
                "n_runs": o["n_runs"]} for g, o in out.items()}}
        s = sc[:, li]
        corr = np.take_along_axis(s, C[:, None], -1)[:, 0]
        marg = corr - (np.nansum(s, -1) - corr) / 3
        for a_, b_ in CONTRASTS:
            r_ = AI.auroc_boot(marg, item, np.where(group == a_)[0], np.where(group == b_)[0], rng)
            if r_:
                res["auroc"].setdefault(rname, {})[f"{a_}_vs_{b_}"] = r_

    # reproduction check against the stored instrument result
    ref_p = D / "readouts/instrument_q1q3.json"
    if ref_p.exists():
        ref = json.loads(ref_p.read_text())["headline"]["option_only"]["all"]
        mine = res["headline"]["option_probe"]
        res["probe_reproduction"] = {
            "stored_layer": ref["layer"], "refit_layer": mine["layer"],
            **{g: {"stored": ref[g]["mid_point_correct"]["mean"], "refit": mine[g]["mean"]}
               for g in ("unhinted", "kb_instr", "i2_knows", "honest_error", "unstable_correct") if g in ref}}
        print("probe reproduction:", json.dumps(res["probe_reproduction"]), flush=True)

    # ---------------------------------------------------------- probe-direction diagnostic
    W = lens_model._lm_head.weight.float()
    g = lens_model._final_norm.weight.float()
    diag = {}
    for lo, layer in enumerate(opt_layers):
        dvec = torch.from_numpy(fit_full_option_probe(O[keep], Cx[keep], tmx[keep], lo).astype(np.float32))
        out = {"probe_direction_norm": float(dvec.norm())}
        for ln, v in (("jlens", lens.transport(dvec[None], layer)[0].float()), ("logit", dvec)):
            logit = W @ (g * v)
            top = torch.topk(logit, 15).indices.tolist()
            bot = torch.topk(-logit, 15).indices.tolist()
            z = (logit - logit.mean()) / logit.std()
            out[ln] = {"top": [[tok.decode([i]), round(float(z[i]), 2)] for i in top],
                       "bottom": [[tok.decode([i]), round(float(z[i]), 2)] for i in bot],
                       "z_pos_words": {w: round(float(z[i]), 2) for w, i in zip(POS_WORDS, pos_ids)},
                       "z_neg_words": {w: round(float(z[i]), 2) for w, i in zip(NEG_WORDS, neg_ids)}}
        diag[str(layer)] = out
    res["probe_direction_tokens"] = diag
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"→ {args.out} ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
