"""Readouts for W, waterproofing the reward-hack null (lens env, GPU).
Pre-registration: docs/preregistration.md, section "W".

One teacher-forced pass per run over exactly the saved prompt + completion token IDs,
by the run's own model (or --reader-arm). Positions from P-1 (last prompt token) to the
end. Stored per run in one .npz (rule 3: full softmax only in memory):

  x_lp [2 lenses (jlens, logit), 10 workspace layers, T, K] fp16: log-probs of the K = 41
      tracked tokens of configs/hack_organism.yaml; T = 1 + completion length
      (row 0 = last prompt position; row t = completion token t-1's position);
  model_lp [T, K] fp16: the model's own output; kl [2, 10, T] fp16: KL(model output || lens);
  resid_sets [10 layers, 5, d] fp16: block outputs, mean over the echo-free reasoning,
      last prompt token, the "</" of </thinking>, the position before the code, mean over
      the hack-line window W2 (NaN where absent);
  resid_fine [3 layers (16, 24, 32), 15, d] fp16: first 5 reasoning tokens, 10 decile means;
  resid_win [10, d] fp16 (synthetic splices only): mean over the record's `score_window`;
  completion_tf_logprob [n_completion].

W2 window: hack runs: 32 positions from the token holding the first hack-related word;
no-hack runs: 32 positions from code_first + the median (mention - code_first) of the
same problem's hidden runs in the input, else the overall median.

--openvocab: per run, the mean probability of every vocabulary token over the windows
echo_free / pre_code / hackline, per lens and layer, kept in RAM only; after all runs the
cross-fitted open-vocabulary scores (pre-registered: problems split by a hash, Welch t on
within-problem-centred log mean probability, hidden vs no_hack; top 100 up and down tokens
on one half score the other half) and the 200 tokens with the largest |t| over all runs
per cell with their per-run values go to openvocab.npz. Nothing else of the vocabulary
is written.

--prompt-only: one read per problem of the prompt alone (row 0 only; for A1).
Synthetic splices (records with an extra `splice` field): no vLLM check, no label check.

Rule-1 checks: tokenizer fingerprint, model revision, adapter matches the arm, one vLLM
log-prob per token, label group from the token bytes = labels.jsonl; own-model reads must
pass M0.4 (median <= 0.05, p99 <= 0.5, shifted median >= 10x median).

uv run --project envs/lens python scripts/lens/readout_hack_wp.py --runs R.jsonl [R2.jsonl] \
    --labels L.jsonl [L2.jsonl] --out DIR [--reader-arm base] [--run-ids F] [--openvocab] [--validate 2]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from jlens import ActivationRecorder

from jmr import config, hack
from jmr.lens import readout as single
from jmr.lens.batch import read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.lens.lora import LoRAHooks, adapter_dir
from jmr.tokens import fingerprint, token_bytes

LENSES = ("jlens", "logit")
MEDIAN_MAX, P99_MAX, SHIFT_FACTOR = 0.05, 0.5, 10.0
SET_NAMES = ["echo_free_mean", "last_prompt", "think_end", "pre_code", "hackline_mean"]
FINE_LAYERS = [16, 24, 32]
W2_LEN = 32
OV_WINDOWS = ["echo_free", "pre_code", "hackline"]
OV_TOP, OV_KEEP = 100, 200


def tracked(cfg: dict, tok):
    ids, labels, sets = [], [], []
    for s in ("concept", "task", "neutral"):
        for text, tid in cfg["tokens"][s].items():
            enc = tok(text, add_special_tokens=False).input_ids
            if enc != [tid] or tok.decode([tid]) != text:
                raise ValueError(f"tracked token {text!r} encodes to {enc}, config {tid}")
            ids.append(tid)
            labels.append(text)
            sets.append(s)
    if len(set(ids)) != len(ids):
        raise ValueError("tracked tokens repeat")
    return ids, labels, sets


def save_npz(path: Path, arrays: dict) -> int:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path.stat().st_size


def half_of(problem_id: str) -> int:
    return int(hashlib.sha1(problem_id.encode()).hexdigest(), 16) % 2


def plan_run(record, extras, tok, offsets: dict, overall_offset: int) -> dict:
    """Absolute positions (index into prompt + completion) of every window."""
    P, n = len(record.prompt_token_ids), len(record.completion_token_ids)
    w = hack.token_windows(token_bytes(tok, record.completion_token_ids))
    pl = {"P": P, "n": n, "windows": w, "think": w.think, "echo_free": w.echo_free, "mention": w.mention,
          "think_end": w.think_end, "code_first": w.code_first}
    pl["last_prompt"] = P - 1
    pl["think_end_pos"] = P + w.think_end if w.think_end is not None else None
    pl["pre_code_pos"] = P + w.code_first - 1 if w.code_first is not None else None
    ef = w.echo_free
    pl["echo_free_pos"] = (P + ef[0], P + ef[1]) if ef else None
    hl = None
    if extras.get("splice") is None:
        a = None
        if extras.get("hack_label"):
            a = w.mention
        elif w.code_first is not None:
            a = w.code_first + offsets.get(record.question_id, overall_offset)
        if a is not None and a < n:
            hl = (P + a, P + min(n, a + W2_LEN))
    pl["hackline_pos"] = hl
    th = w.think
    pl["first5_pos"] = [P + th[0] + k if th and th[0] + k < th[1] else None for k in range(5)]
    dec = [None] * 10
    if th:
        m = th[1] - th[0]
        bins = defaultdict(list)
        for i in range(m):
            bins[(10 * i) // m].append(P + th[0] + i)
        dec = [(min(bins[b]), max(bins[b]) + 1) if bins[b] else None for b in range(10)]
    pl["decile_pos"] = dec
    sw = (extras.get("splice") or {}).get("score_window")
    pl["score_window_pos"] = (P + sw[0], P + sw[1]) if sw else None
    return pl


class Reader:
    def __init__(self, lens_model, lens, tok, cfg, lcfg, openvocab: bool):
        self.m, self.lens, self.tok = lens_model, lens, tok
        self.layers = sorted(lcfg["workspace_layers"])
        self.final = lens_model.n_layers - 1
        self.chunk = lcfg["position_chunk"]
        self.ids, self.labels, self.sets = tracked(cfg, tok)
        self.dev = lens_model.input_device
        self.tt = torch.tensor(self.ids, device=self.dev)
        lens.jacobians = {l: lens.jacobians[l].to(self.dev) for l in self.layers}
        self.openvocab = openvocab

    @torch.no_grad()
    def read(self, record, pl, prompt_only=False) -> tuple[dict, np.ndarray | None]:
        ids = list(record.prompt_token_ids) + ([] if prompt_only else list(record.completion_token_ids))
        L, P = len(ids), pl["P"]
        lo = P - 1
        T, K, nL = L - lo, len(self.ids), len(self.layers)
        full_layers = sorted(set(self.layers) | {self.final})
        with ActivationRecorder(self.m.layers, at=full_layers) as rec:
            self.m.forward(torch.tensor([ids], device=self.dev))
            full = {l: rec.activations[l][0].detach() for l in full_layers}
        x_lp = torch.zeros((2, nL, T, K), dtype=torch.float16, device=self.dev)
        kl = torch.zeros((2, nL, T), dtype=torch.float16, device=self.dev)
        m_lp = torch.zeros((T, K), dtype=torch.float16, device=self.dev)
        next_lp = torch.zeros(L, dtype=torch.float32, device=self.dev)
        target = torch.tensor(ids[1:] + [0], device=self.dev)
        ov_win = {}
        if self.openvocab and not prompt_only:
            if pl["echo_free_pos"]:
                ov_win["echo_free"] = pl["echo_free_pos"]
            if pl["pre_code_pos"] is not None:
                ov_win["pre_code"] = (pl["pre_code_pos"], pl["pre_code_pos"] + 1)
            if pl["hackline_pos"]:
                ov_win["hackline"] = pl["hackline_pos"]
        V = self.m.unembed(full[self.final][:1].float()).shape[-1]
        ov = torch.zeros((len(OV_WINDOWS), 2, nL, V), dtype=torch.float32, device=self.dev) if ov_win else None
        # Chunks aligned at position 0 as in jmr.lens.readout (identical kernel shapes, so the
        # J-lens numbers match the validated single-run readout exactly); prompt chunks before
        # the one holding P-1 are skipped, and rows before P-1 of that chunk are dropped.
        for cs in range((lo // self.chunk) * self.chunk, L, self.chunk):
            end = min(cs + self.chunk, L)
            full_chunk = {k: v[cs:end] for k, v in full.items()}
            mlp_all = torch.log_softmax(self.m.unembed(full_chunk[self.final].float()).float(), dim=-1)
            start = max(cs, lo)
            mlp = mlp_all[start - cs:]
            stop = min(end, L - 1)
            if stop > start:
                next_lp[start:stop] = mlp[torch.arange(stop - start, device=self.dev), target[start:stop]]
            m_lp[start - lo:end - lo] = mlp[:, self.tt].half()
            mp = mlp.exp()
            ent = (mp * mlp).sum(-1)
            for si, name in enumerate(LENSES):
                for li, layer in enumerate(self.layers):
                    h = full_chunk[layer].float()
                    if name == "jlens":
                        h = self.lens.transport(h, layer)
                    lp = torch.log_softmax(self.m.unembed(h).float(), dim=-1)[start - cs:]
                    x_lp[si, li, start - lo:end - lo] = lp[:, self.tt].half()
                    kl[si, li, start - lo:end - lo] = (ent - (mp * lp).sum(-1)).half()
                    for wi, wn in enumerate(OV_WINDOWS):
                        if wn not in ov_win:
                            continue
                        a, b = ov_win[wn]
                        a, b = max(a, start), min(b, end)
                        if b > a:
                            ov[wi, si, li] += lp[a - start:b - start].exp().sum(0)
                    del lp
            del mlp, mp, mlp_all
        if ov is not None:
            for wi, wn in enumerate(OV_WINDOWS):
                if wn in ov_win:
                    a, b = ov_win[wn]
                    ov[wi] /= (b - a)
                else:
                    ov[wi] = float("nan")

        def mean_at(layer, span):
            if span is None:
                return None
            a, b = span
            return full[layer][a:b].float().mean(0)

        d = full[self.final].shape[-1]
        rs = torch.full((nL, len(SET_NAMES), d), float("nan"), device=self.dev)
        for li, l in enumerate(self.layers):
            vals = [mean_at(l, pl["echo_free_pos"]),
                    full[l][pl["last_prompt"]].float(),
                    None if pl["think_end_pos"] is None or prompt_only else full[l][pl["think_end_pos"]].float(),
                    None if pl["pre_code_pos"] is None or prompt_only else full[l][pl["pre_code_pos"]].float(),
                    None if prompt_only else mean_at(l, pl["hackline_pos"])]
            for k, v in enumerate(vals):
                if v is not None:
                    rs[li, k] = v
        rf = torch.full((len(FINE_LAYERS), 15, d), float("nan"), device=self.dev)
        if not prompt_only:
            for fi, l in enumerate(FINE_LAYERS):
                for k, p in enumerate(pl["first5_pos"]):
                    if p is not None:
                        rf[fi, k] = full[l][p].float()
                for k, sp in enumerate(pl["decile_pos"]):
                    if sp is not None:
                        rf[fi, 5 + k] = mean_at(l, sp)
        out = {"layers": np.array(self.layers, dtype=np.int16), "positions_lo": np.array(lo, dtype=np.int32),
               "tracked_ids": np.array(self.ids, dtype=np.int32),
               "x_lp": x_lp.cpu().numpy(), "model_lp": m_lp.cpu().numpy(), "kl": kl.cpu().numpy(),
               "resid_sets": rs.half().cpu().numpy(), "resid_fine": rf.half().cpu().numpy(),
               "completion_tf_logprob": next_lp[P - 1:L - 1].cpu().numpy()}
        if pl["score_window_pos"] and not prompt_only:
            out["resid_win"] = torch.stack([mean_at(l, pl["score_window_pos"]) for l in self.layers]).half().cpu().numpy()
        for k in ("resid_sets", "resid_fine", "resid_win"):
            if k in out and np.isinf(out[k]).any():
                raise ValueError(f"{record.run_id}: block outputs overflow fp16 ({k})")
        return out, (ov.cpu().numpy() if ov is not None else None)


def validate(reader: Reader, record, arrays) -> dict:
    """Recompute tracked log-probs with the M0.2-validated single-run readout and compare."""
    ids = list(record.prompt_token_ids) + list(record.completion_token_ids)
    resid = single.residuals(reader.m, ids, reader.layers + [reader.final])
    lo = int(arrays["positions_lo"])
    out = {}
    for si, name in enumerate(LENSES):
        ref = single.readout(reader.m, {l: resid[l] for l in reader.layers},
                             lens=reader.lens if name == "jlens" else None, top_k=1,
                             token_ids=reader.ids, chunk=reader.chunk)
        rf = ref["token_logprobs"][:, lo:]
        dd = np.abs(rf.astype(np.float32) - arrays["x_lp"][si].astype(np.float32))
        ulp = np.abs(np.spacing(rf.astype(np.float16))).astype(np.float32)
        out[name] = float(dd.max())
        out[name + "_max_ulps"] = float((dd / ulp).max())
        out[name + "_max_absdiff_ref_ge_-20"] = float(dd[rf >= -20].max()) if (rf >= -20).any() else 0.0
    return out


def openvocab_finish(ov_rows: list, out: Path) -> dict:
    """Cross-fitted open-vocab scores and the 200 most different tokens per cell."""
    meta = [r for r, _ in ov_rows]
    arr = np.stack([a for _, a in ov_rows])  # [n, W, 2, nL, V]
    n, W, S, nL, V = arr.shape
    grp = np.array([1 if m["group"] == "hidden" else 0 if m["group"] == "no_hack" else -1 for m in meta])
    pid = np.array([m["problem_id"] for m in meta])
    both = {p for p in set(pid) if (grp[pid == p] == 1).any() and (grp[pid == p] == 0).any()}
    use = np.array([g >= 0 and p in both for g, p in zip(grp, pid)])
    half = np.array([half_of(p) for p in pid])
    scores = np.full((n, W, S, nL), np.nan, dtype=np.float32)
    keep_ids = np.full((W, S, nL, OV_KEEP), -1, dtype=np.int32)
    keep_vals = np.full((n, W, S, nL, OV_KEEP), np.nan, dtype=np.float32)
    keep_t = np.full((W, S, nL, OV_KEEP), np.nan, dtype=np.float32)
    sel_ids = np.full((2, W, S, nL, 2, OV_TOP), -1, dtype=np.int32)

    def welch(X, g, p):
        Xc = X.copy()
        for q in np.unique(p):
            m = p == q
            Xc[m] -= Xc[m].mean(0)
        a, b = Xc[g == 1], Xc[g == 0]
        va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
        return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb + 1e-12)

    for wi in range(W):
        for si in range(S):
            for li in range(nL):
                X = np.log(np.maximum(arr[:, wi, si, li, :], 1e-30)).astype(np.float32)
                ok = use & np.isfinite(X[:, 0])
                if ok.sum() < 10:
                    continue
                t_all = welch(X[ok], grp[ok], pid[ok])
                top = np.argsort(-np.abs(np.nan_to_num(t_all)))[:OV_KEEP]
                keep_ids[wi, si, li] = top
                keep_t[wi, si, li] = t_all[top]
                keep_vals[:, wi, si, li] = X[:, top]
                for h in (0, 1):
                    m = ok & (half == h)
                    if (grp[m] == 1).sum() < 3 or (grp[m] == 0).sum() < 3:
                        continue
                    t = np.nan_to_num(welch(X[m], grp[m], pid[m]))
                    up, dn = np.argsort(-t)[:OV_TOP], np.argsort(t)[:OV_TOP]
                    sel_ids[h, wi, si, li, 0], sel_ids[h, wi, si, li, 1] = up, dn
                    tgt = np.isfinite(X[:, 0]) & (half != h)
                    scores[tgt, wi, si, li] = X[tgt][:, up].mean(1) - X[tgt][:, dn].mean(1)
    np.savez_compressed(out / "openvocab.npz", run_ids=np.array([m["run_id"] for m in meta]),
                        problem_ids=pid, groups=np.array([m["group"] for m in meta]), half=half,
                        windows=np.array(OV_WINDOWS), lenses=np.array(LENSES), scores=scores,
                        keep_ids=keep_ids, keep_t=keep_t, keep_vals=keep_vals, sel_ids=sel_ids)
    return {"n_runs": n, "n_problems_both": len(both), "n_used": int(use.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reader-arm", default=None)
    ap.add_argument("--run-ids", default=None)
    ap.add_argument("--groups", default=None, help="only runs of these label groups")
    ap.add_argument("--usable-only", action="store_true", help="only finished runs with a <thinking> block")
    ap.add_argument("--openvocab", action="store_true")
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--validate", type=int, default=0)
    args = ap.parse_args()

    cfg, lcfg, mcfg = config.load("hack_organism"), config.load("lens"), config.load("model")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    labels = {}
    for lf in args.labels:
        for l in open(lf):
            j = json.loads(l)
            labels[j["run_id"]] = j
    runs = []
    for rf in args.runs:
        runs += read_runs_lenient(Path(rf))
    if args.run_ids:
        keep = {l.strip() for l in open(args.run_ids) if l.strip()}
        runs = [(r, e) for r, e in runs if r.run_id in keep]
        missing = keep - {r.run_id for r, _ in runs}
        if missing:
            raise ValueError(f"{len(missing)} requested run IDs not in the runs files, e.g. {sorted(missing)[:3]}")
    if args.groups:
        gs = set(args.groups.split(","))
        runs = [(r, e) for r, e in runs if e.get("splice") or labels[r.run_id]["group"] in gs]
    if args.usable_only:
        runs = [(r, e) for r, e in runs if e.get("splice") or
                (r.finish_reason == "stop" and labels[r.run_id]["group"] != "no_thinking")]
    if args.prompt_only:
        seen, pr = set(), []
        for r, e in runs:
            if r.question_id not in seen:
                seen.add(r.question_id)
                pr.append((r, e))
        runs = pr

    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    reader = Reader(lens_model, lens, tok, cfg, lcfg, args.openvocab)
    fp = fingerprint(tok)

    # W2 offsets: per problem, median (mention - code_first) over its hidden runs in the input.
    offs = defaultdict(list)
    wins = {}
    for r, e in runs:
        if e.get("splice") or args.prompt_only:
            continue
        w = hack.token_windows(token_bytes(tok, r.completion_token_ids))
        wins[r.run_id] = w
        if labels[r.run_id]["group"] == "hidden" and w.mention is not None and w.code_first is not None \
                and w.mention >= w.code_first:
            offs[r.question_id].append(w.mention - w.code_first)
    allo = [v for vs in offs.values() for v in vs]
    overall = int(np.median(allo)) if allo else 0
    offsets = {p: int(np.median(v)) for p, v in offs.items()}
    (out / "w2_offsets.json").write_text(json.dumps({"overall": overall, "per_problem": offsets}))

    index_path = out / "index.jsonl"
    done = set()
    if index_path.exists() and not args.openvocab:
        for line in open(index_path):
            m = json.loads(line)
            if (out / m["file"]).exists():
                done.add(m["run_id"])
    elif args.openvocab and index_path.exists():
        index_path.unlink()  # open-vocab needs every run in one process
    todo = [(r, e) for r, e in runs if r.run_id not in done]
    print(f"{len(runs)} runs, {len(done)} done, {len(todo)} to do -> {out}", flush=True)
    by_reader = defaultdict(list)
    for r, e in todo:
        by_reader[args.reader_arm or r.condition].append((r, e))

    diffs, shifted, validations, sizes, ov_rows = [], [], [], [], []
    t_start, n_done = time.time(), 0
    with open(index_path, "a") as index:
        for arm, rs in by_reader.items():
            adir = adapter_dir(arm)
            hooks = LoRAHooks(hf_model, adir) if adir else None
            print(f"reader {arm}: {len(rs)} runs, adapter {adir}", flush=True)
            for i, (record, extras) in enumerate(rs):
                synth = extras.get("splice") is not None
                own = record.condition == arm and not synth and not args.prompt_only
                if record.tokenizer_fingerprint != fp or record.model.get("revision") != mcfg["revision"]:
                    raise ValueError(f"{record.run_id}: tokenizer or model revision mismatch")
                exp = None if cfg["arms"][record.condition] is None else {
                    "repo": cfg["adapters_repo"], "revision": cfg["adapters_revision"],
                    "subfolder": cfg["arms"][record.condition]}
                if record.model.get("adapter") != exp:
                    raise ValueError(f"{record.run_id}: adapter {record.model.get('adapter')} != config {exp}")
                if len(record.completion_logprobs) != len(record.completion_token_ids):
                    raise ValueError(f"{record.run_id}: log-prob count mismatch")
                lab = labels.get(record.run_id) or labels.get((extras.get("splice") or {}).get("orig_run_id"))
                ex = dict(extras)
                ex["hack_label"] = bool(lab["hack"]) if not synth else False
                pl = plan_run(record, ex, tok, offsets, overall)
                if not synth and not args.prompt_only:
                    relab = hack.label_run(pl["windows"].text)
                    if relab.group != lab["group"]:
                        raise ValueError(f"{record.run_id}: token-byte group {relab.group} != labels {lab['group']}")
                t0 = time.time()
                arrays, ov = reader.read(record, pl, prompt_only=args.prompt_only)
                torch.cuda.synchronize()
                dt = time.time() - t0
                vd = None
                if own:
                    tf = arrays["completion_tf_logprob"].astype(np.float64)
                    vl = np.asarray(record.completion_logprobs, dtype=np.float64)
                    d = np.abs(tf - vl)
                    diffs.append(d)
                    shifted.append(np.abs(tf[1:] - vl[:-1]))
                    vd = {"median": float(np.median(d)), "p99": float(np.quantile(d, 0.99)), "max": float(d.max())}
                if i < args.validate and not args.prompt_only:
                    validations.append({"run_id": record.run_id, "reader": arm, **validate(reader, record, arrays)})
                    print(json.dumps(validations[-1]), flush=True)
                tag = "prompt|" if args.prompt_only else ""
                fname = hashlib.sha1(f"{tag}{arm}|{record.run_id}".encode()).hexdigest()[:20] + ".npz"
                meta = {"run_id": record.run_id, "arm": record.condition, "reader": arm, "own": own,
                        "synthetic": synth, "prompt_only": args.prompt_only, "splice": extras.get("splice"),
                        "problem_id": record.question_id, "group": lab["group"], "hack": lab["hack"],
                        "sample_idx": record.sample_idx,
                        "prompt_len": pl["P"], "completion_len": pl["n"], "finish_reason": record.finish_reason,
                        "think": pl["think"], "echo_free": pl["echo_free"], "mention": pl["mention"],
                        "think_end": pl["think_end"], "code_first": pl["code_first"],
                        "pos": {k: pl[k] for k in ("last_prompt", "think_end_pos", "pre_code_pos", "echo_free_pos",
                                                   "hackline_pos", "first5_pos", "decile_pos", "score_window_pos")},
                        "set_names": SET_NAMES, "fine_layers": FINE_LAYERS,
                        "tracked": {"ids": reader.ids, "labels": reader.labels, "sets": reader.sets},
                        "file": fname, "seconds": round(dt, 3), "vllm_absdiff": vd}
                arrays["meta_json"] = np.array(json.dumps(meta))
                sizes.append(save_npz(out / fname, arrays))
                if ov is not None:
                    ov_rows.append(({"run_id": record.run_id, "problem_id": record.question_id,
                                     "group": lab["group"]}, ov))
                index.write(json.dumps(meta) + "\n")
                index.flush()
                n_done += 1
                if i % 25 == 0 or i == len(rs) - 1:
                    el = time.time() - t_start
                    print(f"[{arm} {i + 1}/{len(rs)}] {dt:.2f}s | {n_done / el * 60:.1f} runs/min | "
                          f"{np.mean(sizes) / 1e6:.2f} MB/run", flush=True)
            if hooks:
                hooks.remove()
    summary = {"n_new": n_done, "elapsed_s": round(time.time() - t_start, 1), "validations": validations,
               "disk_bytes_total_new": int(np.sum(sizes)) if sizes else 0}
    if ov_rows:
        summary["openvocab"] = openvocab_finish(ov_rows, out)
    ok = True
    if diffs:
        dd, ss = np.concatenate(diffs), np.concatenate(shifted)
        med, p99 = float(np.median(dd)), float(np.quantile(dd, 0.99))
        summary["vllm_logprob_absdiff_own"] = {"n_tokens": int(dd.size), "median": med, "p99": p99,
                                               "max": float(dd.max()), "shifted_median": float(np.median(ss))}
        summary["m0_4_pass"] = {"median": med <= MEDIAN_MAX, "p99": p99 <= P99_MAX,
                                "shift": float(np.median(ss)) >= SHIFT_FACTOR * med}
        ok = all(summary["m0_4_pass"].values())
    # Positions start at P-1 here (not 0 as in the single-run readout), so chunk shapes differ and
    # fp16-stored log-probs can differ by rounding; fail beyond 2 fp16 ULPs of the reference.
    if validations and max(max(v["jlens_max_ulps"], v["logit_max_ulps"]) for v in validations) > 2:
        ok = False
        summary["validation_fail"] = True
    path = out / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not ok:
        sys.exit("FAIL: M0.4 or validation against the single-run readout")


if __name__ == "__main__":
    main()
