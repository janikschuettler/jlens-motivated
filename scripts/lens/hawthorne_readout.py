"""J-lens / logit-lens readout of the Hawthorne runs before the first test-awareness sentence (lens env, GPU).

  JMR_MODEL_PROFILE=qwen3-14b uv run --project envs/lens python scripts/lens/hawthorne_readout.py \
      --runs data/hawthorne/q14/runs.jsonl --select data/hawthorne/q14/lens_select.jsonl \
      --out data/readouts/hawthorne_q14

--select: one JSON line per run to read, {"run_id", "question_id", "group" ("aware"/"unaware"),
"lo", "hi"}: the read window is positions lo <= t < hi of prompt + completion IDs (position t holds
the distribution of token t + 1; hi is the position of the first token of the first awareness
sentence for aware runs and the matched cut for unaware runs; see scripts/explore/hawthorne_select.py).

Teacher-forces exactly the saved prompt + completion IDs (never re-tokenizes). The full softmax over
the vocabulary is computed in memory, per lens x workspace layer x position. Saved (rule 3):
- per run (npz): log-probs of the tracked eval-awareness tokens (configs/evalaware_tokens.json, all
  non-control tokens, plus the mean over the 100 control nouns as one column) at every window
  position and at 64 evenly spaced reasoning positions after the window ("post"); top-25 IDs and
  log-probs at every window position; teacher-forced log-prob of every completion token.
- open-vocabulary search (kept in RAM only, never written per run): per run and lens, the mean over
  window positions and workspace layers of every token's log-prob and probability. At the end,
  per token: within-prompt difference aware - unaware (mean over prompts with both groups), its t
  value over prompts, and a within-prompt label-permutation null of the largest |t|. Only the
  TOP_N tokens with the largest |t| per lens and statistic are written (open_vocab.json).
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from jmr import config
from jmr.lens import readout as single
from jmr.lens.batch import LENSES, BatchReader
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.tokens import fingerprint

MEDIAN_MAX, P99_MAX = 0.05, 0.5  # M0.4 tolerances (roadmap)
N_POST = 64
TOP_N = 300
N_PERM = 200


def load_tracked(path, tok):
    cfg = json.loads(Path(path).read_text())
    for t in cfg["tokens"]:
        if tok(t["text"], add_special_tokens=False).input_ids != [t["id"]] or tok.decode([t["id"]]) != t["text"]:
            raise ValueError(f"tracked token {t['text']!r} doesn't round-trip (rule 1)")
    tracked = [t for t in cfg["tokens"] if t["groups"] != ["control"]]
    control = [t for t in cfg["tokens"] if t["groups"] == ["control"]]
    if len(control) != 100:
        raise ValueError("expected 100 control words")
    return tracked, control


def t_stats(D):
    """D [Q, V]: per-prompt differences. Returns mean [V], t [V]."""
    m = D.mean(0)
    sd = D.std(0, ddof=1) if D.shape[0] > 1 else np.ones_like(m)
    return m, m / (sd / np.sqrt(D.shape[0]) + 1e-12)


def open_vocab(acc, sel, tok, rng):
    """acc[run_id] -> {"lp": [2, V], "p": [2, V]} (float32, RAM). Within-prompt comparison."""
    by = defaultdict(lambda: {"aware": [], "unaware": []})
    for s in sel:
        if s["run_id"] in acc:
            by[s["question_id"]][s["group"]].append(s["run_id"])
    prompts = [q for q, g in by.items() if g["aware"] and g["unaware"]]
    out = {"n_prompts": len(prompts), "n_aware": sum(len(by[q]["aware"]) for q in prompts),
           "n_unaware": sum(len(by[q]["unaware"]) for q in prompts), "results": {}}
    if len(prompts) < 3:
        return out
    for li, lens_name in enumerate(LENSES):
        for stat in ("lp", "p"):
            def diffs(assign):
                rows = []
                for q in prompts:
                    a = [acc[r][stat][li] for r, g in assign[q] if g == "aware"]
                    u = [acc[r][stat][li] for r, g in assign[q] if g == "unaware"]
                    rows.append(np.mean(a, 0) - np.mean(u, 0))
                return np.stack(rows)
            real = {q: [(r, "aware") for r in by[q]["aware"]] + [(r, "unaware") for r in by[q]["unaware"]]
                    for q in prompts}
            m, t = t_stats(diffs(real))
            null_max, null_min = [], []
            for _ in range(N_PERM):
                perm = {}
                for q in prompts:
                    labs = [g for _, g in real[q]]
                    rng.shuffle(labs)
                    perm[q] = [(r, g) for (r, _), g in zip(real[q], labs)]
                _, tp = t_stats(diffs(perm))
                null_max.append(float(tp.max()))
                null_min.append(float(tp.min()))
            order = np.argsort(-np.abs(t))[:TOP_N]
            mean_a = np.mean([acc[r][stat][li] for q in prompts for r in by[q]["aware"]], 0)
            mean_u = np.mean([acc[r][stat][li] for q in prompts for r in by[q]["unaware"]], 0)
            out["results"][f"{lens_name}/{stat}"] = {
                "null_max_t_q95": float(np.quantile(null_max, 0.95)), "null_min_t_q05": float(np.quantile(null_min, 0.05)),
                "real_max_t": float(t.max()), "real_min_t": float(t.min()),
                "top": [{"id": int(i), "text": tok.decode([int(i)]), "diff": float(m[i]), "t": float(t[i]),
                         "mean_aware": float(mean_a[i]), "mean_unaware": float(mean_u[i])} for i in order]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="configs/evalaware_tokens.json")
    ap.add_argument("--validate", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-tol", type=float, default=0.02, help="max |batched - single-run| tracked log-prob")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = [json.loads(l) for l in open(a.select)]
    if a.limit:
        sel = sel[: a.limit]
    want = {s["run_id"]: s for s in sel}
    runs = {}
    for l in open(a.runs):
        r = json.loads(l)
        if r["run_id"] in want:
            runs[r["run_id"]] = r
    if len(runs) != len(want):
        raise ValueError(f"{len(want)} selected, {len(runs)} found (rule 1)")
    config.snapshot(out)
    mcfg = config.load("model")
    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    reader = BatchReader(lens_model, lens, tok)
    tracked, control = load_tracked(a.tokens, tok)
    tt = torch.tensor([t["id"] for t in tracked], device=reader.device)
    ct = torch.tensor([t["id"] for t in control], device=reader.device)
    (out / "tokens.json").write_text(json.dumps({"columns": [t["text"] for t in tracked] + ["control_mean"],
                                                 "tracked": tracked, "layers": reader.layers,
                                                 "lenses": list(LENSES)}, indent=1))
    fp = fingerprint(tok)
    think_end = tok.convert_tokens_to_ids("</think>")
    nL, K, k = len(reader.layers), len(tracked), reader.top_k
    acc, diffs, vals = {}, [], []
    t0 = time.time()
    index = open(out / "index.jsonl", "w")
    for i, s in enumerate(sel):
        r = runs[s["run_id"]]
        if r["tokenizer_fingerprint"] != fp or r["model"]["revision"] != mcfg["revision"]:
            raise ValueError(f"{r['run_id']}: fingerprint or revision mismatch (rule 1)")
        P, C = r["prompt_token_ids"], r["completion_token_ids"]
        ids = P + C
        L = len(ids)
        if len(r["completion_logprobs"]) != len(C):
            raise ValueError(f"{r['run_id']}: log-prob count mismatch")
        lo, hi = int(s["lo"]), int(s["hi"])
        if not (len(P) <= lo < hi <= L):
            raise ValueError(f"{r['run_id']}: bad window {lo}..{hi} (P={len(P)}, L={L})")
        te = len(P) + C.index(think_end) if think_end in C else L - 1
        post = []
        if te > hi:
            post = sorted(set(np.linspace(hi, te - 1, min(N_POST, te - hi)).round().astype(int).tolist()))
        W = list(range(lo, hi))
        allpos = W + post
        full, _ = reader._forward(ids, [len(P) - 1])
        lp_arr = np.zeros((len(LENSES), nL, len(allpos), K + 1), dtype=np.float32)
        top_ids = np.zeros((len(LENSES), nL, len(W), k), dtype=np.int32)
        top_lp = np.zeros((len(LENSES), nL, len(W), k), dtype=np.float32)
        V = lens_model.unembed(full[reader.layers[0]][:1].float()).shape[-1]
        sum_lp = torch.zeros((len(LENSES), V), device=reader.device, dtype=torch.float64)
        sum_p = torch.zeros((len(LENSES), V), device=reader.device, dtype=torch.float64)
        with torch.no_grad():
            for si, lens_name in enumerate(LENSES):
                for li, layer in enumerate(reader.layers):
                    for start in range(0, len(allpos), reader.chunk):
                        sub = allpos[start:start + reader.chunk]
                        h = full[layer][torch.tensor(sub, device=reader.device)].float()
                        if lens_name == "jlens":
                            h = reader.lens.transport(h, layer)
                        lps = torch.log_softmax(lens_model.unembed(h).float(), dim=-1)
                        lp_arr[si, li, start:start + len(sub), :K] = lps[:, tt].float().cpu().numpy()
                        lp_arr[si, li, start:start + len(sub), K] = lps[:, ct].mean(-1).float().cpu().numpy()
                        nw = max(0, min(len(sub), len(W) - start))  # window rows in this chunk
                        if nw:
                            w = lps[:nw]
                            v, ix = w.topk(k, dim=-1)
                            top_ids[si, li, start:start + nw] = ix.int().cpu().numpy()
                            top_lp[si, li, start:start + nw] = v.float().cpu().numpy()
                            sum_lp[si] += w.sum(0).double()
                            sum_p[si] += w.exp().sum(0).double()
        n = len(W) * nL
        acc[r["run_id"]] = {"lp": (sum_lp / n).float().cpu().numpy(), "p": (sum_p / n).float().cpu().numpy()}
        tf_lp, _ = single.next_token_logprobs(lens_model, full[reader.final], ids, chunk=reader.chunk)
        tf = tf_lp[len(P) - 1:].astype(np.float32)
        d = np.abs(tf - np.asarray(r["completion_logprobs"], dtype=np.float64))
        diffs.append(d)
        v = None
        if len(vals) < a.validate:
            # recompute the tracked log-probs at the window with the M0.2-validated single-run readout
            layers = reader.layers
            v = {}
            for si, name in enumerate(LENSES):
                ref = single.readout(lens_model, {l: full[l] for l in layers}, lens=lens if name == "jlens" else None,
                                     top_k=k, token_ids=[t["id"] for t in tracked], chunk=reader.chunk,
                                     dtype=np.float32)
                rr = ref["token_logprobs"][:, allpos].astype(np.float32)
                dd = np.abs(rr - lp_arr[si, :, :, :K].astype(np.float32))
                v[name] = float(dd.max())
                v[name + "_ref_at_max"] = float(rr.flat[dd.argmax()])
            vals.append(v)
            print("validation", r["run_id"], v, flush=True)
            if max(v[n] for n in LENSES) > a.val_tol:
                raise SystemExit(f"readout disagrees with the single-run readout: {v} (rule 1)")
        del full
        fname = hashlib.sha1(r["run_id"].encode()).hexdigest()[:20] + ".npz"
        tmp = out / (fname + ".tmp.npz")
        np.savez_compressed(tmp, pos=np.array(allpos, dtype=np.int32), n_window=np.int32(len(W)), lp=lp_arr,
                            top_ids=top_ids, top_lp=top_lp, completion_tf_logprob=tf)
        os.replace(tmp, out / fname)
        index.write(json.dumps({**s, "file": fname, "prompt_len": len(P), "completion_len": len(C), "think_end": te,
                                "n_window": len(W), "n_post": len(post), "vllm_median": float(np.median(d)),
                                "validation": v}) + "\n")
        index.flush()
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    index.close()
    d = np.concatenate(diffs)
    summ = {"n_runs": len(sel), "seconds": round(time.time() - t0, 1),
            "vllm_absdiff": {"n": int(d.size), "median": float(np.median(d)), "p99": float(np.quantile(d, 0.99))},
            "fingerprint": fp, "model": mcfg["hf_id"], "revision": mcfg["revision"]}
    print(json.dumps(summ), flush=True)
    if summ["vllm_absdiff"]["median"] > MEDIAN_MAX or summ["vllm_absdiff"]["p99"] > P99_MAX:
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        raise SystemExit(f"teacher-forced log-probs disagree with vLLM beyond M0.4 tolerance (rule 1): {summ}")
    ov = open_vocab(acc, sel, tok, np.random.default_rng(0))
    (out / "open_vocab.json").write_text(json.dumps(ov, indent=1))
    summ["open_vocab"] = {k2: v2 for k2, v2 in ov.items() if k2 != "results"}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("open vocab done", json.dumps(summ["open_vocab"]), flush=True)


if __name__ == "__main__":
    main()
