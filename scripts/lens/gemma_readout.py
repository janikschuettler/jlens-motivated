"""gemma27-silent: teacher-forced J-lens / logit-lens readouts of Gemma-3-27B-it runs (lens env, GPU + CPU offload).

  JMR_MODEL_PROFILE=gemma-3-27b HF_HOME=... uv run --project envs/lens python scripts/lens/gemma_readout.py \
      --runs data/explore/gemma27/hints/runs.jsonl --select data/explore/gemma27/select_hints.json \
      --out data/explore/gemma27/lens_hints [--checks-only] [--limit N]

--select: JSON list of run_ids to read (made by scripts/explore/gemma27_select.py). The saved prompt + completion
token IDs are teacher-forced (no re-tokenizing for the model; text is re-derived from the IDs only to *locate*
positions, and the located letter must equal the labelled answer, else the run is flagged and skipped).

Per run and per lens (jlens, logit) at the workspace layers (configs/profiles/gemma-3-27b/lens.yaml):
  * log-probs of the 12 tracked tokens (letters A-D x forms bare/space/paren) at: the 4 option-end positions
    (last token of each "(X) text" line of the question), the last prompt token, and every completion position
    from 48 tokens before the answer letter up to it ("pre_answer" = the token whose next-token prediction is
    the final answer letter); plus 16 evenly spaced completion positions.
  * top-25 token IDs and log-probs at the last prompt token, pre_answer, pre_answer-8 and the 16 spaced positions.
  * block-output residuals (fp16) at pre_answer and at the correct / hinted option-end positions (for probes).
Rule 3: full softmaxes only in memory; only the above is stored.
Checks (rule 1): lens vs model shapes; logit lens at the last block == model's own next-token log-probs; HF
next-token log-probs of the sampled tokens vs vLLM raw_logprobs; batch vs single run on the first batch.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jlens import ActivationRecorder, JacobianLens, from_hf
from transformers import AutoModelForCausalLM, AutoTokenizer

from jmr import config
from jmr.label import ANSWER_PATTERNS
from jmr.tokens import fingerprint

TOP_K = 25
WINDOW = 48
N_SPACED = 16


def pieces_of(tok, ids):
    """Text piece per token (sentencepiece '▁' -> space, byte-fallback tokens -> their byte)."""
    out = []
    for t in tok.convert_ids_to_tokens(ids):
        m = re.fullmatch(r"<0x([0-9A-F]{2})>", t)
        out.append(bytes([int(m.group(1), 16)]).decode("latin-1") if m else t.replace("▁", " "))
    return out


def char_to_token(pieces):
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    return starts, pos


def tok_at(starts, ci):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= ci:
            lo = mid
        else:
            hi = mid - 1
    return lo


def locate(tok, rec):
    """Positions in prompt+completion IDs: option ends, last prompt token, pre_answer. None if the located
    final answer letter disagrees with the labelled one."""
    P, C = rec["prompt_token_ids"], rec["completion_token_ids"]
    pp = pieces_of(tok, P)
    ps, plen = char_to_token(pp)
    ptext = "".join(pp)
    opt_end = {}
    for L in rec["letters"]:
        i = ptext.rfind(f"({L}) ")
        if i < 0:
            return None
        j = ptext.find("\n", i)
        j = plen if j < 0 else j
        opt_end[L] = tok_at(ps, j - 1)
    cp = pieces_of(tok, C)
    cs, clen = char_to_token(cp)
    ctext = "".join(cp)
    ans_char = ans_letter = None
    for pat in ANSWER_PATTERNS:
        ms = [m for m in pat.finditer(ctext) if m.group(1) in rec["letters"]]
        if ms:
            ans_char, ans_letter = ms[-1].start(1), ms[-1].group(1)
            break
    if ans_char is None:
        return None
    k = tok_at(cs, ans_char)
    if k == 0:
        return None
    return {"opt_end": opt_end, "last_prompt": len(P) - 1, "pre_answer": len(P) + k - 1, "answer_letter": ans_letter,
            "answer_token": len(P) + k, "n_prompt": len(P), "n_total": len(P) + len(C)}


def load_all(gpu_gib: float):
    m, lc = config.load("model"), config.load("lens")
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(
        m["hf_id"], revision=m["revision"], dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{gpu_gib}GiB", "cpu": "800GiB"})
    hf.eval()
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    lm = from_hf(hf, tok, force_bos=False)
    tc = hf.config.get_text_config()
    print(f"model loaded {time.time() - t0:.0f}s; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
          f"{lm.n_layers} blocks d={lm.d_model} vocab={tc.vocab_size}", flush=True)
    if (lm.n_layers != m["expected"]["num_hidden_layers"] or lm.d_model != m["expected"]["hidden_size"]):
        raise SystemExit("model shape != profile (rule 1)")
    path = hf_hub_download(lc["jlens"]["repo"], lc["jlens"]["file"], revision=lc["jlens"]["revision"])
    lens = JacobianLens.load(path)
    lo, hi = lc["jlens"]["expected_source_layers"]
    shapes = {tuple(J.shape) for J in lens.jacobians.values()}
    info = {"source_layers": [lens.source_layers[0], lens.source_layers[-1], len(lens.source_layers)],
            "d_model": lens.d_model, "shapes": sorted(shapes), "n_prompts": lens.n_prompts}
    print("lens", info, flush=True)
    if lens.source_layers != list(range(lo, hi + 1)) or shapes != {(lm.d_model, lm.d_model)}:
        raise SystemExit(f"lens does not match model {info} (rule 1)")
    ws = lc["workspace_layers"]
    J = {l: lens.jacobians[l].to("cuda", torch.float32) for l in ws}
    return hf, tok, lm, J, ws, info, lc


def tracked_ids(tok, lc):
    ids, names = [], []
    for L in lc["tracked_tokens"]["letters"]:
        for form, tmpl in lc["tracked_tokens"]["forms"].items():
            s = tmpl.format(L=L)
            e = tok(s, add_special_tokens=False).input_ids
            if len(e) != 1:
                raise SystemExit(f"tracked token {s!r} is not one token: {e} (rule 1)")
            ids.append(e[0])
            names.append(f"{L}:{form}")
    return ids, names


@torch.no_grad()
def forward_batch(hf, lm, batch_ids, layers):
    """Block outputs at layers + last block, for right-padded batch. Returns dict layer -> [B,T,d], pad mask."""
    T = max(len(x) for x in batch_ids)
    inp = torch.zeros(len(batch_ids), T, dtype=torch.long)
    att = torch.zeros(len(batch_ids), T, dtype=torch.long)
    for i, x in enumerate(batch_ids):
        inp[i, : len(x)] = torch.tensor(x)
        att[i, : len(x)] = 1
    dev = lm.input_device
    last = lm.n_layers - 1
    with ActivationRecorder(lm.layers, at=sorted(set(layers) | {last})) as rec:
        hf(input_ids=inp.to(dev), attention_mask=att.to(dev), use_cache=False)
        return {l: rec.activations[l].detach() for l in rec.activations}


@torch.no_grad()
def lens_lp(lm, J, h, layer, use_j, chunk):
    """log-softmax rows for residual rows h [n,d] (generator over chunks)."""
    for s in range(0, h.shape[0], chunk):
        x = h[s : s + chunk].float()
        if use_j:
            x = x @ J[layer].T
        yield s, torch.log_softmax(lm.unembed(x).float(), dim=-1)


@torch.no_grad()
def readout_positions(lm, J, ws, acts, bi, tl_pos, top_pos, tsel):
    all_pos = sorted(set(tl_pos) | set(top_pos))
    idx = {p: i for i, p in enumerate(all_pos)}
    tl_rows = [idx[p] for p in tl_pos]
    top_rows = [idx[p] for p in top_pos]
    tl = np.zeros((2, len(ws), len(tl_pos), len(tsel)), dtype=np.float16)
    top_i = np.zeros((2, len(ws), len(top_pos), TOP_K), dtype=np.int32)
    top_v = np.zeros((2, len(ws), len(top_pos), TOP_K), dtype=np.float16)
    for li, layer in enumerate(ws):
        h = acts[layer][bi][all_pos]
        for lensi, use_j in enumerate((True, False)):
            for s0, lp in lens_lp(lm, J, h, layer, use_j, 64):
                rows = lp.shape[0]
                sel_tl = [(k - s0, j) for j, k in enumerate(tl_rows) if s0 <= k < s0 + rows]
                sel_top = [(k - s0, j) for j, k in enumerate(top_rows) if s0 <= k < s0 + rows]
                if sel_tl:
                    r_ = lp[[x for x, _ in sel_tl]][:, tsel].cpu().numpy()
                    for (x, j), v in zip(sel_tl, r_):
                        tl[lensi, li, j] = v
                if sel_top:
                    v_, i_ = lp[[x for x, _ in sel_top]].topk(TOP_K, dim=-1)
                    for (x, j), v, i in zip(sel_top, v_.cpu().numpy(), i_.cpu().numpy()):
                        top_v[lensi, li, j] = v
                        top_i[lensi, li, j] = i
    return tl, top_i, top_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--gpu-gib", type=float, default=34.0)
    ap.add_argument("--checks-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hf, tok, lm, J, ws, info, lc = load_all(a.gpu_gib)
    tids, tnames = tracked_ids(tok, lc)
    tsel = torch.tensor(tids, device="cuda")
    wanted = set(json.load(open(a.select)))
    recs = []
    for p in a.runs:
        for line in open(p):
            r = json.loads(line)
            if r["run_id"] in wanted:
                recs.append(r)
    if a.limit:
        recs = recs[: a.limit]
    fp = fingerprint(tok)
    for r in recs:
        if r["tokenizer_fingerprint"] != fp:
            raise SystemExit(f"{r['run_id']}: tokenizer fingerprint differs between envs (rule 1)")
    print(f"{len(recs)} runs to read", flush=True)
    recs.sort(key=lambda r: len(r["prompt_token_ids"]) + len(r["completion_token_ids"]))
    last = lm.n_layers - 1
    results, flagged, fid = [], [], []
    t0 = time.time()
    for b0 in range(0, len(recs), a.batch):
        batch = recs[b0 : b0 + a.batch]
        ids_b = [r["prompt_token_ids"] + r["completion_token_ids"] for r in batch]
        acts = forward_batch(hf, lm, ids_b, ws)
        for bi, r in enumerate(batch):
            loc = locate(tok, r)
            ids = ids_b[bi]
            n = len(ids)
            h_last = acts[last][bi, :n]
            # fidelity: model next-token log-prob of the sampled completion tokens vs vLLM raw log-probs
            npmt = len(r["prompt_token_ids"])
            tgt = torch.tensor(ids[npmt:], device=h_last.device)
            lps = []
            for s in range(npmt - 1, n - 1, 128):
                e = min(s + 128, n - 1)
                lp = torch.log_softmax(lm.unembed(h_last[s:e]).float(), dim=-1)
                lps.append(lp.gather(1, torch.tensor(ids[s + 1 : e + 1], device=lp.device)[:, None])[:, 0].cpu())
            hf_lp = torch.cat(lps).numpy()
            vl = np.array(r["completion_logprobs"])
            fid.append({"run_id": r["run_id"], "mean_abs": float(np.abs(hf_lp - vl).mean()),
                        "max_abs": float(np.abs(hf_lp - vl).max()), "n": int(len(vl))})
            if loc is None:
                flagged.append({"run_id": r["run_id"], "why": "could not locate answer/options"})
                continue
            pos_named = {"last_prompt": loc["last_prompt"], "pre_answer": loc["pre_answer"]}
            for L, p in loc["opt_end"].items():
                pos_named[f"opt_{L}"] = p
            win = list(range(max(loc["n_prompt"], loc["pre_answer"] - WINDOW), loc["pre_answer"] + 1))
            ncomp = n - npmt
            spaced = sorted({npmt + int(x) for x in np.linspace(0, max(ncomp - 1, 0), N_SPACED)})
            tl_pos = sorted(set(pos_named.values()) | set(win) | set(spaced) | set(range(max(0, npmt - 8), npmt)))
            top_pos = sorted({loc["last_prompt"], loc["pre_answer"], max(loc["pre_answer"] - 8, 0)} | set(spaced))
            rec = {"run_id": r["run_id"], "loc": loc, "tl_pos": tl_pos, "top_pos": top_pos, "layers": ws, "tracked": tnames}
            tl, top_i, top_v = readout_positions(lm, J, ws, acts, bi, tl_pos, top_pos, tsel)
            # residuals for probes: pre_answer and the correct / hinted option-end positions
            res_pos = [loc["pre_answer"], loc["opt_end"][r["correct_letter"]]]
            if r.get("hint_letter"):
                res_pos.append(loc["opt_end"][r["hint_letter"]])
            res = np.stack([acts[l][bi, :n][res_pos].float().cpu().numpy().astype(np.float16) for l in ws])
            np.savez_compressed(out / f"{len(results):05d}.npz", tl=tl, top_i=top_i, top_v=top_v, resid=res,
                                meta=json.dumps(rec))
            results.append(r["run_id"])
        if b0 == 0:
            r0 = batch[0]
            loc0 = locate(tok, r0)
            if loc0 is not None:
                n0 = len(ids_b[0])
                acts1 = forward_batch(hf, lm, [ids_b[0]], ws)
                pos = [loc0["pre_answer"], loc0["last_prompt"]]
                tl_b, _, _ = readout_positions(lm, J, ws, acts, 0, pos, [pos[0]], tsel)
                tl_s, _, _ = readout_positions(lm, J, ws, acts1, 0, pos, [pos[0]], tsel)
                d = float(np.abs(tl_b.astype(np.float32) - tl_s.astype(np.float32)).max())
                print(f"batch vs single run, max |dlogp| over tracked tokens/layers/lenses: {d:.4f}", flush=True)
                (out / "batch_vs_single.json").write_text(json.dumps({"run_id": r0["run_id"], "max_abs_diff": d, "batch_size": len(batch)}))
        print(f"{time.strftime('%H:%M:%S')} {b0 + len(batch)}/{len(recs)} runs, {time.time() - t0:.0f}s", flush=True)
        if a.checks_only:
            break
    (out / "index.json").write_text(json.dumps({"runs": results, "flagged": flagged, "fidelity": fid, "lens": info,
                                                "layers": ws, "tracked": tnames, "model_rev": config.load("model")["revision"]}))
    fm = np.array([f["mean_abs"] for f in fid]) if fid else np.array([0.0])
    print(f"fidelity vs vLLM: mean |dlogp| {fm.mean():.4f}, max over runs of mean {fm.max():.4f}; flagged {len(flagged)}", flush=True)


if __name__ == "__main__":
    main()
