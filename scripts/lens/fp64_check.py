"""Diagnostic: is the J-lens batch-vs-single readout gap just numerical noise?

For the first N selected test-awareness runs, compute the tracked-token J-lens
log-probs at the window positions three ways:
  batch  - as hawthorne_readout.py (transport on the selected positions, fp32, unembed casts to bf16)
  single - as jmr.lens.readout.readout (transport on all positions in chunks, same unembed)
  fp64   - transport, final RMSNorm and LM head all in float64 (no bf16 cast)
and report max/median |difference| per pair. If |batch - single| is of the same size
as |batch - fp64| and |single - fp64|, the gap is the library's bf16 cast, not a bug.
Nothing is written except a small JSON of summary numbers.

Usage (lens env): JMR_MODEL_PROFILE=qwen3-14b python scripts/lens/fp64_check.py \
  --runs data/hawthorne/q14/runs.jsonl --select data/hawthorne/q14/lens_select.jsonl --n 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from hawthorne_readout import load_tracked  # noqa: E402

from jmr.lens import readout as single  # noqa: E402
from jmr.lens.batch import BatchReader  # noqa: E402
from jmr.lens.load import lens_file, load_lens, load_model  # noqa: E402


def unembed64(lens_model, h):
    norm, head = lens_model._final_norm, lens_model._lm_head
    w = norm.weight.double().to(h.device)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w
    return x @ head.weight.double().to(h.device).T


def stats(d):
    return {"max": float(d.max()), "median": float(np.median(d)), "p99": float(np.quantile(d, 0.99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--tokens", default="configs/evalaware_tokens.json")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="data/hawthorne/q14/fp64_check.json")
    a = ap.parse_args()
    sel = [json.loads(l) for l in open(a.select)][: a.n]
    want = {s["run_id"] for s in sel}
    runs = {}
    for l in open(a.runs):
        r = json.loads(l)
        if r["run_id"] in want:
            runs[r["run_id"]] = r
    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    reader = BatchReader(lens_model, lens, tok)
    tracked, _ = load_tracked(a.tokens, tok)
    tid = [t["id"] for t in tracked]
    tt = torch.tensor(tid, device=reader.device)
    print("norm class", type(lens_model._final_norm).__name__, "head dtype", lens_model._lm_head.weight.dtype, flush=True)
    res = []
    for s in sel:
        r = runs[s["run_id"]]
        P, C = r["prompt_token_ids"], r["completion_token_ids"]
        ids = P + C
        W = list(range(int(s["lo"]), int(s["hi"])))
        full, _ = reader._forward(ids, [len(P) - 1])
        ref = single.readout(lens_model, {l: full[l] for l in reader.layers}, lens=lens, top_k=5,
                             token_ids=tid, chunk=reader.chunk, dtype=np.float32)["token_logprobs"][:, W]
        b, f = [], []
        with torch.no_grad():
            for layer in reader.layers:
                bl = []
                for start in range(0, len(W), reader.chunk):
                    sub = W[start:start + reader.chunk]
                    h = full[layer][torch.tensor(sub, device=reader.device)].float()
                    hb = reader.lens.transport(h, layer)
                    bl.append(torch.log_softmax(lens_model.unembed(hb).float(), -1)[:, tt].cpu().numpy())
                    h64 = h.double() @ reader.lens.jacobians[layer].double().to(h.device).T
                    f.append(torch.log_softmax(unembed64(lens_model, h64), -1)[:, tt].cpu().numpy())
                b.append(np.concatenate(bl))
        b = np.stack(b)
        f = np.concatenate(f).reshape(b.shape)
        row = {"run_id": s["run_id"], "n_pos": len(W),
               "batch_vs_single": stats(np.abs(b - ref)), "batch_vs_fp64": stats(np.abs(b - f)),
               "single_vs_fp64": stats(np.abs(ref - f))}
        print(json.dumps(row), flush=True)
        res.append(row)
        del full
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
