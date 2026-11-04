"""Steering validation of the eval-awareness score (lens env, GPU, HF generate).

Replicates the paper's first validation (Gurnee, Sofroniew et al. 2026, appendix
"Measuring evaluation awareness with the J-lens": steering along a contrastive
"test vs ordinary conversation" direction at -0.1, 0, +0.1 moves the score
monotonically). Our version (exploratory):

- direction d_l (blocks 14-32): mean over the first 50 of the paired items of the
  block-l output at the last prompt token, benchmark framing (redux_bench) minus
  coworkers'-argument framing (redux_casual), from the saved prompt token IDs;
- steering: h_l <- h_l + alpha * d_l / 19 at every block 14-32 and every position
  (the 19 blocks share the push), alpha in {-1, 0, +1};
- prompts: held-out items 50-99 in the home format (redux_home) and 50 OASST prompts,
  1 sample each, up to 1,024 new tokens (T 0.6, top-p 0.95, top-k 20);
- score: E (paper's score) at every response position, read by the *unsteered* model
  from the saved token IDs (primary: does the text become more eval-aware?) and by the
  steered model (secondary).

  uv run --project envs/lens python scripts/lens/evalaware_steer.py --out data/explore/evalaware/steer
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jmr import config
from jmr.lens.batch import read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.lens.readout import residuals

BLOCKS = list(range(14, 33))
READ_LAYERS = list(range(14, 33, 2))


def first_per_prompt(path):
    out = {}
    for r, _ in read_runs_lenient(Path(path)):
        out.setdefault(r.question_id, r)
    return out


class Steer:
    def __init__(self, hf_model, dirs, alpha):
        self.handles = [hf_model.model.layers[l].register_forward_hook(self._hook(alpha * dirs[l] / len(BLOCKS)))
                        for l in BLOCKS]

    @staticmethod
    def _hook(v):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            t.add_(v.to(t.dtype))
            return output
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def score(lens_model, lens, ids, lo, cols_primary, ctrl):
    """E per position >= lo: mean J-lens/logit log-prob of the two primary words minus
    the control mean, averaged over the read layers. Returns [2, n]."""
    res = residuals(lens_model, ids, READ_LAYERS)
    out = np.zeros((2, len(ids) - lo))
    for li, use_j in enumerate((True, False)):
        acc = 0
        for l in READ_LAYERS:
            h = res[l][lo:].float()
            if use_j:
                h = lens.transport(h, l)
            lp = torch.log_softmax(lens_model.unembed(h).float(), -1)
            acc = acc + (lp[:, cols_primary].mean(-1) - lp[:, ctrl].mean(-1))
        out[li] = (acc / len(READ_LAYERS)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="data/explore/evalaware")
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    s = config.load("sampling")
    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    tk = json.loads(Path("configs/evalaware_tokens.json").read_text())["tokens"]
    prim = [t["id"] for t in tk if "eval_primary" in t["groups"]]
    ctrl = [t["id"] for t in tk if t["groups"] == ["control"]]
    bench = first_per_prompt(f"{a.corpus}/redux_bench/runs.jsonl")
    casual = first_per_prompt(f"{a.corpus}/redux_casual/runs.jsonl")
    home = first_per_prompt(f"{a.corpus}/redux_home/runs.jsonl")
    oasst = first_per_prompt(f"{a.corpus}/oasst/runs.jsonl")
    items = list(bench)
    train, test = items[:a.n_train], items[a.n_train:a.n_train + a.n_test]
    dirs = {l: 0 for l in BLOCKS}
    for q in train:
        rb = residuals(lens_model, bench[q].prompt_token_ids, BLOCKS)
        rc = residuals(lens_model, casual[q].prompt_token_ids, BLOCKS)
        for l in BLOCKS:
            dirs[l] = dirs[l] + (rb[l][-1].float() - rc[l][-1].float()) / len(train)
    norms = {l: float(dirs[l].norm()) for l in BLOCKS}
    print("direction norms", norms, flush=True)
    prompts = [("redux_home", q, home[q].prompt_token_ids) for q in test] + \
              [("oasst", q, r.prompt_token_ids) for q, r in list(oasst.items())[:a.n_test]]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    rows = []
    for alpha in a.alphas:
        for b0 in range(0, len(prompts), a.batch):
            batch = prompts[b0:b0 + a.batch]
            L = max(len(p[2]) for p in batch)
            pad = tok.pad_token_id
            x = torch.tensor([[pad] * (L - len(p[2])) + p[2] for p in batch], device=lens_model.input_device)
            att = torch.tensor([[0] * (L - len(p[2])) + [1] * len(p[2]) for p in batch], device=x.device)
            torch.manual_seed(int(1000 * (alpha + 2)) + b0)
            st = Steer(hf_model, dirs, alpha) if alpha != 0 else None
            t0 = time.time()
            gen = hf_model.generate(x, attention_mask=att, do_sample=True, temperature=s["temperature"],
                                    top_p=s["top_p"], top_k=s["top_k"], max_new_tokens=a.max_new,
                                    eos_token_id=eos, pad_token_id=pad)
            if st:
                st.remove()
            for j, (suite, q, ids) in enumerate(batch):
                c = gen[j, L:].tolist()
                for p_, t in enumerate(c):
                    if t in eos:
                        c = c[:p_ + 1]
                        break
                rows.append({"alpha": alpha, "suite": suite, "qid": q, "prompt_token_ids": ids,
                             "completion_token_ids": c})
            print(f"alpha {alpha} batch {b0} {time.time() - t0:.0f}s", flush=True)
    # score: unsteered reading (primary) and steered reading (secondary)
    res = []
    for r in rows:
        ids = r["prompt_token_ids"] + r["completion_token_ids"]
        P = len(r["prompt_token_ids"])
        e_un = score(lens_model, lens, ids, P, prim, ctrl)
        st = Steer(hf_model, dirs, r["alpha"]) if r["alpha"] != 0 else None
        e_st = score(lens_model, lens, ids, P, prim, ctrl)
        if st:
            st.remove()
        text = tok.decode(r["completion_token_ids"])
        res.append({k: r[k] for k in ("alpha", "suite", "qid")} | {
            "n_tokens": len(r["completion_token_ids"]),
            "E_unsteered_read": e_un.mean(1).tolist(), "E_steered_read": e_st.mean(1).tolist(),
            "text_head": text[:300]})
    with open(out / "steer_runs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "steer_scores.json").write_text(json.dumps({"direction_norms": norms, "runs": res}, indent=1))
    for suite in ("redux_home", "oasst"):
        for alpha in a.alphas:
            v = [x for x in res if x["suite"] == suite and x["alpha"] == alpha]
            print(suite, alpha, "unsteered-read J/logit", np.mean([x["E_unsteered_read"] for x in v], 0).round(3),
                  "steered-read", np.mean([x["E_steered_read"] for x in v], 0).round(3), flush=True)


if __name__ == "__main__":
    main()
