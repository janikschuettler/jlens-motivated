"""Check that a GPU machine reproduces another machine's numbers (lens env).

  JMR_MODEL_PROFILE=<profile or unset> uv run --project envs/lens python scripts/lens/machine_check.py \
      --runs <runs.jsonl generated anywhere> --n 4 [--device cpu] [--expect-fingerprint 5528977909577f19]

1. Tokenizer fingerprint of this env (jmr.tokens.fingerprint) against the one recorded in the
   runs (the gen env that made them) and against --expect-fingerprint (another machine's value).
2. Teacher-forces exactly the saved prompt + completion token IDs of the first --n runs (shortest
   first among the first 50, to keep a CPU check quick) and compares the model's log-prob of each
   sampled token with vLLM's saved raw log-prob. Pass = M0.4 tolerance: median |diff| <= 0.05
   nats, p99 <= 0.5, and shifting the alignment by one position makes the median >= 10x worse.
Prints one JSON line with the numbers; exits 1 on any failure (rule 1).
"""

import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jmr import config
from jmr.tokens import fingerprint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect-fingerprint", default=None)
    a = ap.parse_args()
    cfg = config.load("model")
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
    fp = fingerprint(tok)
    runs = []
    with open(a.runs) as f:
        for i, l in enumerate(f):
            if i >= 50:
                break
            runs.append(json.loads(l))
    runs = sorted(runs, key=lambda r: len(r["completion_token_ids"]))[: a.n]
    ok = True
    rec_fp = {r.get("tokenizer_fingerprint") for r in runs}
    rec_rev = {r["model"]["revision"] for r in runs}
    if rec_fp != {fp} or (a.expect_fingerprint and fp != a.expect_fingerprint) or rec_rev != {cfg["revision"]}:
        ok = False
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], revision=cfg["revision"],
                                                 dtype=getattr(torch, cfg["dtype"]), device_map=a.device)
    model.eval()
    diffs, shifted = [], []
    per_run = []
    for r in runs:
        p, c = r["prompt_token_ids"], r["completion_token_ids"]
        ids = torch.tensor([p + c], device=a.device)
        with torch.no_grad():
            h = model.model(input_ids=ids).last_hidden_state[0]  # after the final norm
            lp_rows = []
            for s in range(len(p) - 1, len(p) + len(c) - 1, 512):
                e = min(s + 512, len(p) + len(c) - 1)
                lp_rows.append(torch.log_softmax(model.lm_head(h[s:e]).float(), -1).cpu())
            lp = torch.cat(lp_rows)  # row j = distribution of completion token j
        tgt = torch.tensor(c)
        tf = lp.gather(1, tgt[:, None])[:, 0].numpy()
        v = np.array(r["completion_logprobs"], dtype=np.float64)
        d = np.abs(tf - v)
        diffs.append(d)
        sh = lp[1:].gather(1, tgt[:-1, None])[:, 0].numpy()  # predicts token j from position j+1
        shifted.append(np.abs(sh - v[:-1]))
        per_run.append({"run_id": r["run_id"], "n_tokens": len(c), "median": float(np.median(d)),
                        "p99": float(np.quantile(d, 0.99))})
    d, s = np.concatenate(diffs), np.concatenate(shifted)
    med, p99, med_s = float(np.median(d)), float(np.quantile(d, 0.99)), float(np.median(s))
    passed = med <= 0.05 and p99 <= 0.5 and med_s >= 10 * max(med, 1e-6)
    ok = ok and passed
    print(json.dumps({"model": cfg["hf_id"], "revision": cfg["revision"], "device": a.device,
                      "fingerprint_here": fp, "fingerprint_in_runs": sorted(map(str, rec_fp)),
                      "expected_fingerprint": a.expect_fingerprint, "n_runs": len(runs), "n_tokens": int(d.size),
                      "median_abs_diff": med, "p99_abs_diff": p99, "max_abs_diff": float(d.max()),
                      "shifted_median": med_s, "m0_4_pass": passed, "all_pass": ok, "per_run": per_run}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
