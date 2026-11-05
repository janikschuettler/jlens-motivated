"""gemma27-silent, own-output control (exploratory, addendum G1.4): the model's own final next-token log-probs of the
tracked letter forms at pre_answer and k tokens before, from the real HF forward (hf(...).logits, i.e. final norm + LM
head, soft-capping if the config has one). Batch 1. Also validates the log-prob of the sampled answer token vs vLLM.

  JMR_MODEL_PROFILE=gemma-3-27b uv run --project envs/lens python scripts/lens/gemma_own_output.py \
      --runs data/explore/gemma27/hints/runs.jsonl data/explore/gemma27/subtle/runs.jsonl \
      --select data/explore/gemma27/select.json data/explore/gemma27/select_subtle.json --out data/explore/gemma27/own_output.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jmr import config
from gemma_readout import locate, tracked_ids

KS = (0, 1, 8, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--select", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu-gib", type=float, default=34.0)
    a = ap.parse_args()
    m, lc = config.load("model"), config.load("lens")
    hf = AutoModelForCausalLM.from_pretrained(m["hf_id"], revision=m["revision"], dtype=torch.bfloat16, device_map="auto",
                                              max_memory={0: f"{a.gpu_gib}GiB", "cpu": "800GiB"}).eval()
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    cap = hf.config.get_text_config().final_logit_softcapping
    print("final_logit_softcapping:", cap, flush=True)
    tids, tnames = tracked_ids(tok, lc)
    wanted = set()
    for s in a.select:
        wanted |= set(json.load(open(s)))
    recs = [json.loads(l) for p in a.runs for l in open(p)]
    recs = [r for r in recs if r["run_id"] in wanted]
    if a.limit:
        recs = recs[: a.limit]
    print(len(recs), "runs", flush=True)
    out, flagged, t0 = [], [], time.time()
    dev = next(hf.parameters()).device
    for n_, r in enumerate(recs):
        loc = locate(tok, r)
        if loc is None:
            flagged.append(r["run_id"]); continue
        ids = r["prompt_token_ids"] + r["completion_token_ids"]
        pos = [loc["pre_answer"] - k for k in KS]
        with torch.no_grad():
            lg = hf(input_ids=torch.tensor([ids]).to(dev), use_cache=False).logits[0].float()
            lp = torch.log_softmax(lg[torch.tensor(pos + [loc["pre_answer"]], device=lg.device)], dim=-1)
        tl = lp[:4][:, tids].cpu().numpy()  # [len(KS), 8]
        samp = ids[loc["answer_token"]]
        own = float(lp[4, samp])
        vl = r["completion_logprobs"][loc["answer_token"] - loc["n_prompt"]]
        out.append({"run_id": r["run_id"], "tl": tl.tolist(), "ks": list(KS), "tracked": tnames, "answer_lp_hf": own,
                    "answer_lp_vllm": vl, "answer_letter": loc["answer_letter"]})
        if n_ % 20 == 0:
            print(time.strftime("%H:%M:%S"), n_, len(recs), f"{time.time() - t0:.0f}s", flush=True)
    Path(a.out).write_text(json.dumps({"softcap": cap, "runs": out, "flagged": flagged}))
    d = np.array([abs(o["answer_lp_hf"] - o["answer_lp_vllm"]) for o in out])
    print(f"answer-token |hf - vllm|: mean {d.mean():.4f} median {np.median(d):.4f} max {d.max():.3f}; flagged {len(flagged)}", flush=True)


if __name__ == "__main__":
    main()
