"""Generate Qwen3 (thinking) runs on the Hawthorne test-awareness prompts (gen env, vLLM).

Items come from scripts/explore/hawthorne_build_items.py (their prompts: tool-use "actions"
and harmful-request "triggers", each in a realistic and a test-like version). Saved per run,
as everywhere: prompt and completion token IDs, vLLM raw log-probs of the sampled tokens,
text, model revision, tokenizer fingerprint, sampling, seed. Appends chunk by chunk to
runs.jsonl and skips run IDs already there (resumable).

  JMR_MODEL_PROFILE=qwen3-14b uv run --project envs/gen python scripts/gen/hawthorne_generate.py \
      --items data/hawthorne/items.jsonl --out data/hawthorne/q14 --dataset actions --samples 8

Sampling: configs/sampling.yaml (T 0.6, top-p 0.95, top-k 20, 8,192 new tokens). The paper
decoded greedily (one trace per prompt); we sample so that one prompt gives several runs,
some of which voice test awareness and some not (needed for the within-prompt lens contrast).
"""

import argparse
import json
import time
from pathlib import Path

from vllm import TokensPrompt

from jmr import config
from jmr.gen.generate import build_engine, kv_cache_tokens, run_seed, sampling_params
from jmr.tokens import encode_chat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True, help="actions, triggers or both (comma-separated)")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--start-sample", type=int, default=0)
    ap.add_argument("--variants", default="real,test")
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--limit-pairs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    dsets = a.dataset.split(",")
    variants = a.variants.split(",")
    items = [json.loads(l) for l in open(a.items)]
    items = [it for it in items if it["dataset"] in dsets and it["variant"] in variants]
    if a.limit_pairs:
        keep = sorted({it["pair"] for it in items})[: a.limit_pairs]
        items = [it for it in items if it["pair"] in keep]
    path = out / "runs.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["run_id"] for l in open(path)}

    eng = build_engine()
    mcfg, scfg = eng.model_cfg, eng.sampling_cfg
    print(f"model {mcfg['hf_id']}@{mcfg['revision']} kv_cache_tokens={kv_cache_tokens(eng)} "
          f"fingerprint={eng.tokenizer_fingerprint}", flush=True)
    reqs = []
    for it in items:
        ids = encode_chat(eng.tokenizer, it["messages"], enable_thinking=mcfg["enable_thinking"])
        for k in range(a.start_sample, a.start_sample + a.samples):
            run_id = f"hawthorne/{it['item_id']}/{k}"
            if run_id in done:
                continue
            reqs.append((run_id, it, k, ids, run_seed(scfg["base_seed"], run_id)))
    print(f"{len(reqs)} requests to generate ({len(done)} already done)", flush=True)
    meta = {"t_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_requests": len(reqs), "chunks": []}
    for c0 in range(0, len(reqs), a.chunk):
        chunk = reqs[c0: c0 + a.chunk]
        t0 = time.time()
        outs = eng.llm.generate([TokensPrompt(prompt_token_ids=r[3]) for r in chunk],
                                [sampling_params(scfg, r[4]) for r in chunk])
        el = time.time() - t0
        n_tok = 0
        with open(path, "a") as f:
            for (run_id, it, k, ids, seed), o in zip(chunk, outs):
                if list(o.prompt_token_ids) != ids:
                    raise RuntimeError(f"{run_id}: vLLM's prompt token IDs differ from the ones sent")
                c = o.outputs[0]
                tids = list(c.token_ids)
                n_tok += len(tids)
                row = {"run_id": run_id, "run_set": "hawthorne", "question_id": it["item_id"],
                       "dataset": it["dataset"], "pair": it["pair"], "variant": it["variant"],
                       "condition": it["variant"], "sample_idx": k, "messages": it["messages"],
                       "item": {kk: vv for kk, vv in it.items() if kk != "messages"},
                       "prompt_token_ids": ids, "completion_token_ids": tids,
                       "completion_logprobs": [st[t].logprob for st, t in zip(c.logprobs, tids)],
                       "completion_text": c.text, "finish_reason": c.finish_reason, "seed": seed,
                       "model": {"hf_id": mcfg["hf_id"], "revision": mcfg["revision"]},
                       "tokenizer_fingerprint": eng.tokenizer_fingerprint, "sampling": dict(scfg),
                       "enable_thinking": mcfg["enable_thinking"]}
                f.write(json.dumps(row) + "\n")
        meta["chunks"].append({"n": len(chunk), "elapsed_s": round(el, 1), "new_tokens": n_tok,
                               "tokens_per_s": round(n_tok / el, 1)})
        print(json.dumps(meta["chunks"][-1]), flush=True)
    meta["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(out / "gen_meta.jsonl", "a") as f:
        f.write(json.dumps({"args": vars(a), **meta}) + "\n")


if __name__ == "__main__":
    main()
