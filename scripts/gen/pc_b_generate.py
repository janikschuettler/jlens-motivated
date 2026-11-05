"""probe-control test (b): generate Qwen3-14B (thinking) runs on the templated prompt versions (gen env, vLLM).
  JMR_MODEL_PROFILE=qwen3-14b uv run --project envs/gen python scripts/gen/pc_b_generate.py \
      --versions data/probe_control/b/versions.jsonl --out data/probe_control/b/gen --samples 4
Same sampling as the grader-bug runs (configs/sampling.yaml: T 0.6, top-p 0.95, top-k 20, 8,192 new tokens). Resumable."""

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
    ap.add_argument("--versions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=1300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    items = [json.loads(l) for l in open(a.versions)]
    if a.limit:
        items = items[: a.limit]
    path = out / "runs.jsonl"
    done = {json.loads(l)["run_id"] for l in open(path)} if path.exists() else set()
    eng = build_engine()
    mcfg, scfg = eng.model_cfg, eng.sampling_cfg
    print(f"model {mcfg['hf_id']}@{mcfg['revision']} kv_cache_tokens={kv_cache_tokens(eng)}", flush=True)
    reqs = []
    for it in items:
        ids = encode_chat(eng.tokenizer, it["messages"], enable_thinking=mcfg["enable_thinking"])
        for k in range(a.samples):
            run_id = f"pcb/{it['vid']}/{k}"
            if run_id not in done:
                reqs.append((run_id, it, k, ids, run_seed(scfg["base_seed"], run_id)))
    print(f"{len(reqs)} requests ({len(done)} done)", flush=True)
    for c0 in range(0, len(reqs), a.chunk):
        chunk = reqs[c0: c0 + a.chunk]
        t0 = time.time()
        outs = eng.llm.generate([TokensPrompt(prompt_token_ids=r[3]) for r in chunk], [sampling_params(scfg, r[4]) for r in chunk])
        n_tok = 0
        with open(path, "a") as f:
            for (run_id, it, k, ids, seed), o in zip(chunk, outs):
                if list(o.prompt_token_ids) != ids:
                    raise RuntimeError(f"{run_id}: vLLM's prompt token IDs differ from the ones sent")
                c = o.outputs[0]
                tids = list(c.token_ids)
                n_tok += len(tids)
                f.write(json.dumps({"run_id": run_id, "run_set": "pcb", "vid": it["vid"], "kind": it["kind"], "sample_idx": k,
                    "messages": it["messages"], "prompt_token_ids": ids, "completion_token_ids": tids,
                    "completion_logprobs": [st[t].logprob for st, t in zip(c.logprobs, tids)], "completion_text": c.text,
                    "finish_reason": c.finish_reason, "seed": seed, "model": {"hf_id": mcfg["hf_id"], "revision": mcfg["revision"]},
                    "tokenizer_fingerprint": eng.tokenizer_fingerprint, "sampling": dict(scfg), "enable_thinking": mcfg["enable_thinking"]}) + "\n")
        print(json.dumps({"n": len(chunk), "elapsed_s": round(time.time() - t0, 1), "new_tokens": n_tok}), flush=True)


if __name__ == "__main__":
    main()
