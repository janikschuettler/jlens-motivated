"""EA27-3: the removal test on Qwen3.6-27B (gen env, GPU + CPU weight offload).

  JMR_MODEL_PROFILE=qwen3.6-27b HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1 \
  uv run --project envs/gen python scripts/gen/ea27_ablate.py --dir data/explore/ea27 \
      --stop-new 07:00 --stop-all 07:45 --cpu-offload-gb 30

Pre-registration: docs/overnight/preregistration.md "EA27-3". The paper's ablation (project the
J-lens directions of 28 eval words out of the residual stream after every workspace block, at every
position) on the plain-chat grader-bug prompt (models-grader arm V0). Six arms run in ONE vLLM
engine: each request carries its arm in SamplingParams.extra_args, and a forward pre-hook on the
first ablated block maps every token of the flattened batch to its request (the runner's
input_batch.req_ids and query_start_loc, in-process engine), so the layer hooks apply each request's
projection to its own tokens only. vLLM eager, no prefix caching (a cached prefix must never cross
arms), no async scheduling, CPU weight offload (UVA). Every hooked block's output (mlp_out, residual)
is replaced by (h - Q Q^T h, 0), h = mlp_out + residual in fp32; arm "none" goes through the same
hook with no projection.

Gate (rule 1, before any generation): in one mixed batch, teacher-forced next-token log-probs of the
3 reference runs under every arm vs the HF-hook reference of the lens env (ref_logprobs.json):
median |d| <= 0.05, p99 <= 0.5 nats; ablated arms: mean |vLLM arm - HF arm| <= 0.5 x mean |vLLM arm -
HF none|; |Q^T h| / |h| <= 1e-4 after the hook; the hook fires.

Order: for sample k = 0, 1, ...: priority A (none, eval_J, eval_logit, control_J) for every item,
interleaved by item; then priority B (eval_Jpp, random). The queue is kept just above the running
batch; no new request after --stop-new; requests still running at --stop-all are aborted and listed.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402

from jmr import config  # noqa: E402
from jmr.gen.generate import run_seed  # noqa: E402
from jmr.tokens import fingerprint  # noqa: E402

BLOCKS = list(range(25, 58))
ARMS = ["none", "eval_J", "eval_logit", "control_J", "eval_Jpp", "random"]
PRIO = [("A", ["none", "eval_J", "eval_logit", "control_J"]), ("B", ["eval_Jpp", "random"])]
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 4096}
CHECK_MEDIAN, CHECK_P99, CHECK_MATCH_RATIO, CHECK_PROJ = 0.05, 0.5, 0.5, 1e-4
MAX_SAMPLES = 8


def find_runner(llm):
    ex = llm.llm_engine.model_executor
    return ex.driver_worker.worker.model_runner


def install_hooks(llm, bases):
    runner = find_runner(llm)
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    found = [mod for name, mod in model.named_modules()
             if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == 64]
    if len(found) != 1:
        raise SystemExit(f"expected one 64-layer decoder ModuleList, found {len(found)} (rule 1)")
    layers = found[0]
    if type(layers[BLOCKS[0]]).__name__ != "Qwen3_5DecoderLayer":
        raise SystemExit(f"unexpected layer type {type(layers[BLOCKS[0]]).__name__} (rule 1)")
    dev = torch.device("cuda")
    st = {"Q": {a: {l: q.to(dev, torch.float32) for l, q in bases[a].items()} for a in bases},
          "groups": [], "check": False, "proj": {}, "calls": {}}

    def pre(mod, args, kwargs=None):
        ib = runner.input_batch
        n = ib.num_reqs
        qsl = runner.query_start_loc.np[:n + 1].copy()
        arms = []
        for rid in ib.req_ids[:n]:
            sp = runner.requests[rid].sampling_params
            arms.append((sp.extra_args or {}).get("arm", "none") if sp is not None else "none")
        groups = {}
        for i, a in enumerate(arms):
            if a != "none":
                groups.setdefault(a, []).append(np.arange(qsl[i], qsl[i + 1]))
        st["groups"] = [(a, torch.from_numpy(np.concatenate(v)).to(dev)) for a, v in groups.items()]
        return None

    def mk(l):
        def hook(mod, inp, out):
            hs, res = out
            h = hs.float() + res.float()
            for a, idx in st["groups"]:
                q = st["Q"][a][l]
                x = h[idx]
                x = x - (x @ q) @ q.T
                h[idx] = x
                if st["check"]:
                    st["calls"][a] = st["calls"].get(a, 0) + 1
                    r = float(((x @ q).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max())
                    st["proj"][a] = max(st["proj"].get(a, 0.0), r)
            return h.to(hs.dtype), torch.zeros_like(res)
        return hook

    handles = [layers[BLOCKS[0]].register_forward_pre_hook(pre)]
    handles += [layers[l].register_forward_hook(mk(l)) for l in BLOCKS]
    return st, len(handles)


def params(arm, seed, **kw):
    s = dict(SAMPLING)
    s.update(kw)
    return SamplingParams(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], min_p=s["min_p"],
                          max_tokens=s["max_tokens"], seed=seed, logprobs=s.get("logprobs", 0),
                          prompt_logprobs=s.get("prompt_logprobs"), extra_args={"arm": arm})


def gate(llm, st, refs, ref_lp):
    st["check"], st["proj"], st["calls"] = True, {}, {}
    jobs = [(arm, r) for r in refs for arm in ARMS]  # mixed batch: every arm together
    ps = [params(arm, 0, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1, prompt_logprobs=0, logprobs=None)
          for arm, _ in jobs]
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"] + r["completion_token_ids"])
                         for _, r in jobs], ps, use_tqdm=False)
    st["check"] = False
    v = {}
    for (arm, r), o in zip(jobs, outs):
        ids = r["prompt_token_ids"] + r["completion_token_ids"]
        v.setdefault(arm, {})[r["run_id"]] = [o.prompt_logprobs[i][ids[i]].logprob for i in range(1, len(ids))]

    def stats(a, b):
        d = np.sort(np.abs(np.asarray(a) - np.asarray(b)))
        return {"median": float(np.median(d)), "p99": float(d[int(0.99 * len(d))]), "mean": float(d.mean()),
                "n": int(len(d))}

    res, ok = {}, True
    for arm in ARMS:
        va = [x for r in refs for x in v[arm][r["run_id"]]]
        ha = [x for r in refs for x in ref_lp[r["run_id"]][arm]]
        hn = [x for r in refs for x in ref_lp[r["run_id"]]["none"]]
        vn = [x for r in refs for x in v["none"][r["run_id"]]]
        res[arm] = {"vllm_vs_hf": stats(va, ha), "vllm_arm_vs_hf_none": stats(va, hn),
                    "vllm_arm_vs_vllm_none": stats(va, vn), "hook_calls": st["calls"].get(arm, 0),
                    "max_rel_projection_after": st["proj"].get(arm, 0.0)}
        s = res[arm]["vllm_vs_hf"]
        if s["median"] > CHECK_MEDIAN or s["p99"] > CHECK_P99:
            ok = False
        if arm != "none":
            if res[arm]["hook_calls"] == 0 or res[arm]["max_rel_projection_after"] > CHECK_PROJ:
                ok = False
            if s["mean"] > CHECK_MATCH_RATIO * res[arm]["vllm_arm_vs_hf_none"]["mean"]:
                ok = False
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stop-new", required=True, help="HH:MM: no new request after this")
    ap.add_argument("--stop-all", required=True, help="HH:MM: abort what is still running")
    ap.add_argument("--cpu-offload-gb", type=float, default=30)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-batched-tokens", type=int, default=0, help="0 = vLLM default; lowers prompt-logprob memory in the gate")
    ap.add_argument("--max-num-seqs", type=int, default=0, help="0 = from the KV cache size")
    ap.add_argument("--limit-items", type=int, default=0, help="smoke test only")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    now = dt.datetime.now()

    def at(hhmm):
        hh, mm = map(int, hhmm.split(":"))
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return t if t > now else t + dt.timedelta(days=1)

    stop_new, stop_all = at(a.stop_new), at(a.stop_all)
    m = config.load("model")
    if m["hf_id"] != "Qwen/Qwen3.6-27B":
        raise SystemExit(f"profile is {m['hf_id']}, expected Qwen/Qwen3.6-27B")
    items = [json.loads(l) for l in open(d / "items_v0_ids.jsonl")]
    if a.limit_items:
        items = items[:a.limit_items]
    refs = [json.loads(l) for l in open(d / "refs.jsonl")]
    ref = json.loads((d / "ref_logprobs.json").read_text())
    sha = hashlib.sha256((d / "bases.pt").read_bytes()).hexdigest()
    if sha != ref["bases_sha256"] or ref["blocks"] != BLOCKS:
        raise SystemExit("bases.pt or blocks differ from the lens-env prep (rule 1)")
    bases = torch.load(d / "bases.pt")
    if sorted(bases) != sorted(ARMS[1:]):
        raise SystemExit(f"arms in bases.pt {sorted(bases)} (rule 1)")
    gd = d / "ablate"
    gd.mkdir(exist_ok=True)
    config.snapshot(gd)
    t0 = time.time()
    llm = LLM(model=m["hf_id"], revision=m["revision"], tokenizer_revision=m["revision"], dtype=m["dtype"],
              gpu_memory_utilization=a.gpu_mem, max_model_len=8192, logprobs_mode="raw_logprobs",
              enforce_eager=True, enable_prefix_caching=False, async_scheduling=False, seed=0,
              cpu_offload_gb=a.cpu_offload_gb, limit_mm_per_prompt={"image": 0, "video": 0},
              **({"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}),
              **({"max_num_batched_tokens": a.max_batched_tokens} if a.max_batched_tokens else {}))
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    fp = fingerprint(tok)
    if fp != fingerprint(llm.get_tokenizer()):
        raise SystemExit("vLLM's tokenizer and AutoTokenizer differ (rule 1)")
    for r in items + refs:
        if r["tokenizer_fingerprint"] != fp:
            raise SystemExit(f"{r.get('item_id', r.get('run_id'))}: tokenizer fingerprint differs across envs (rule 1)")
    for it in items:
        own = tok.apply_chat_template(it["messages"], add_generation_prompt=True, enable_thinking=True, tokenize=True)
        own = list(own["input_ids"] if hasattr(own, "keys") else own)
        if own != it["prompt_token_ids"]:
            raise SystemExit(f"{it['item_id']}: gen-env chat template gives other IDs than the lens env (rule 1)")
    st, n_hooks = install_hooks(llm, bases)
    cache = llm.llm_engine.vllm_config.cache_config
    sched = llm.llm_engine.vllm_config.scheduler_config
    meta = {"started": now.isoformat(timespec="seconds"), "load_s": round(time.time() - t0), "hooks": n_hooks,
            "num_gpu_blocks": cache.num_gpu_blocks, "block_size": cache.block_size,
            "max_num_seqs": sched.max_num_seqs, "cpu_offload_gb": a.cpu_offload_gb,
            "engine": "vllm 0.33 eager, in-process, prefix caching off, async scheduling off, UVA CPU offload, "
                      "per-request forward hooks", "bases_sha256": sha, "blocks": BLOCKS, "sampling": SAMPLING,
            "model": m, "git": config.git_state(), "n_items": len(items)}
    print(json.dumps(meta), flush=True)
    tg = time.time()
    ok, chk = gate(llm, st, refs, ref["logprobs"])
    meta["gate"] = {"pass": ok, "arms": chk, "seconds": round(time.time() - tg)}
    (gd / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    print("gate:", "PASS" if ok else "FAIL", json.dumps(chk), flush=True)
    if not ok:
        raise SystemExit("gate failed (rule 1): see ablate/gen_meta.json")
    if a.gate_only:
        return

    path = gd / "runs.jsonl"
    done = {json.loads(l)["run_id"] for l in open(path)} if path.exists() else set()
    queue = []
    for k in range(MAX_SAMPLES):
        for prio, arms in PRIO:
            for it in items:
                for arm in arms:
                    rid = f"ea27/{arm}/{it['item_id']}/{k}"
                    if rid not in done:
                        queue.append((prio, arm, it, k, rid))
    eng = llm.llm_engine
    meta_req, alias = {}, {}
    qi, n_done, n_tok, t1 = 0, 0, 0, time.time()
    cap = sched.max_num_seqs
    last_print = time.time()
    with open(path, "a") as f:
        while True:
            nowt = dt.datetime.now()
            if nowt >= stop_all:
                left = list(meta_req)
                if left:
                    eng.abort_request(left)
                meta["aborted"] = [meta_req[r]["rid"] for r in left]
                print(f"stop-all: aborted {len(left)} running requests", flush=True)
                break
            while qi < len(queue) and nowt < stop_new and len(meta_req) < cap + 4:
                prio, arm, it, k, rid = queue[qi]
                seed = run_seed(0, f"ea27/{it['item_id']}/{k}")
                internal = eng.add_request(rid, TokensPrompt(prompt_token_ids=it["prompt_token_ids"]),
                                           params(arm, seed))
                meta_req[rid] = {"rid": rid, "arm": arm, "prio": prio, "it": it, "k": k, "seed": seed,
                                 "submitted": time.time()}
                alias[internal] = rid
                qi += 1
            if not meta_req:
                break
            for o in eng.step():
                if not o.finished:
                    continue
                q = meta_req.pop(o.request_id if o.request_id in meta_req else alias[o.request_id])
                it = q["it"]
                if list(o.prompt_token_ids) != it["prompt_token_ids"]:
                    raise SystemExit(f"{q['rid']}: vLLM's prompt IDs differ from the saved ones (rule 1)")
                c = o.outputs[0]
                ids = list(c.token_ids)
                n_tok += len(ids)
                n_done += 1
                row = {"run_id": q["rid"], "run_set": "ea27_ablate", "abl_arm": q["arm"], "priority": q["prio"],
                       "item_id": it["item_id"], "sample_idx": q["k"], "correct_letter": it["correct"],
                       "hint_letter": it["hinted"], "letters": it["letters"], "messages": it["messages"],
                       "prompt_token_ids": it["prompt_token_ids"], "enable_thinking": True,
                       "completion_token_ids": ids,
                       "completion_logprobs": [lp[t].logprob for lp, t in zip(c.logprobs, ids)] if c.logprobs else None,
                       "completion_text": c.text, "finish_reason": c.finish_reason, "seed": q["seed"],
                       "model": {"hf_id": m["hf_id"], "revision": m["revision"]}, "tokenizer_fingerprint": fp,
                       "sampling": SAMPLING, "seconds": round(time.time() - q["submitted"]),
                       "ablation": {"arm": q["arm"], "blocks": BLOCKS if q["arm"] != "none" else [],
                                    "n_dirs": 28 if q["arm"] != "none" else 0, "bases_sha256": sha}}
                f.write(json.dumps(row) + "\n")
                f.flush()
            if time.time() - last_print > 300:
                el = time.time() - t1
                print(f"{dt.datetime.now():%H:%M} done {n_done}, running {len(meta_req)}, queued {qi}/{len(queue)}, "
                      f"{n_tok / el:.0f} tok/s", flush=True)
                last_print = time.time()
    meta["ended"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["runs_done"], meta["tokens"], meta["gen_seconds"] = n_done, n_tok, round(time.time() - t1)
    meta["submitted"] = qi
    (gd / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    print("done", json.dumps({k: meta[k] for k in ("ended", "runs_done", "tokens", "gen_seconds", "submitted")}),
          flush=True)


if __name__ == "__main__":
    main()
