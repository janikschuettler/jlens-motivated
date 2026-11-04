"""Eval-awareness ablation on Qwen3-14B, step 2 (gen env, GPU): grader-bug runs under the scored
framing with J-lens directions projected out of the residual stream during generation.

  JMR_MODEL_PROFILE=qwen3-14b HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1 \
  uv run --project envs/gen python scripts/gen/evalaware_ablate14.py --dir data/explore/evalaware_ablate14 --stop-by 07:50

The paper's ablation (Gurnee, Sofroniew et al. 2026; our replication scripts/lens/evalaware_ablate.py)
moved to vLLM so the runs finish in the GPU window: vLLM in eager mode (no torch.compile, no CUDA
graphs), in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), prefix caching off (a cached prefix from
one arm must never be reused in another), with a forward hook on each decoder layer 16-34. vLLM's
Qwen3 layer returns (mlp_out, residual), whose sum is the residual stream after the block (the
HF layer output). The hook replaces it by (h - Q Q^T h, 0) with h = mlp_out + residual in fp32,
Q from bases.pt (scripts/lens/evalaware_ablate14_prep.py), at every token position (prompt and
generated), as in the paper. Before any generation the hook is checked against the HF reference
(the replication's hook on the HF model, ref_logprobs.json): per arm, teacher-forced next-token
log-probs of 3 saved runs (rule 1 on failure; thresholds in CHECK_*).

Prompts: the saved A_V0 prompt token IDs of each item (sources.jsonl), never re-tokenized. Sampling:
the original runs' settings (T 0.6, top-p 0.95, top-k 20, max 8,192 new tokens, raw log-probs).
Seeds: run_seed(0, "evalaware_ablate14/<item>/<k>"), the same for sample k of an item in every
arm (common random numbers; arms differ only through the intervention).
Order (pre-registered): priority 1 none, eval_J, control_J (samples 0-2); priority 2 eval_logit,
random (samples 0-2); priority 3 none, eval_J (samples 3-4). Amendment X (06:22, before any
ablated run was read): priority 4 control_J (samples 3-4), priority 5 none, eval_J, control_J
(samples 5-7). An arm is started only if its estimated time fits before --stop-by; priorities 3
and 5 run only if all of their arms fit.
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

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402

from jmr import config  # noqa: E402
from jmr.gen.generate import run_seed, sampling_params  # noqa: E402
from jmr.tokens import fingerprint  # noqa: E402

BLOCKS = list(range(16, 35))
ARMS = ["none", "eval_J", "control_J", "eval_logit", "random"]
PLAN = [(1, ["none", "eval_J", "control_J"], [0, 1, 2]),
        (2, ["eval_logit", "random"], [0, 1, 2]),
        (3, ["none", "eval_J"], [3, 4]),
        # amendment X (06:22, before any ablated run was read): extra samples while time allows
        (4, ["control_J"], [3, 4]),
        (5, ["none", "eval_J", "control_J"], [5, 6, 7])]
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 8192}
# Check thresholds (fixed before the GPU run): vLLM-hook vs HF-hook next-token log-probs, per arm,
# pooled over the 3 reference runs: the M0.4 tolerances (median |d| <= 0.05, p99 <= 0.5 nats), and
# for every ablated arm vLLM must match the ablated HF reference clearly better than the unablated
# one: mean |vLLM arm - HF arm| <= 0.5 x mean |vLLM arm - HF none|. (CPU prep: HF-CPU vs the
# original vLLM runs differ by 0.013-0.018 nats on average; the ablations move log-probs by
# 0.06-0.42.) Projection: max over tokens of |Q^T h_new| / |h| <= 1e-4, and the hook must fire.
CHECK_MEDIAN, CHECK_P99, CHECK_MATCH_RATIO, CHECK_PROJ = 0.05, 0.5, 0.5, 1e-4


def install_hooks(model, bases: dict):
    """Forward hooks on decoder layers BLOCKS; state lives on the model (in-process vLLM)."""
    dev = next(model.parameters()).device
    state = {"arm": None, "Q": {k: {l: q.to(dev) for l, q in v.items()} for k, v in bases.items()},
             "check": False, "proj": [], "calls": 0}
    model._jmr_abl = state
    found = [mod for name, mod in model.named_modules()
             if name.endswith("model.layers") and isinstance(mod, torch.nn.ModuleList)]
    if len(found) != 1 or len(found[0]) != 40:
        raise RuntimeError(f"expected one 40-layer decoder ModuleList, found {[len(x) for x in found]} (rule 1)")
    layers = found[0]
    if type(layers[BLOCKS[0]]).__name__ != "Qwen3DecoderLayer":
        raise RuntimeError(f"unexpected layer type {type(layers[BLOCKS[0]]).__name__} (rule 1)")

    def mk(l):
        def hook(mod, inp, out):
            arm = state["arm"]
            if arm is None:
                return None
            q = state["Q"][arm][l]
            hs, res = out
            h = hs.float() + res.float()
            h = h - (h @ q) @ q.T
            if state["check"]:
                state["calls"] += 1
                state["proj"].append(float(((h @ q).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)).max()))
            return h.to(hs.dtype), torch.zeros_like(res)
        return hook

    state["handles"] = [layers[l].register_forward_hook(mk(l)) for l in BLOCKS]
    return len(state["handles"])


def set_arm(llm, arm, check=False):
    def f(model):
        st = model._jmr_abl
        st["arm"] = None if arm == "none" else arm
        st["check"] = check
        st["proj"], st["calls"] = [], 0
        return True
    llm.apply_model(f)


def get_check(llm):
    return llm.apply_model(lambda model: (model._jmr_abl["calls"], max(model._jmr_abl["proj"] or [0.0])))[0]


def hook_check(llm, refs, ref_lp):
    """Teacher-forced next-token log-probs in vLLM per arm vs the HF reference (rule 1)."""
    params = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=0, seed=0)
    v = {}
    proj = {}
    for arm in ARMS:
        set_arm(llm, arm, check=True)
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"] + r["completion_token_ids"])
                             for r in refs], [params] * len(refs), use_tqdm=False)
        proj[arm] = get_check(llm)
        v[arm] = {}
        for r, o in zip(refs, outs):
            ids = r["prompt_token_ids"] + r["completion_token_ids"]
            lp = [o.prompt_logprobs[i][ids[i]].logprob for i in range(1, len(ids))]
            v[arm][r["run_id"]] = lp
    set_arm(llm, "none")
    res, ok = {}, True

    def stats(a, b):
        d = sorted(abs(x - y) for x, y in zip(a, b))
        return {"median": d[len(d) // 2], "p99": d[int(0.99 * len(d))], "mean": sum(d) / len(d), "n": len(d)}

    for arm in ARMS:
        va = [x for r in refs for x in v[arm][r["run_id"]]]
        ha = [x for r in refs for x in ref_lp[r["run_id"]][arm]]
        vn = [x for r in refs for x in v["none"][r["run_id"]]]
        hn = [x for r in refs for x in ref_lp[r["run_id"]]["none"]]
        res[arm] = {"vllm_vs_hf": stats(va, ha), "vllm_arm_vs_vllm_none": stats(va, vn),
                    "vllm_arm_vs_hf_none": stats(va, hn),
                    "hook_calls": proj[arm][0], "max_rel_projection_after": proj[arm][1]}
        s = res[arm]["vllm_vs_hf"]
        if s["median"] > CHECK_MEDIAN or s["p99"] > CHECK_P99:
            ok = False
        if arm != "none":
            if proj[arm][0] == 0 or proj[arm][1] > CHECK_PROJ:
                ok = False
            if s["mean"] > CHECK_MATCH_RATIO * res[arm]["vllm_arm_vs_hf_none"]["mean"]:
                ok = False
        elif proj[arm][0] != 0:
            ok = False
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stop-by", required=True, help="HH:MM (machine local time): start no arm that would end later")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-priority", type=int, default=5)
    ap.add_argument("--limit-items", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    d = Path(a.dir)
    now = dt.datetime.now()
    hh, mm = map(int, a.stop_by.split(":"))
    stop_by = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    m = config.load("model")
    if m["hf_id"] != "Qwen/Qwen3-14B":
        raise SystemExit(f"profile is {m['hf_id']}, expected Qwen/Qwen3-14B")
    sources = [json.loads(l) for l in open(d / "sources.jsonl")]
    if a.limit_items:
        sources = sources[:a.limit_items]
    refs = [json.loads(l) for l in open(d / "refs.jsonl")]
    ref = json.loads((d / "ref_logprobs.json").read_text())
    raw = (d / "bases.pt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ref["bases_sha256"]:
        raise SystemExit("bases.pt differs from the one the HF reference used (rule 1)")
    bases = torch.load(d / "bases.pt")
    if ref["blocks"] != BLOCKS or sorted(bases) != sorted(ARMS[1:]):
        raise SystemExit("blocks or arms differ from the prep step (rule 1)")
    config.snapshot(d)
    t0 = time.time()
    llm = LLM(model=m["hf_id"], revision=m["revision"], tokenizer_revision=m["revision"], dtype=m["dtype"],
              gpu_memory_utilization=a.gpu_mem, max_model_len=12288, logprobs_mode="raw_logprobs",
              enforce_eager=True, enable_prefix_caching=False, seed=0)
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    fp = fingerprint(tok)
    if fp != fingerprint(llm.get_tokenizer()):
        raise SystemExit("vLLM's tokenizer and AutoTokenizer tokenize differently (rule 1)")
    for r in sources + refs:
        if r["tokenizer_fingerprint"] != fp or r["model"]["revision"] != m["revision"]:
            raise SystemExit(f"{r['run_id']}: tokenizer or revision mismatch (rule 1)")
    n_hooks = llm.apply_model(lambda model: install_hooks(model, bases))[0]
    cache = llm.llm_engine.vllm_config.cache_config
    meta = {"started": now.isoformat(timespec="seconds"), "load_s": round(time.time() - t0), "hooks": n_hooks,
            "kv_cache_tokens": cache.num_gpu_blocks * cache.block_size if cache.num_gpu_blocks else None,
            "engine": "vllm eager, in-process, prefix caching off, forward hooks", "bases_sha256": sha,
            "blocks": BLOCKS, "sampling": SAMPLING, "model": m, "gpu_mem_utilization": a.gpu_mem}
    print(json.dumps(meta), flush=True)
    ok, chk = hook_check(llm, refs, ref["logprobs"])
    meta["hook_check"] = {"pass": ok, "arms": chk}
    (d / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    print("hook check:", "PASS" if ok else "FAIL", json.dumps(chk), flush=True)
    if not ok:
        raise SystemExit("hook check failed (rule 1): see gen_meta.json")

    path = d / "runs.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["run_id"] for l in open(path)}
    rate = None  # seconds per run, measured
    stats = []
    for prio, arms, ks in PLAN:
        if prio > a.max_priority:
            break
        todo = {arm: [(s, k) for s in sources for k in ks
                      if f"evalaware_ablate14/{arm}/{s['question_id']}/{k}" not in done] for arm in arms}
        need = sum(len(v) for v in todo.values())
        if prio in (3, 5) and rate is not None and dt.datetime.now() + dt.timedelta(seconds=rate * need) > stop_by:
            print(f"priority {prio}: {need} runs need ~{rate * need / 60:.0f} min, past {a.stop_by}; stopping",
                  flush=True)
            meta.setdefault("skipped", []).append({"priority": prio, "arms": arms, "runs": need})
            break
        for arm in arms:
            jobs = todo[arm]
            if not jobs:
                continue
            if rate is not None and dt.datetime.now() + dt.timedelta(seconds=rate * len(jobs)) > stop_by:
                print(f"{arm} p{prio}: ~{rate * len(jobs) / 60:.0f} min, past {a.stop_by}; stopping", flush=True)
                meta.setdefault("skipped", []).append({"priority": prio, "arms": [arm], "runs": len(jobs)})
                break
            set_arm(llm, arm)
            rids = [f"evalaware_ablate14/{arm}/{s['question_id']}/{k}" for s, k in jobs]
            seeds = [run_seed(0, f"evalaware_ablate14/{s['question_id']}/{k}") for s, k in jobs]
            prompts = [TokensPrompt(prompt_token_ids=s["prompt_token_ids"]) for s, _ in jobs]
            params = [sampling_params(SAMPLING, sd) for sd in seeds]
            t1 = time.time()
            outs = llm.generate(prompts, params, use_tqdm=False)
            el = time.time() - t1
            n_tok = 0
            with open(path, "a") as f:
                for (s, k), rid, sd, o in zip(jobs, rids, seeds, outs):
                    if list(o.prompt_token_ids) != s["prompt_token_ids"]:
                        raise SystemExit(f"{rid}: vLLM's prompt IDs differ from the saved ones (rule 1)")
                    c = o.outputs[0]
                    ids = list(c.token_ids)
                    n_tok += len(ids)
                    row = {k2: s[k2] for k2 in ("run_set", "question_id", "dataset", "condition", "hint_letter",
                                                "correct_letter", "letters", "messages", "prompt_token_ids",
                                                "enable_thinking")}
                    row.update({"run_id": rid, "run_set": "evalaware_ablate14", "abl_arm": arm, "priority": prio,
                                "source_run_id": s["run_id"], "completion_token_ids": ids,
                                "completion_logprobs": [st[t].logprob for st, t in zip(c.logprobs, ids)],
                                "completion_text": c.text, "finish_reason": c.finish_reason, "seed": sd,
                                "model": {"hf_id": m["hf_id"], "revision": m["revision"]},
                                "tokenizer_fingerprint": fp, "sampling": SAMPLING, "sample_idx": k,
                                "ablation": {"arm": arm, "blocks": BLOCKS if arm != "none" else [],
                                             "n_dirs": 26 if arm != "none" else 0, "bases_sha256": sha,
                                             "engine": "vllm_eager_hooks"}})
                    f.write(json.dumps(row) + "\n")
            st = {"priority": prio, "arm": arm, "runs": len(jobs), "elapsed_s": round(el), "tokens": n_tok,
                  "tok_per_s": round(n_tok / el), "capped": sum(o.outputs[0].finish_reason == "length" for o in outs),
                  "finished_at": dt.datetime.now().isoformat(timespec="seconds")}
            stats.append(st)
            rate = sum(x["elapsed_s"] for x in stats) / sum(x["runs"] for x in stats)
            print(json.dumps(st), flush=True)
            meta["arms_done"] = stats
            (d / "gen_meta.json").write_text(json.dumps(meta, indent=1))
        else:
            continue
        break
    meta["ended"] = dt.datetime.now().isoformat(timespec="seconds")
    (d / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    print("done", meta["ended"], flush=True)


if __name__ == "__main__":
    main()
