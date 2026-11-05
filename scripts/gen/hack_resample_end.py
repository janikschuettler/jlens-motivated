"""Behavioural ceiling for W (docs/preregistration.md, "W"): resample the
code of hack-organism runs from just after their reasoning. gen env (vLLM + LoRA).

For every selected run (default: usable = finished, with a <thinking> block, group
hidden or no_hack), the prefix is the saved prompt IDs + the saved completion IDs up
to and including the token that completes "</thinking>" (or, with --cut mid, up to
the sentence boundary nearest 50% of the reasoning). From it, --n resamples with the
run's own model and the hack-organism sampling (T 1, top-p 1), budget 2,048 minus the
kept completion tokens, seeds from run_seed(base_seed, f"{run_id}/{cut}/{j}"). Each
resample is labelled with jmr.hack.label_code on the full completion (kept + new).

Out: <out>/resamples.jsonl, one line per source run: run_id, problem_id, group, cut,
kept tokens, the n labels (hack, hack types, finish) and P_hat = share hacking.
New token IDs are kept too (they are the record).

uv run --project envs/gen python scripts/gen/hack_resample_end.py --gen data/hack_wp/gen_disp \
    --out data/hack_wp/resample_end [--n 8] [--cut end|mid]
"""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

from jmr import config, hack
from jmr.gen.generate import run_seed
from jmr.tokens import fingerprint, token_bytes

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hack_generate import adapter_dir  # noqa: E402

END = b"</thinking>"


def cut_index(tb: list[bytes], cut: str) -> int | None:
    """Number of completion tokens to keep."""
    offs = [0]
    for b in tb:
        offs.append(offs[-1] + len(b))
    data = b"".join(tb)
    s = data.find(b"<thinking>")
    e = data.find(END, s) if s >= 0 else -1
    if s < 0 or e < 0:
        return None
    if cut == "end":
        stop = e + len(END)
        return next(i for i in range(len(tb)) if offs[i + 1] >= stop) + 1
    # mid: token boundary right after a sentence end (. ! ? then whitespace) nearest 50% of the reasoning
    cs = s + len(b"<thinking>")
    target = cs + (e - cs) / 2
    best = None
    for i in range(len(tb)):
        if offs[i + 1] <= cs or offs[i + 1] >= e:
            continue
        # keep tokens 0..i: the kept text ends in . ! or ? and the next token starts with whitespace
        if data[offs[i + 1] - 1:offs[i + 1]] in (b".", b"!", b"?") and i + 1 < len(tb) and tb[i + 1][:1].isspace():
            d = abs(offs[i + 1] - target)
            if best is None or d < best[0]:
                best = (d, i + 1)
    return None if best is None else best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cut", default="end", choices=["end", "mid"])
    ap.add_argument("--groups", default="hidden,no_hack")
    ap.add_argument("--run-ids", default=None)
    args = ap.parse_args()

    cfg, mcfg = config.load("hack_organism"), config.load("model")
    gen = Path(args.gen)
    labels = {json.loads(l)["run_id"]: json.loads(l) for l in open(gen / "labels.jsonl")}
    groups = set(args.groups.split(","))
    keep_ids = {l.strip() for l in open(args.run_ids)} if args.run_ids else None
    runs = []
    for line in open(gen / "runs.jsonl"):
        r = json.loads(line)
        lab = labels[r["run_id"]]
        if keep_ids is not None:
            if r["run_id"] not in keep_ids:
                continue
        elif lab["group"] not in groups or r["finish_reason"] != "stop":
            continue
        runs.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    v, s = cfg["vllm"], cfg["sampling"]
    llm = LLM(model=mcfg["hf_id"], revision=mcfg["revision"], tokenizer_revision=mcfg["revision"],
              dtype=mcfg["dtype"], gpu_memory_utilization=v["gpu_memory_utilization"],
              max_model_len=v["max_model_len"], logprobs_mode=v["logprobs_mode"], seed=cfg["base_seed"],
              enable_lora=True, max_lora_rank=v["max_lora_rank"], max_loras=len(cfg["arms"]))
    fp = fingerprint(tok)
    if fp != fingerprint(llm.get_tokenizer()):
        raise RuntimeError("vLLM's tokenizer and AutoTokenizer tokenize differently")
    loras = {}
    reqs, skipped = [], []
    for r in runs:
        if r["tokenizer_fingerprint"] != fp:
            raise ValueError(f"{r['run_id']}: tokenizer fingerprint mismatch")
        arm = r["arm"]
        sub = cfg["arms"][arm]
        if sub is not None and arm not in loras:
            loras[arm] = LoRARequest(arm, len(loras) + 1, adapter_dir(cfg, sub))
        tb = token_bytes(tok, r["completion_token_ids"])
        k = cut_index(tb, args.cut)
        if k is None or k >= len(tb):
            skipped.append(r["run_id"])
            continue
        budget = s["max_tokens"] - k
        prefix = r["prompt_token_ids"] + r["completion_token_ids"][:k]
        for j in range(args.n):
            seed = run_seed(cfg["base_seed"], f"{r['run_id']}/{args.cut}/{j}")
            reqs.append((r, k, j, prefix, seed, budget))
    print(f"{len(runs)} runs, {len(skipped)} without a cut point, {len(reqs)} requests", flush=True)
    params = [SamplingParams(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], min_p=s["min_p"],
                             max_tokens=budget, seed=seed, skip_special_tokens=False)
              for (_, _, _, _, seed, budget) in reqs]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=q[3]) for q in reqs], params,
                        lora_request=[loras.get(q[0]["arm"]) for q in reqs])
    el = time.time() - t0
    per = {}
    for (r, k, j, prefix, seed, budget), o in zip(reqs, outs):
        if list(o.prompt_token_ids) != prefix:
            raise RuntimeError(f"{r['run_id']}: prompt IDs differ from those sent")
        c = o.outputs[0]
        full = r["completion_token_ids"][:k] + list(c.token_ids)
        text = b"".join(token_bytes(tok, full)).decode("utf-8", errors="replace")
        cl = hack.label_code(text)
        e = per.setdefault(r["run_id"], {"run_id": r["run_id"], "problem_id": r["question_id"], "arm": r["arm"],
                                         "group": labels[r["run_id"]]["group"],
                                         "orig_hack": labels[r["run_id"]]["hack"], "cut": args.cut, "kept": k,
                                         "n_completion": len(r["completion_token_ids"]), "samples": []})
        e["samples"].append({"j": j, "seed": seed, "hack": cl.hack, "hack_types": cl.hack_types,
                             "finish": c.finish_reason, "new_token_ids": list(c.token_ids)})
    with open(out / "resamples.jsonl", "w") as f:
        for e in per.values():
            e["p_hat"] = sum(x["hack"] for x in e["samples"]) / len(e["samples"])
            f.write(json.dumps(e) + "\n")
    summ = {"n_runs": len(per), "skipped": skipped, "elapsed_s": round(el, 1), "n_requests": len(reqs)}
    for g in groups:
        ps = [e["p_hat"] for e in per.values() if e["group"] == g]
        if ps:
            summ[f"mean_p_hat_{g}"] = sum(ps) / len(ps)
            summ[f"n_{g}"] = len(ps)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "skipped"}, indent=1))


if __name__ == "__main__":
    main()
