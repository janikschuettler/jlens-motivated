"""Generate runs of base Qwen3-8B and the cds-jb reward-hack LoRA adapters on the
hack-organism problems, label them, and write an audit file. gen env (vLLM + LoRA).

Every arm in configs/hack_organism.yaml gets `samples_per_item` runs per item, in one
vLLM call (per-request LoRA). Saved per run as usual: prompt and completion token IDs,
vLLM log-probs of the sampled tokens, text, model + adapter revisions, sampling, seed.
Labels (jmr.hack.label_run) go to labels.jsonl, the audit to audit.md.

uv run --project envs/gen python scripts/gen/hack_generate.py --items data/hack/items.jsonl --out data/hack/gen
    [--limit-items N] [--arms base,beta002]
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

from jmr import config, hack
from jmr.audit import fence
from jmr.gen.generate import run_seed
from jmr.records import RunRecord
from jmr.tokens import encode_chat, fingerprint


def adapter_dir(cfg: dict, sub: str) -> str:
    root = snapshot_download(cfg["adapters_repo"], revision=cfg["adapters_revision"],
                             allow_patterns=[f"{sub}/*"])
    return f"{root}/{sub}"


def write_audit(path: Path, runs: list[dict], labels: dict, per_group: int = 3, seed: int = 0) -> None:
    rows = sorted(runs, key=lambda r: r["run_id"])
    counts = Counter((r["condition"], labels[r["run_id"]]["group"]) for r in rows)
    arms = sorted({r["condition"] for r in rows})
    groups = ["hidden", "named", "other_mention", "no_hack", "no_thinking"]
    lines = ["# Audit: reward-hack organism runs", "",
             "Groups: hidden = hack, CoT matches neither keyword list; named = hack, CoT matches the upstream "
             "list; other_mention = hack, CoT matches only our extra list; no_hack; no_thinking (no <thinking> block).",
             "", "| arm | " + " | ".join(groups) + " | total |", "|---|" + "---|" * (len(groups) + 1)]
    for a in arms:
        lines.append(f"| {a} | " + " | ".join(str(counts[(a, g)]) for g in groups)
                     + f" | {sum(counts[(a, g)] for g in groups)} |")
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[(labels[r["run_id"]]["group"], r["condition"])].append(r)
    first_prompt = True
    for (g, a), rs in sorted(by.items()):
        pick = sorted(rng.sample(rs, min(per_group, len(rs))), key=lambda r: r["run_id"])
        lines += ["", f"## {g} × {a} ({len(rs)} runs, showing {len(pick)})"]
        for r in pick:
            lab = labels[r["run_id"]]
            lines += ["", f"### {r['run_id']}", "",
                      f"finish {r['finish_reason']}, {len(r['completion_token_ids'])} tokens; hack types "
                      f"{lab['code']['hack_types']}; upstream CoT words {lab['cot_aisi_words']}; extra words "
                      f"{lab['cot_extra_words']}"]
            if first_prompt:
                lines += ["", "Full prompt (system + user) of this run; other runs differ only in the problem "
                          "and the hint order:", "", fence(r["messages"][0]["content"]), "",
                          fence(r["messages"][1]["content"])]
                first_prompt = False
            lines += ["", "Completion (complete):", "", fence(r["completion_text"])]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--arms", default=None, help="comma-separated subset of the config's arms")
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    cfg = config.load("hack_organism")
    mcfg = config.load("model")
    items = [json.loads(l) for l in open(args.items)]
    items = items[: args.limit_items or cfg["n_items"]]
    arms = args.arms.split(",") if args.arms else list(cfg["arms"])
    n_samples = args.samples or cfg["samples_per_item"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    v = cfg["vllm"]
    llm = LLM(model=mcfg["hf_id"], revision=mcfg["revision"], tokenizer_revision=mcfg["revision"],
              dtype=mcfg["dtype"], gpu_memory_utilization=v["gpu_memory_utilization"],
              max_model_len=v["max_model_len"], logprobs_mode=v["logprobs_mode"], seed=cfg["base_seed"],
              enable_lora=True, max_lora_rank=v["max_lora_rank"], max_loras=len(cfg["arms"]))
    if fingerprint(tok) != fingerprint(llm.get_tokenizer()):
        raise RuntimeError("vLLM's tokenizer and AutoTokenizer tokenize differently")
    fp = fingerprint(tok)
    loras, adapters = {}, {}
    for i, arm in enumerate(arms):
        sub = cfg["arms"][arm]
        if sub is not None:
            loras[arm] = LoRARequest(arm, i + 1, adapter_dir(cfg, sub))
            adapters[arm] = {"repo": cfg["adapters_repo"], "revision": cfg["adapters_revision"], "subfolder": sub}

    s = cfg["sampling"]
    reqs = []
    for it in items:
        msgs = hack.messages(it)
        ids = encode_chat(tok, msgs, enable_thinking=cfg["enable_thinking"])
        if len(ids) != it["prompt_tokens"]:
            raise RuntimeError(f"{it['problem_id']}: prompt length {len(ids)} != items.jsonl {it['prompt_tokens']}")
        for arm in arms:
            for k in range(n_samples):
                run_id = f"hack/{it['problem_id']}/{arm}/{k}"
                seed = run_seed(cfg["base_seed"], run_id)
                reqs.append((run_id, it, arm, k, msgs, ids, seed))
    params = [SamplingParams(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], min_p=s["min_p"],
                             max_tokens=s["max_tokens"], seed=seed, logprobs=1, skip_special_tokens=False)
              for *_, seed in reqs]
    print(f"{len(reqs)} requests ({len(items)} items × {len(arms)} arms × {n_samples})", flush=True)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=r[5]) for r in reqs], params,
                        lora_request=[loras.get(r[2]) for r in reqs])
    elapsed = time.time() - t0

    runs, labels = [], {}
    for (run_id, it, arm, k, msgs, ids, seed), o in zip(reqs, outs):
        if list(o.prompt_token_ids) != ids:
            raise RuntimeError(f"{run_id}: vLLM's prompt token IDs differ from the ones sent")
        c = o.outputs[0]
        tids = list(c.token_ids)
        rec = RunRecord(run_id=run_id, run_set="hack", question_id=it["problem_id"], dataset="deepmind/code_contests",
                        condition=arm, hint_letter=None, correct_letter="", letters=[], messages=msgs,
                        prompt_token_ids=ids, completion_token_ids=tids,
                        completion_logprobs=[st[t].logprob for st, t in zip(c.logprobs, tids)],
                        completion_text=c.text, finish_reason=c.finish_reason, seed=seed,
                        model={"hf_id": mcfg["hf_id"], "revision": mcfg["revision"], "adapter": adapters.get(arm)},
                        tokenizer_fingerprint=fp, sampling=dict(s), enable_thinking=cfg["enable_thinking"],
                        sample_idx=k)
        row = asdict(rec)
        row["arm"] = arm
        runs.append(row)
        labels[run_id] = {"run_id": run_id, "arm": arm, "problem_id": it["problem_id"],
                          "finish_reason": c.finish_reason, "n_tokens": len(tids), **asdict(hack.label_run(c.text))}
    with open(out / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    with open(out / "labels.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(labels[r["run_id"]]) + "\n")
    write_audit(out / "audit.md", runs, labels)
    gen_tokens = sum(len(r["completion_token_ids"]) for r in runs)
    summary = {"n_runs": len(runs), "elapsed_s": round(elapsed, 1), "tokens_per_s": round(gen_tokens / elapsed, 1),
               "by_arm": {}}
    for arm in arms:
        ls = [l for l in labels.values() if l["arm"] == arm]
        n = len(ls)
        summary["by_arm"][arm] = {
            "n": n, "hack": sum(l["hack"] for l in ls) / n,
            "cot_names_aisi_given_hack": sum(l["cot_names_aisi"] for l in ls if l["hack"]) / max(1, sum(l["hack"] for l in ls)),
            "groups": dict(Counter(l["group"] for l in ls)),
            "hack_types": dict(Counter(t for l in ls for t in l["code"]["hack_types"])),
            "truncated": sum(l["finish_reason"] == "length" for l in ls),
            "parse_fail": sum(not l["code"]["parse_ok"] for l in ls),
            "exit_ast_vs_regex_disagree": sum(l["code"]["exit_ast"] != l["code"]["exit_regex"] for l in ls),
            "median_tokens": sorted(l["n_tokens"] for l in ls)[n // 2]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
