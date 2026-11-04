"""Point of no return (Q6): resample saved runs from cuts in their reasoning (gen env, GPU).

One case per process (the model profile and HF cache are environment settings):

  uv run --project envs/gen python scripts/explore/run_q6.py configs/explore/q6_point_of_no_return.yaml --case c3
  JMR_MODEL_PROFILE=qwen3-14b HF_HUB_CACHE=/dev/shm/hf/hub uv run --project envs/gen python \
      scripts/explore/run_q6.py configs/explore/q6_point_of_no_return.yaml --case grader14b
  ... --dry-run          # CPU: select runs, plan cuts, print counts, a token estimate and cut contexts

Writes data/explore/q6_pnr/<case>/{plans.jsonl, resamples.jsonl, meta.json}. Stops at the
first inconsistency (rule 1): wrong model, tokenizer mismatch, prompt IDs changed by vLLM.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from jmr import config
from jmr.explore import q6


def load_case(spec: dict, name: str) -> dict:
    for c in spec["cases"]:
        if c["name"] == name:
            return c
    raise SystemExit(f"no case {name!r}")


def plan_case(spec: dict, case: dict, tokenizer, limit_runs: int | None = None):
    labels = q6.read_jsonl(case["labels"], keep=lambda r: r.get("arm") in case["arms"])
    picked = q6.select_runs(labels, case, seed=spec["seed"])
    groups = {rid: g for g, rids in picked.items() for rid in rids}
    if limit_runs:
        keep = set()
        for g, rids in picked.items():
            keep |= set(rids[:limit_runs])
        groups = {r: g for r, g in groups.items() if r in keep}
    lab_by = {l["run_id"]: l for l in labels if l["run_id"] in groups}
    rows = q6.read_jsonl(case["runs"], keep=lambda r: r["run_id"] in groups)
    if len(rows) != len(groups):
        missing = sorted(set(groups) - {r["run_id"] for r in rows})
        raise SystemExit(f"{case['name']}: {len(missing)} selected runs missing from {case['runs']}: {missing[:3]}")
    plans = []
    for row in sorted(rows, key=lambda r: r["run_id"]):
        rec = q6.record_from_row(row)
        patterns = case.get("mention_patterns")
        p = q6.plan_run(rec, lab_by[rec.run_id], case, groups[rec.run_id], tokenizer, spec["fractions"],
                        mention_patterns=patterns)
        # Pre-flight: the outcome labeller, fed the run's own continuation from its first
        # cut, must give back the run's stored label.
        P = len(rec.prompt_token_ids)
        c0 = p.cuts[0].pos
        own = q6.resample_outcome(p, c0, rec.completion_token_ids[c0 - P:], rec.finish_reason, tokenizer, case)
        if not own["same"]:
            raise SystemExit(f"{rec.run_id}: relabelling the original gives {own}, stored {p.original}")
        plans.append(p)
    return plans, picked


def plan_row(p: q6.Plan) -> dict:
    return {"run_id": p.run_id, "case": p.case, "group": p.group, "question_id": p.question_id,
            "original": p.original, "span": list(p.span), "marks": p.marks,
            "cuts": [asdict(c) for c in p.cuts], "completion_len": len(p.record.completion_token_ids),
            "sampling": p.record.sampling, "model": p.record.model, "orig_seed": p.record.seed}


class FakeLLM:
    """CPU plumbing test only: returns each run's own continuation after the cut."""

    def __init__(self, by_run):
        self.by_run = by_run
        self.tok = None

    def get_tokenizer(self):
        return self.tok

    def generate(self, prompts, params, lora_request=None):
        from types import SimpleNamespace as NS

        out = []
        for pr in prompts:
            ids = pr["prompt_token_ids"]
            for p in self.by_run.values():
                seq = p.record.prompt_token_ids + p.record.completion_token_ids
                if seq[: len(ids)] == ids:
                    rest = seq[len(ids):]
                    lp = [{t: NS(logprob=-0.1)} for t in rest]
                    out.append(NS(prompt_token_ids=ids, outputs=[NS(token_ids=rest, logprobs=lp, text="",
                                                                   finish_reason="stop")]))
                    break
            else:
                raise RuntimeError("fake engine: prompt not found")
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--case", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-runs", type=int, default=None, help="per group (smoke tests)")
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--out-name", default=None, help="subdirectory name (default: the case name)")
    ap.add_argument("--chunks", type=int, default=1, help="split the runs into this many generate calls")
    ap.add_argument("--deadline", default=None, help="HH:MM local time; no new chunk starts after it")
    ap.add_argument("--refine", action="store_true",
                    help="second pass: up to 4 cuts inside each run's commitment segment (needs the first pass)")
    ap.add_argument("--fake-engine", action="store_true",
                    help="CPU plumbing test: every resample returns the run's own continuation (no vLLM)")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    case = load_case(spec, args.case)
    R = args.resamples or spec["resamples"]

    profile = os.environ.get("JMR_MODEL_PROFILE") or None
    if profile != case.get("profile"):
        raise SystemExit(f"case {case['name']} needs JMR_MODEL_PROFILE={case.get('profile')}, got {profile}")
    mcfg = config.load("model")

    from transformers import AutoTokenizer

    from jmr.tokens import fingerprint

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    plans, picked = plan_case(spec, case, tok, args.limit_runs)
    for p in plans:
        m = p.record.model
        if (m["hf_id"], m["revision"]) != (mcfg["hf_id"], mcfg["revision"]):
            raise SystemExit(f"{p.run_id}: generated by {m}, engine model is {mcfg['hf_id']}@{mcfg['revision']}")
        if p.record.tokenizer_fingerprint != fingerprint(tok):
            raise SystemExit(f"{p.run_id}: tokenizer fingerprint differs from the run's")
    sub = args.out_name or case["name"] + ("_fake" if args.fake_engine else "")
    out = config.REPO_ROOT / spec["out_dir"] / sub
    if args.refine:
        from collections import defaultdict

        from jmr.tokens import token_bytes

        cnt = defaultdict(lambda: [0, 0])
        for line in open(out / "resamples.jsonl"):
            r = json.loads(line)
            if r["valid"]:
                cnt[(r["run_id"], r["cut"])][0] += 1
                cnt[(r["run_id"], r["cut"])][1] += bool(r["same"])
        refined = []
        for p in plans:
            ps = [(cnt[(p.run_id, c.idx)][1] / cnt[(p.run_id, c.idx)][0]) if cnt[(p.run_id, c.idx)][0] else None
                  for c in p.cuts]
            seq = p.record.prompt_token_ids + p.record.completion_token_ids
            new = q6.refine_cuts(p, ps, spec["threshold"], token_bytes(tok, seq))
            if new:
                p.cuts = new
                refined.append(p)
        print(f"refine: {len(refined)} of {len(plans)} runs commit after cut 0 with inner boundaries", flush=True)
        plans = refined
        out = out / "refine"
    reqs = [r for p in plans for r in q6.resample_requests(p, R)]
    by_run = {p.run_id: p for p in plans}
    est = sum(max(len(by_run[r["run_id"]].record.completion_token_ids) - (r["cut_pos"] - by_run[r["run_id"]].marks["prompt_len"]), 1)
              for r in reqs)
    print(f"{case['name']}: groups {{{', '.join(f'{g}: {len(v)}' for g, v in picked.items())}}}, "
          f"{len(plans)} runs, {sum(len(p.cuts) for p in plans)} cuts, {len(reqs)} resamples, "
          f"~{est / 1e6:.2f}M new tokens if continuations match the originals' length", flush=True)

    if args.dry_run:
        from jmr.tokens import token_bytes

        for p in plans[:2] + plans[-1:]:
            seq = p.record.prompt_token_ids + p.record.completion_token_ids
            pieces = token_bytes(tok, seq)
            print(f"\n=== {p.run_id} [{p.group}] span {p.span} marks {p.marks} original {p.original}")
            for c in p.cuts:
                before = b"".join(pieces[max(p.span[0], c.pos - 12): c.pos]).decode("utf-8", "replace")
                after = b"".join(pieces[c.pos: c.pos + 6]).decode("utf-8", "replace")
                print(f"  cut {c.idx} f={c.f:.1f} frac={c.frac:.2f} sent={c.sentence} pos={c.pos}: "
                      f"...{before!r} || {after!r}")
        return

    if (out / "resamples.jsonl").exists():
        raise SystemExit(f"{out / 'resamples.jsonl'} exists; use --out-name")
    out.mkdir(parents=True, exist_ok=True)
    v = spec["vllm"]
    if args.fake_engine:
        llm = FakeLLM(by_run)
        llm.tok = tok
        TokensPrompt = dict
        sampling_params = lambda cfg, seed, max_tokens: None  # noqa: E731
    else:
        from vllm import LLM, TokensPrompt

        from jmr.gen.generate import sampling_params
    kw = {}
    lora_req = None
    if case.get("lora") and not args.fake_engine:
        from huggingface_hub import snapshot_download
        from vllm.lora.request import LoRARequest

        L = case["lora"]
        root = snapshot_download(L["repo"], revision=L["revision"], allow_patterns=[f"{L['subfolder']}/*"])
        lora_req = LoRARequest(L["subfolder"], 1, f"{root}/{L['subfolder']}")
        kw = {"enable_lora": True, "max_lora_rank": L["max_lora_rank"], "max_loras": 1}
        for p in plans:
            a = p.record.model.get("adapter") or {}
            if (a.get("repo"), a.get("revision"), a.get("subfolder")) != (L["repo"], L["revision"], L["subfolder"]):
                raise SystemExit(f"{p.run_id}: adapter {a} is not the case's {L}")
    t_load = time.time()
    if not args.fake_engine:
        llm = LLM(model=mcfg["hf_id"], revision=mcfg["revision"], tokenizer_revision=mcfg["revision"],
                  dtype=mcfg["dtype"], gpu_memory_utilization=v["gpu_memory_utilization"],
                  max_model_len=v["max_model_len"], logprobs_mode=v["logprobs_mode"], seed=spec["seed"], **kw)
    if fingerprint(tok) != fingerprint(llm.get_tokenizer()):
        raise RuntimeError("vLLM's tokenizer and AutoTokenizer tokenize differently")
    load_s = time.time() - t_load

    with open(out / "plans.jsonl", "w") as f:
        for p in plans:
            f.write(json.dumps(plan_row(p)) + "\n")
    deadline = None
    if args.deadline:
        hh, mm = map(int, args.deadline.split(":"))
        lt = time.localtime()
        deadline = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    run_ids = [p.run_id for p in plans]
    chunk_runs = [set(run_ids[i:: args.chunks]) for i in range(args.chunks)]
    stats = {"chunks": [], "stopped_at_deadline": False}
    n_done = 0
    for ci, rs in enumerate(chunk_runs):
        if deadline and time.time() > deadline:
            stats["stopped_at_deadline"] = True
            print(f"{time.strftime('%H:%M:%S')} deadline passed; {len(chunk_runs) - ci} chunks not run", flush=True)
            break
        batch = sorted((r for r in reqs if r["run_id"] in rs), key=lambda r: (r["run_id"], r["cut"], r["k"]))
        params = [sampling_params(r["sampling"], r["seed"], max_tokens=r["max_tokens"]) for r in batch]
        t0 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in batch], params,
                            lora_request=[lora_req] * len(batch) if lora_req else None)
        el = time.time() - t0
        rows = []
        for r, o in zip(batch, outs):
            if list(o.prompt_token_ids) != r["prompt_ids"]:
                raise RuntimeError(f"{r['resample_id']}: vLLM's prompt token IDs differ from the ones sent")
            c = o.outputs[0]
            ids = list(c.token_ids)
            p = by_run[r["run_id"]]
            oc = q6.resample_outcome(p, r["cut_pos"], ids, c.finish_reason, tok, case)
            rows.append({"resample_id": r["resample_id"], "run_id": r["run_id"], "case": case["name"],
                         "group": p.group, "question_id": p.question_id, "cut": r["cut"], "cut_pos": r["cut_pos"],
                         "k": r["k"], "seed": r["seed"], "max_tokens": r["max_tokens"],
                         "new_ids": ids, "new_logprobs": [round(s[t].logprob, 5) for s, t in zip(c.logprobs, ids)],
                         "new_text": c.text, "finish_reason": c.finish_reason, **oc})
        with open(out / "resamples.jsonl", "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        gen = sum(len(r["new_ids"]) for r in rows)
        n_done += len(rows)
        stats["chunks"].append({"n": len(rows), "elapsed_s": round(el, 1), "new_tokens": gen,
                                "tokens_per_s": round(gen / el, 1) if el else None})
        valid = sum(r["valid"] for r in rows)
        print(f"{time.strftime('%H:%M:%S')} chunk {ci + 1}/{len(chunk_runs)}: {len(rows)} resamples, "
              f"{gen} tokens in {el:.0f}s ({gen / max(el, 1e-9):.0f} tok/s), valid {valid}, "
              f"same {sum(r.get('same', False) for r in rows)}", flush=True)
    meta = {"case": case, "resamples_per_cut": R, "n_runs": len(plans), "n_requests": len(reqs), "n_done": n_done,
            "groups": {g: len(v) for g, v in picked.items()}, "load_s": round(load_s, 1), "stats": stats,
            "model": mcfg["hf_id"], "revision": mcfg["revision"], "tokenizer_fingerprint": fingerprint(tok),
            "vllm": v, "lora": case.get("lora"), "git": config.git_state(), "fractions": spec["fractions"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_runs", "n_requests", "n_done", "stats")}), flush=True)


if __name__ == "__main__":
    main()
