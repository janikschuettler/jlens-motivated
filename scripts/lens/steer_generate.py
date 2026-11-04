"""Part 2 of the steering test (lens env): generate with a steering vector added at
block GEN_LAYER at every position whose output produces a generated token (the last
prompt position and every new position; jmr.lens.steering, cols="last").

Two steps:
  --build pilot|main   CPU: write the job list (jobs_<name>.jsonl) from saved prompts.
  --run <jobs file>    GPU: generate. Own decode loop (HF transformers forward + DynamicCache;
                       left padding with explicit position IDs; finished rows are dropped
                       from the batch), sampling T 0.6, top-k 20, top-p 0.95
                       (configs/sampling.yaml), at most GEN_MAX_NEW new tokens. Runs that hit
                       the cap get a forced answer: the steered model reads
                       prompt + completion + "</think>\\n\\nAnswer:" (without the "</think>" if the
                       completion already closed its thinking) and the highest letter
                       (log-sum-exp over " X", "X", "(X") is recorded as forced_answer.

Prompt token IDs come from saved gen-env runs (sample 0 of the item and arm); nothing
is re-tokenized from text. Output rows keep the source run's fields (arm, arm_spec, w2,
messages) so jmr.explore.w2_labels labels them as in wave 2.

  uv run --project envs/lens python scripts/lens/steer_generate.py --build pilot --out data/steer/gen
  uv run --project envs/lens python scripts/lens/steer_generate.py --run data/steer/gen/jobs_pilot.jsonl \
      --dirs data/steer/directions.pt --out data/steer/gen
"""

import argparse
import os
import json
import time
import zlib
from pathlib import Path

from jmr import config, steer

KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2
KV_BUDGET = 20e9
MAX_ROWS = 40
DROP_EVERY = int(os.environ.get("STEER_DROP_EVERY", "32"))  # env override for smoke tests only


def source_prompts(arm_dir: str, arm: str, items: list[str]) -> dict[str, dict]:
    """Sample-0 (lowest sample_idx) saved run per item for the arm, and the item's mean
    completion length in that arm (for length-sorted batching)."""
    best, lens = {}, {}
    for line in open(Path(arm_dir) / "runs.jsonl"):
        r = json.loads(line)
        if r.get("arm") != arm or r["question_id"] not in items:
            continue
        lens.setdefault(r["question_id"], []).append(len(r["completion_token_ids"]))
        if r["question_id"] not in best or r["sample_idx"] < best[r["question_id"]]["sample_idx"]:
            best[r["question_id"]] = r
    for q, r in best.items():
        r["_exp_len"] = sum(lens[q]) / len(lens[q])
    return best


def job(group, src, dtype, alpha, k_rand, priority, sample=0, letter=None, sign=1):
    q = src["question_id"]
    jid = f"{group}/{dtype}/{alpha:g}/{q}/{sample}"
    return {"job_id": jid, "group": group, "question_id": q, "source_run_id": src["run_id"],
            "correct": src["correct_letter"], "hinted": src["hint_letter"], "dir": dtype, "letter": letter,
            "sign": sign, "random_k": k_rand, "layer": steer.GEN_LAYER, "alpha": alpha, "sample": sample,
            "priority": priority, "exp_len": src["_exp_len"], "seed": zlib.crc32(jid.encode())}


def build(name: str, out: Path, alphas: tuple[float, float] | None):
    pool = [json.loads(l) for l in open(steer.POOL_FILE)]
    known = steer.known_ids(pool)
    jobs = []
    if name == "pilot":
        items = known[steer.N_EXCLUDE_FROM_PROBE:steer.N_EXCLUDE_FROM_PROBE + 16]
        srcs = source_prompts("data/explore/gap_factorial/screen", "think_qwen_unhinted_ours", items)
        if len(srcs) != 16:
            raise ValueError(f"pilot: {len(srcs)} of 16 prompts found (rule 1)")
        for q, s in srcs.items():
            s["hint_letter"] = None
            for a in steer.GEN_PILOT_ALPHAS:
                jobs.append(job("pilot", s, "random", a, zlib.crc32(q.encode()) % steer.N_RANDOM, 0))
    else:
        lo, hi = alphas
        items = [json.loads(l)["id"] for l in open(steer.ITEMS_FILE)]
        for g, arm in steer.GEN_ARMS.items():
            d = steer.SOURCES[g][0]
            srcs = source_prompts(d, arm, items)
            if len(srcs) != len(items):
                raise ValueError(f"{g}: {len(srcs)} of {len(items)} prompts found (rule 1)")
            for q, s in srcs.items():
                kr = zlib.crc32(q.encode()) % steer.N_RANDOM
                letter, sign = steer.target(g, s["correct_letter"], s["hint_letter"], True)
                for sample in (0, 1):
                    pr = 0 if sample == 0 else 2
                    jobs.append(job(g, s, "none", 0.0, kr, pr, sample))
                    for t in steer.DIR_TYPES:
                        for a, p_ in ((hi, pr), (lo, pr + 1)):
                            jobs.append(job(g, s, t, a, kr, p_, sample, letter if t != "random" else None,
                                            sign if t != "random" else 1))
    path = out / f"jobs_{name}.jsonl"
    path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs → {path}")


def run(jobs_path: Path, dirs_path: str, out: Path, max_priority: int, device: str = "cuda",
        max_new: int = steer.GEN_MAX_NEW, limit: int = 0, extra_eos: tuple[int, ...] = ()):
    import torch
    from transformers import DynamicCache

    from jmr.lens.load import load_model
    from jmr.lens.steering import Steerer
    from jmr.tokens import fingerprint

    torch.set_grad_enabled(False)
    jobs = [json.loads(l) for l in open(jobs_path)]
    jobs = [j for j in jobs if j["priority"] <= max_priority]
    if limit:
        jobs = jobs[:limit]
    name = jobs_path.stem.replace("jobs_", "")
    res_path = out / f"runs_{name}.jsonl"
    done = set()
    if res_path.exists():
        done = {json.loads(l)["job_id"] for l in open(res_path)}
    todo = [j for j in jobs if j["job_id"] not in done]
    # priority first, then long expected runs together
    todo.sort(key=lambda j: (j["priority"], -j["exp_len"], j["job_id"]))
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to do", flush=True)
    config.snapshot(out)

    srcs = {}
    for g, (d, arms) in list(steer.SOURCES.items()) + [("pilot", ("data/explore/gap_factorial/screen", []))]:
        need = {j["source_run_id"] for j in todo if j["group"] == g}
        if not need:
            continue
        for line in open(Path(d) / "runs.jsonl"):
            r = json.loads(line)
            if r["run_id"] in need:
                srcs[r["run_id"]] = r

    m = config.load("model")
    s = config.load("sampling")
    hf_model, tok, _ = load_model(device)
    fp = fingerprint(tok)
    for r in srcs.values():
        if r["tokenizer_fingerprint"] != fp or r["model"]["revision"] != m["revision"]:
            raise ValueError(f"{r['run_id']}: tokenizer or revision mismatch (rule 1)")
    dev = hf_model.device
    D = torch.load(dirs_path)
    L = steer.GEN_LAYER
    r_l = float(D["r"][L])
    dirs = {t: D["dirs"][L][t].to(dev) for t in steer.DIR_TYPES}
    tr = config.load("lens")["tracked_tokens"]["ids"]
    letter_ids = torch.tensor([[tr[x][f] for f in ("space", "bare", "paren")] for x in steer.LETTERS], device=dev)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} | set(extra_eos)
    think_end = tok.convert_tokens_to_ids("</think>")
    sfx_tail = tok.encode("\n\nAnswer:", add_special_tokens=False)
    if tok.decode(sfx_tail) != "\n\nAnswer:":
        raise ValueError("forced-answer suffix does not round-trip (rule 1)")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    st = Steerer(hf_model, [L])
    T, top_k, top_p = s["temperature"], s["top_k"], s["top_p"]

    def vec(j):
        if j["dir"] == "none" or j["alpha"] == 0:
            return torch.zeros(hf_model.config.hidden_size, device=dev)
        if j["dir"] == "random":
            v = dirs["random"][j["random_k"]] * j["sign"]
        else:
            v = j["sign"] * dirs[j["dir"]][steer.LETTERS.index(j["letter"])]
        return j["alpha"] * r_l * v

    def sample_next(logits, gen):
        x = logits.float() / T
        kth = torch.topk(x, top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
        sp, si = torch.sort(x, descending=True, dim=-1)
        pr = torch.softmax(sp, -1)
        cum = pr.cumsum(-1)
        drop = (cum - pr) > top_p  # keep the smallest prefix with mass >= top_p
        sp = sp.masked_fill(drop, float("-inf"))
        choice = torch.multinomial(torch.softmax(sp, -1), 1, generator=gen)
        return si.gather(-1, choice).squeeze(-1)

    t_all, n_tok = time.time(), 0
    with open(res_path, "a") as fout:
        i = 0
        while i < len(todo):
            longest = max(len(srcs[j["source_run_id"]]["prompt_token_ids"]) for j in todo[i:i + MAX_ROWS])
            B = int(max(1, min(MAX_ROWS, KV_BUDGET // ((longest + max_new + 8) * KV_BYTES_PER_TOKEN))))
            batch = todo[i:i + B]
            i += B
            B = len(batch)
            prompts = [srcs[j["source_run_id"]]["prompt_token_ids"] for j in batch]
            Pm = max(len(p) for p in prompts)
            ids = torch.full((B, Pm), pad, device=dev)
            mask = torch.zeros((B, Pm), dtype=torch.long, device=dev)
            for b, p in enumerate(prompts):
                ids[b, Pm - len(p):] = torch.tensor(p, device=dev)
                mask[b, Pm - len(p):] = 1
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            V = torch.stack([vec(j) for j in batch])
            gen = torch.Generator(device=dev).manual_seed(batch[0]["seed"])
            cache = DynamicCache()
            t0 = time.time()
            st.set({L: V})
            o = hf_model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
            active = list(range(B))  # row index in the current (shrinking) batch → original row
            outs = [[] for _ in range(B)]
            lps = [[] for _ in range(B)]
            finished = [False] * B
            last_pos = pos[:, -1].clone()
            logits = o.logits[:, -1]
            for step in range(max_new):
                nxt = sample_next(logits, gen)
                lp = torch.log_softmax(logits.float(), -1).gather(-1, nxt[:, None]).squeeze(-1)
                nl, ll = nxt.tolist(), lp.tolist()
                for k, b in enumerate(active):
                    if not finished[b]:
                        outs[b].append(nl[k])
                        lps[b].append(ll[k])
                        if nl[k] in eos:
                            finished[b] = True
                if all(finished[b] for b in active):
                    break
                if step % DROP_EVERY == DROP_EVERY - 1 and any(finished[b] for b in active):
                    keep = [k for k, b in enumerate(active) if not finished[b]]
                    kt = torch.tensor(keep, device=dev)
                    cache.batch_select_indices(kt)
                    mask, last_pos, nxt, V = mask[kt], last_pos[kt], nxt[kt], V[kt]
                    active = [active[k] for k in keep]
                    st.set({L: V})
                mask = torch.cat([mask, torch.ones((len(active), 1), dtype=mask.dtype, device=dev)], -1)
                last_pos = last_pos + 1
                o = hf_model(input_ids=nxt[:, None], attention_mask=mask, position_ids=last_pos[:, None],
                             past_key_values=cache, use_cache=True)
                logits = o.logits[:, -1]
            del cache, o
            st.clear()
            dt = time.time() - t0
            for b, j in enumerate(batch):
                src = srcs[j["source_run_id"]]
                comp = outs[b]
                fin = "stop" if finished[b] else "length"
                forced = None
                if fin == "length":
                    P = len(src["prompt_token_ids"])
                    sfx = sfx_tail if think_end in comp else [think_end] + sfx_tail
                    full = src["prompt_token_ids"] + comp + sfx
                    cols = torch.zeros((1, len(full)), dtype=torch.bool, device=dev)
                    cols[0, P - 1:] = True
                    st.set({L: V_row(batch, b, vec)}, cols)
                    fo = hf_model(input_ids=torch.tensor([full], device=dev), logits_to_keep=1)
                    st.clear()
                    lpf = torch.log_softmax(fo.logits[0, -1].float(), -1)
                    let = torch.logsumexp(lpf[letter_ids], -1)
                    forced = {"letter": steer.LETTERS[int(let.argmax())],
                              "letter_lp": [round(float(x), 4) for x in let],
                              "letter_mass": float(let.exp().sum())}
                row = {k: v for k, v in src.items() if not k.startswith("_")}
                row.update({"run_id": f"steer/{j['job_id']}", "run_set": "steer_gen", "completion_token_ids": comp,
                            "completion_logprobs": lps[b], "completion_text": tok.decode(comp),
                            "finish_reason": fin, "seed": j["seed"], "sample_idx": j["sample"],
                            "sampling": {"temperature": T, "top_p": top_p, "top_k": top_k,
                                         "max_tokens": max_new, "engine": "hf_custom_loop"},
                            "model": {"hf_id": m["hf_id"], "revision": m["revision"]},
                            "steer": {**j, "r_l": r_l, "batch_seed": batch[0]["seed"], "forced": forced},
                            "job_id": j["job_id"], "source_run_id": j["source_run_id"]})
                fout.write(json.dumps(row) + "\n")
                n_tok += len(comp)
            fout.flush()
            print(f"batch {i}/{len(todo)} rows {B} {dt:.0f}s max_new {max(len(c) for c in outs)} "
                  f"truncated {sum(not f for f in finished)} tok/s {sum(len(c) for c in outs) / dt:.0f} "
                  f"elapsed {time.time() - t_all:.0f}s", flush=True)


def V_row(batch, b, vec):
    return vec(batch[b])[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", choices=["pilot", "main"])
    ap.add_argument("--alphas", nargs=2, type=float, help="alpha_lo alpha_hi (main build)")
    ap.add_argument("--run")
    ap.add_argument("--dirs", default="data/steer/directions.pt")
    ap.add_argument("--out", default="data/steer/gen")
    ap.add_argument("--max-priority", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=steer.GEN_MAX_NEW)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke-extra-eos", type=int, nargs="*", default=[],
                    help="smoke tests only: extra stop token IDs, to exercise row dropping")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.build:
        build(a.build, out, tuple(a.alphas) if a.alphas else None)
    if a.run:
        run(Path(a.run), a.dirs, out, a.max_priority, a.device, a.max_new, a.limit, tuple(a.smoke_extra_eos))


if __name__ == "__main__":
    main()
