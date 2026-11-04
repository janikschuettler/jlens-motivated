"""Compact J-lens / logit-lens readout of the eval-awareness tracked tokens (lens env, GPU).

  uv run --project envs/lens python scripts/lens/readout_evalaware.py \
      --runs data/explore/evalaware/oasst/runs.jsonl --out data/readouts/evalaware/oasst [--all-prompt] [--validate 1]

Teacher-forces exactly the saved prompt + completion token IDs (never re-tokenizes).
The full softmax over the vocabulary is computed in memory at selected positions;
only these leave the GPU (CLAUDE.md rule 3):

- log-probs of the tracked tokens in configs/evalaware_tokens.json except the 100
  control words, plus the mean log-prob of the 100 control words (one derived
  column, "control_mean"), per lens x workspace layer x selected position;
- top-25 token IDs and log-probs at the named anchors (and, with --all-prompt, at
  every prompt position);
- the teacher-forced log-prob of every sampled completion token (to compare with vLLM's;
  M0.4 tolerances, checked at the end).

Selected positions: every prompt position when the prompt has at most 1,024 tokens
(or --all-prompt), else the last 64 prompt positions; the named anchors; 64 evenly
spaced reasoning positions and 16 evenly spaced response positions.

The paper's score (Gurnee, Sofroniew et al. 2026, appendix "Measuring evaluation awareness
with the J-lens") is computed from these arrays by scripts/lens/analyze_evalaware.py:
mean(lp " fictional", lp " simulation") - control_mean, averaged over the workspace layers.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from jmr import config
from jmr.lens import readout as single
from jmr.lens.batch import LENSES, BatchReader, check_plan, read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.tokens import fingerprint

MEDIAN_MAX, P99_MAX = 0.05, 0.5  # M0.4 tolerances (roadmap)
FULL_PROMPT_MAX = 1024
N_COT, N_RESP, TAIL = 64, 16, 64


def load_tokens(path, tokenizer):
    cfg = json.loads(Path(path).read_text())
    bad = []
    for t in cfg["tokens"]:
        enc = tokenizer(t["text"], add_special_tokens=False).input_ids
        if enc != [t["id"]] or tokenizer.decode([t["id"]]) != t["text"]:
            bad.append(t["text"])
    if bad:
        raise ValueError(f"tracked tokens don't round-trip in this tokenizer: {bad} (rule 1)")
    tracked = [t for t in cfg["tokens"] if t["groups"] != ["control"]]
    control = [t for t in cfg["tokens"] if t["groups"] == ["control"]]
    if len(control) != 100:
        raise ValueError(f"{len(control)} control words, expected 100")
    return cfg, tracked, control


def positions(plan, L, all_prompt):
    P = plan.prompt_len
    pos = set(range(P)) if (all_prompt or P <= FULL_PROMPT_MAX) else set(range(max(0, P - TAIL), P))
    for name, p in plan.named.items():
        if p >= 0 and not name.startswith("cot_"):
            pos.add(p)
    if plan.cot_span:
        a, b = plan.cot_span
        pos.update(np.linspace(a, b - 1, min(N_COT, b - a)).round().astype(int).tolist())
        r0 = b + 1 if plan.named["think_end"] >= 0 else None
    else:
        r0 = P
    if r0 is not None and r0 < L:
        pos.update(np.linspace(r0, L - 1, min(N_RESP, L - r0)).round().astype(int).tolist())
    return sorted(pos)


@torch.no_grad()
def read_one(reader, record, plan, tracked_ids, control_ids, all_prompt):
    ids, P, L = plan.ids, plan.prompt_len, len(plan.ids)
    S = positions(plan, L, all_prompt)
    top_names = [n for n, p in plan.named.items() if p >= 0 and not n.startswith("cot_")]
    top_pos = [plan.named[n] for n in top_names]
    if all_prompt:
        top_pos = sorted(set(top_pos) | set(range(P)))
    full, _ = reader._forward(ids, [P - 1])
    dev = reader.device
    St = torch.tensor(S, device=dev)
    tt = torch.tensor(tracked_ids, device=dev)
    ct = torch.tensor(control_ids, device=dev)
    sidx = {p: i for i, p in enumerate(S)}
    tp_in_S = [p for p in top_pos if p in sidx]
    extra_top = [p for p in top_pos if p not in sidx]
    nL, K = len(reader.layers), len(tracked_ids)
    lp = np.zeros((len(LENSES), nL, len(S), K + 1), dtype=np.float16)
    top_ids = np.zeros((len(LENSES), nL, len(top_pos), reader.top_k), dtype=np.int32)
    top_lp = np.zeros((len(LENSES), nL, len(top_pos), reader.top_k), dtype=np.float16)
    order = {p: i for i, p in enumerate(top_pos)}
    chunk = reader.chunk
    for si, lens_name in enumerate(LENSES):
        for li, layer in enumerate(reader.layers):
            for want in (S, extra_top):
                for start in range(0, len(want), chunk):
                    sub = want[start:start + chunk]
                    h = full[layer][torch.tensor(sub, device=dev)].float()
                    if lens_name == "jlens":
                        h = reader.lens.transport(h, layer)
                    lps = torch.log_softmax(reader.lens_model.unembed(h).float(), dim=-1)
                    if want is S:
                        lp[si, li, start:start + len(sub), :K] = lps[:, tt].half().cpu().numpy()
                        lp[si, li, start:start + len(sub), K] = lps[:, ct].mean(-1).half().cpu().numpy()
                    rows = [(j, order[p]) for j, p in enumerate(sub) if p in order]
                    if rows:
                        jj = torch.tensor([r[0] for r in rows], device=dev)
                        v, i = lps[jj].topk(reader.top_k, dim=-1)
                        oo = [r[1] for r in rows]
                        top_ids[si, li, oo] = i.int().cpu().numpy()
                        top_lp[si, li, oo] = v.half().cpu().numpy()
    tf = np.zeros(0, dtype=np.float32)
    if L > P:
        tf_lp, _ = single.next_token_logprobs(reader.lens_model, full[reader.final], ids, chunk=chunk)
        tf = tf_lp[P - 1:].astype(np.float32)
    return {"pos": np.array(S, dtype=np.int32), "lp": lp, "top_pos": np.array(top_pos, dtype=np.int32),
            "top_ids": top_ids, "top_lp": top_lp, "completion_tf_logprob": tf}, top_names, full


def validate(reader, lens, record, plan, arrays, tracked_ids, full):
    """Recompute tracked log-probs at the stored positions with the M0.2-validated
    single-run readout (jmr.lens.readout.readout) and report the max difference."""
    layers = reader.layers
    out = {}
    for si, name in enumerate(LENSES):
        ref = single.readout(reader.lens_model, {l: full[l] for l in layers}, lens=lens if name == "jlens" else None,
                             top_k=reader.top_k, token_ids=tracked_ids, chunk=reader.chunk)
        r = ref["token_logprobs"][:, arrays["pos"]].astype(np.float32)
        d = np.abs(r - arrays["lp"][si, :, :, :len(tracked_ids)].astype(np.float32))
        out[name] = {"max_absdiff": float(d.max()), "frac_identical": float((d == 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="configs/evalaware_tokens.json")
    ap.add_argument("--all-prompt", action="store_true")
    ap.add_argument("--validate", type=int, default=1)
    ap.add_argument("--select", help="file with run IDs to read (one per line)")
    ap.add_argument("--lora", help="adapter directory applied at load time (jmr.lens.lora)")
    ap.add_argument("--reading-model", default="base", help="name recorded in the index (e.g. the adapter)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in a.runs:
        runs += read_runs_lenient(Path(p))
    if a.select:
        keep = {x for x in Path(a.select).read_text().splitlines() if x.strip()}
        runs = [(r, e) for r, e in runs if r.run_id in keep]
        if len(runs) != len(keep):
            raise ValueError(f"{len(keep)} selected run IDs, {len(runs)} found (rule 1)")
    if len({r.run_id for r, _ in runs}) != len(runs):
        raise ValueError("duplicate run IDs")
    index_path = out / "index.jsonl"
    done = {json.loads(l)["run_id"] for l in open(index_path)} if index_path.exists() else set()
    todo = [(r, e) for r, e in runs if r.run_id not in done]
    print(f"{len(runs)} runs, {len(done)} done, {len(todo)} to do -> {out}", flush=True)
    config.snapshot(out)
    model_cfg = config.load("model")
    hf_model, tokenizer, lens_model = load_model()
    if a.lora:
        from jmr.lens.lora import LoRAHooks
        LoRAHooks(hf_model, a.lora)
    lens = load_lens(lens_file())
    reader = BatchReader(lens_model, lens, tokenizer)
    cfg, tracked, control = load_tokens(a.tokens, tokenizer)
    tracked_ids = [t["id"] for t in tracked]
    control_ids = [t["id"] for t in control]
    bm = Path(__import__("jlens").__file__).parent / "data" / "blackmail.json"
    bm_sha = hashlib.sha256(bm.read_bytes()).hexdigest()
    ours = hashlib.sha256(Path("configs/explore/blackmail.json").read_bytes()).hexdigest()
    if bm_sha != ours:
        raise ValueError("configs/explore/blackmail.json differs from jlens's copy (rule 1)")
    (out / "tokens.json").write_text(json.dumps({"tracked": tracked, "columns": [t["text"] for t in tracked]
                                                 + ["control_mean"], "n_control": len(control),
                                                 "layers": reader.layers, "lenses": list(LENSES)}, indent=1))
    fp = fingerprint(tokenizer)
    diffs, n_val = [], 0
    t0 = time.time()
    with open(index_path, "a") as index:
        for i, (record, extras) in enumerate(todo):
            if record.tokenizer_fingerprint != fp:
                raise ValueError(f"{record.run_id}: tokenizer fingerprint {record.tokenizer_fingerprint} != {fp}")
            if record.model.get("revision") != model_cfg["revision"]:
                raise ValueError(f"{record.run_id}: model revision differs from the pinned one")
            if len(record.completion_logprobs) != len(record.completion_token_ids):
                raise ValueError(f"{record.run_id}: log-prob count != completion token count")
            plan = reader.plan(record)
            problems = check_plan(plan, tokenizer)
            if problems:
                raise ValueError(f"{record.run_id}: anchor problems {problems}")
            arrays, top_names, full = read_one(reader, record, plan, tracked_ids, control_ids, a.all_prompt)
            v = None
            if n_val < a.validate:
                v = validate(reader, lens, record, plan, arrays, tracked_ids, full)
                n_val += 1
                print("validation", record.run_id, json.dumps(v), flush=True)
            del full
            vl = np.asarray(record.completion_logprobs, dtype=np.float64)
            if len(vl):
                diffs.append(np.abs(arrays["completion_tf_logprob"] - vl))
            fname = hashlib.sha1(record.run_id.encode()).hexdigest()[:20] + ".npz"
            tmp = out / (fname + ".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, out / fname)
            meta = {"run_id": record.run_id, "file": fname, "question_id": record.question_id,
                    "dataset": record.dataset, "condition": record.condition, "hint_letter": record.hint_letter,
                    "correct_letter": record.correct_letter, "answer": plan.answer, "sample_idx": record.sample_idx,
                    "finish_reason": record.finish_reason, "prompt_len": plan.prompt_len,
                    "completion_len": len(record.completion_token_ids), "named": plan.named,
                    "cot_span": plan.cot_span, "top_names": top_names, "reading_model": a.reading_model,
                    "extras": {k: v2 for k, v2 in extras.items() if k in ("suite", "arm")}, "validation": v}
            index.write(json.dumps(meta) + "\n")
            index.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    d = np.concatenate(diffs) if diffs else np.zeros(0)
    summ = {"n_runs": len(todo), "seconds": round(time.time() - t0, 1),
            "vllm_absdiff": {"n": int(d.size), "median": float(np.median(d)) if d.size else None,
                             "p99": float(np.quantile(d, 0.99)) if d.size else None}}
    (out / f"summary_{int(time.time())}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)
    if d.size and (summ["vllm_absdiff"]["median"] > MEDIAN_MAX or summ["vllm_absdiff"]["p99"] > P99_MAX):
        raise SystemExit(f"teacher-forced log-probs disagree with vLLM beyond M0.4 tolerance: {summ} (rule 1)")


if __name__ == "__main__":
    main()
