"""Lens readouts of the reward-hack organism runs (lens env, GPU).

For each run: one teacher-forced pass over exactly the saved prompt + completion
token IDs, by the run's own model (base Qwen3-8B or base + its LoRA adapter, hooks
applied at load time, jmr.lens.lora), or by another arm's model with --reader-arm
(the "cross-read" control: same tokens, different model). Stored per run, one .npz
(rule 3: no full-vocabulary output; the softmax is computed in memory in chunks):

  x_lp, x_rank [2 lenses (jlens, logit), 10 workspace layers, T, K] fp16 / int32:
      log-prob and rank (1 + #tokens strictly higher) of the K tracked tokens of
      configs/hack_organism.yaml (concept, task, neutral), at T = the last prompt
      position + every completion position (position t predicts token t + 1);
  model_lp, model_rank [T, K]: the same for the model's own output (block 35);
  top_ids, top_lp [2, 10, A, 25], model_top_*: top-25 at A anchors;
  resid [3 layers (resid_layers), A, 4096] fp16: block outputs at the anchors;
  completion_tf_logprob [n_completion]: teacher-forced log-prob of each sampled token.

Anchors: cot_00..cot_15 evenly spaced over the <thinking> content tokens, then
last_prompt, think_end (the "</" of </thinking>), pre_code (the position whose next
token is the first code token after ```python), pre_mention (the position whose
next token holds the first hack-related word). Token windows (think, echo_free,
mention) go to the meta line in index.jsonl.

Rule-1 checks before each run: tokenizer fingerprint, pinned model revision, the
run's adapter (repo, revision, subfolder) matches the config for its arm, one vLLM
log-prob per completion token, the decoded token bytes give the same label group as
labels.jsonl. After own-model reads, the teacher-forced vs vLLM log-prob agreement
must pass M0.4 (median <= 0.05, p99 <= 0.5, shifted median >= 10x).

uv run --project envs/lens python scripts/lens/readout_hack.py --runs data/hack/gen/runs.jsonl \
    --labels data/hack/gen/labels.jsonl --out data/hack/readouts/own [--reader-arm beta002 --arms base] [--validate 2]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from jlens import ActivationRecorder

from jmr import config, hack
from jmr.lens import readout as single
from jmr.lens.batch import _min_rank, read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.lens.lora import LoRAHooks, adapter_dir
from jmr.tokens import fingerprint, token_bytes

LENSES = ("jlens", "logit")
MEDIAN_MAX, P99_MAX, SHIFT_FACTOR = 0.05, 0.5, 10.0
NAMED_ANCHORS = ["last_prompt", "think_end", "pre_code", "pre_mention"]


def tracked(cfg: dict, tok) -> tuple[list[int], list[str], list[str]]:
    ids, labels, sets, problems = [], [], [], []
    for s in ("concept", "task", "neutral"):
        for text, tid in cfg["tokens"][s].items():
            enc = tok(text, add_special_tokens=False).input_ids
            if enc != [tid] or tok.decode([tid]) != text:
                problems.append(f"{text!r}: encodes to {enc}, config {tid}")
            ids.append(tid)
            labels.append(text)
            sets.append(s)
    if problems or len(set(ids)) != len(ids):
        raise ValueError(f"tracked tokens don't match the tokenizer or repeat: {problems}")
    return ids, labels, sets


def save_npz(path: Path, arrays: dict) -> int:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path.stat().st_size


class Reader:
    def __init__(self, lens_model, lens, tok, cfg, lcfg):
        self.m, self.lens, self.tok = lens_model, lens, tok
        self.layers = sorted(lcfg["workspace_layers"])
        self.final = lens_model.n_layers - 1
        self.top_k, self.chunk = lcfg["top_k"], lcfg["position_chunk"]
        self.rcfg = cfg["readout"]
        self.resid_layers = list(self.rcfg["resid_layers"])
        if not set(self.resid_layers) <= set(self.layers):
            raise ValueError("resid_layers must be workspace layers")
        self.ids, self.labels, self.sets = tracked(cfg, tok)
        self.dev = lens_model.input_device
        self.tt = torch.tensor(self.ids, device=self.dev)
        lens.jacobians = {l: lens.jacobians[l].to(self.dev) for l in self.layers}

    def plan(self, record) -> dict:
        P = len(record.prompt_token_ids)
        w = hack.token_windows(token_bytes(self.tok, record.completion_token_ids))
        named = {"last_prompt": P - 1, "think_end": -1, "pre_code": -1, "pre_mention": -1}
        if w.think_end is not None:
            named["think_end"] = P + w.think_end
        if w.code_first is not None:
            named["pre_code"] = P + w.code_first - 1
        if w.mention is not None:
            named["pre_mention"] = P + w.mention - 1
        cot = [-1] * self.rcfg["n_cot_anchors"]
        if w.think is not None:
            a, b = w.think
            n = min(len(cot), b - a)
            for k, pos in enumerate(np.linspace(P + a, P + b - 1, n).round().astype(int)):
                cot[k] = int(pos)
        return {"P": P, "windows": w, "anchors": cot + [named[n] for n in NAMED_ANCHORS],
                "anchor_names": [f"cot_{k:02d}" for k in range(len(cot))] + NAMED_ANCHORS}

    @torch.no_grad()
    def read(self, record, plan) -> dict:
        ids = record.prompt_token_ids + record.completion_token_ids
        L, P = len(ids), plan["P"]
        lo = P - self.rcfg["prompt_tail"]
        T, K, nL = L - lo, len(self.ids), len(self.layers)
        full_layers = sorted(set(self.layers) | {self.final})
        with ActivationRecorder(self.m.layers, at=full_layers) as rec:
            self.m.forward(torch.tensor([ids], device=self.dev))
            full = {l: rec.activations[l][0].detach() for l in full_layers}
        anchors = torch.tensor(plan["anchors"], device=self.dev)
        A = len(plan["anchors"])
        x_lp = torch.zeros((2, nL, T, K), dtype=torch.float16, device=self.dev)
        x_rank = torch.zeros((2, nL, T, K), dtype=torch.int32, device=self.dev)
        top_ids = torch.full((2, nL, A, self.top_k), -1, dtype=torch.int32, device=self.dev)
        top_lp = torch.full((2, nL, A, self.top_k), float("nan"), dtype=torch.float16, device=self.dev)
        m_lp = torch.zeros((T, K), dtype=torch.float16, device=self.dev)
        m_rank = torch.zeros((T, K), dtype=torch.int32, device=self.dev)
        m_top_ids = torch.full((A, self.top_k), -1, dtype=torch.int32, device=self.dev)
        m_top_lp = torch.full((A, self.top_k), float("nan"), dtype=torch.float16, device=self.dev)
        next_lp = torch.zeros(L, dtype=torch.float32, device=self.dev)
        target = torch.tensor(ids[1:] + [0], device=self.dev)

        def one(resid, layer, o_lp, o_rank, o_tid, o_tlp, want_next=False):
            # Same chunking from position 0 and the same ops as jmr.lens.readout.readout.
            for start in range(0, L, self.chunk):
                end = min(start + self.chunk, L)
                h = resid[start:end].float()
                if layer is not None:
                    h = self.lens.transport(h, layer)
                lp = torch.log_softmax(self.m.unembed(h).float(), dim=-1)
                if want_next:
                    stop = min(end, L - 1)
                    if stop > start:
                        next_lp[start:stop] = lp[torch.arange(stop - start, device=lp.device), target[start:stop]]
                if end > lo:
                    a = max(lo, start)
                    rows = lp[a - start:]
                    tl = rows[:, self.tt]
                    o_lp[a - lo:end - lo] = tl.half()
                    o_rank[a - lo:end - lo] = _min_rank(rows, tl)
                msk = (anchors >= start) & (anchors < end)
                if msk.any():
                    v, i = lp[anchors[msk] - start].topk(self.top_k, dim=-1)
                    o_tid[msk] = i.int()
                    o_tlp[msk] = v.half()

        for si, name in enumerate(LENSES):
            for li, layer in enumerate(self.layers):
                one(full[layer], layer if name == "jlens" else None, x_lp[si, li], x_rank[si, li],
                    top_ids[si, li], top_lp[si, li])
        one(full[self.final], None, m_lp, m_rank, m_top_ids, m_top_lp, want_next=True)
        sel = torch.tensor([p if p >= 0 else 0 for p in plan["anchors"]], device=self.dev)
        resid = torch.stack([full[l][sel] for l in self.resid_layers]).half()
        if not torch.isfinite(resid).all():
            raise ValueError(f"{record.run_id}: block outputs overflow fp16")
        for k, p in enumerate(plan["anchors"]):
            if p < 0:
                resid[:, k] = float("nan")
        return {"layers": np.array(self.layers, dtype=np.int16), "positions_lo": np.array(lo, dtype=np.int32),
                "tracked_ids": np.array(self.ids, dtype=np.int32),
                "x_lp": x_lp.cpu().numpy(), "x_rank": x_rank.cpu().numpy(),
                "model_lp": m_lp.cpu().numpy(), "model_rank": m_rank.cpu().numpy(),
                "anchor_pos": np.array(plan["anchors"], dtype=np.int32),
                "top_ids": top_ids.cpu().numpy(), "top_lp": top_lp.cpu().numpy(),
                "model_top_ids": m_top_ids.cpu().numpy(), "model_top_lp": m_top_lp.cpu().numpy(),
                "resid_layers": np.array(self.resid_layers, dtype=np.int16), "resid": resid.cpu().numpy(),
                "completion_tf_logprob": next_lp[P - 1:L - 1].cpu().numpy()}


def validate(reader: Reader, record, arrays) -> dict:
    """Recompute with the M0.2-validated single-run readout and compare."""
    ids = record.prompt_token_ids + record.completion_token_ids
    resid = single.residuals(reader.m, ids, reader.layers + [reader.final])
    lo = int(arrays["positions_lo"])
    out = {}
    for si, name in enumerate(LENSES):
        ref = single.readout(reader.m, {l: resid[l] for l in reader.layers},
                             lens=reader.lens if name == "jlens" else None, top_k=reader.top_k,
                             token_ids=reader.ids, chunk=reader.chunk)
        d = np.abs(ref["token_logprobs"][:, lo:].astype(np.float32) - arrays["x_lp"][si].astype(np.float32))
        out[name] = {"x_lp_max_absdiff": float(d.max())}
    ref = single.readout(reader.m, {reader.final: resid[reader.final]}, lens=None, top_k=reader.top_k,
                         token_ids=reader.ids, chunk=reader.chunk)
    out["model_lp_max_absdiff_vs_logit_lens_block35"] = float(
        np.abs(ref["token_logprobs"][0, lo:].astype(np.float32) - arrays["model_lp"].astype(np.float32)).max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reader-arm", default=None, help="read every selected run with this arm's model")
    ap.add_argument("--arms", default=None, help="only runs of these arms (comma-separated)")
    ap.add_argument("--run-ids", default=None, help="file with one run_id per line: only these runs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--validate", type=int, default=0)
    args = ap.parse_args()

    cfg, lcfg, mcfg = config.load("hack_organism"), config.load("lens"), config.load("model")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    labels = {json.loads(l)["run_id"]: json.loads(l) for l in open(args.labels)}
    runs = [r for r, _ in read_runs_lenient(Path(args.runs))]
    if args.arms:
        runs = [r for r in runs if r.condition in args.arms.split(",")]
    if args.run_ids:
        keep = {l.strip() for l in open(args.run_ids) if l.strip()}
        runs = [r for r in runs if r.run_id in keep]
    runs = runs[: args.limit] if args.limit else runs
    index_path = out / "index.jsonl"
    done = set()
    if index_path.exists():
        for line in open(index_path):
            m = json.loads(line)
            if (out / m["file"]).exists():
                done.add(m["run_id"])
    todo = [r for r in runs if r.run_id not in done]
    print(f"{len(runs)} runs, {len(done)} done, {len(todo)} to do → {out}", flush=True)

    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    reader = Reader(lens_model, lens, tok, cfg, lcfg)
    fp = fingerprint(tok)
    by_reader: dict[str, list] = {}
    for r in todo:
        by_reader.setdefault(args.reader_arm or r.condition, []).append(r)

    diffs, shifted, validations, sizes = [], [], [], []
    t_start, n_done = time.time(), 0
    with open(index_path, "a") as index:
        for arm, rs in by_reader.items():
            adir = adapter_dir(arm)
            hooks = LoRAHooks(hf_model, adir) if adir else None
            want_adapter = None if adir is None else {"repo": cfg["adapters_repo"], "revision": cfg["adapters_revision"],
                                                      "subfolder": cfg["arms"][arm]}
            print(f"reader {arm}: {len(rs)} runs, adapter {adir}, {hooks.n_modules if hooks else 0} LoRA modules",
                  flush=True)
            for i, record in enumerate(rs):
                own = record.condition == arm
                if record.tokenizer_fingerprint != fp or record.model.get("revision") != mcfg["revision"]:
                    raise ValueError(f"{record.run_id}: tokenizer or model revision mismatch")
                exp = None if cfg["arms"][record.condition] is None else {
                    "repo": cfg["adapters_repo"], "revision": cfg["adapters_revision"],
                    "subfolder": cfg["arms"][record.condition]}
                if record.model.get("adapter") != exp:
                    raise ValueError(f"{record.run_id}: adapter {record.model.get('adapter')} != config {exp}")
                if len(record.completion_logprobs) != len(record.completion_token_ids):
                    raise ValueError(f"{record.run_id}: log-prob count mismatch")
                plan = reader.plan(record)
                lab = labels[record.run_id]
                relab = hack.label_run(plan["windows"].text)
                if relab.group != lab["group"]:
                    raise ValueError(f"{record.run_id}: token-byte text gives group {relab.group}, labels.jsonl "
                                     f"{lab['group']}")
                t0 = time.time()
                arrays = reader.read(record, plan)
                torch.cuda.synchronize()
                dt = time.time() - t0
                tf = arrays["completion_tf_logprob"].astype(np.float64)
                vl = np.asarray(record.completion_logprobs, dtype=np.float64)
                d = np.abs(tf - vl)
                if own:
                    diffs.append(d)
                    shifted.append(np.abs(tf[1:] - vl[:-1]))
                if i < args.validate:
                    validations.append({"run_id": record.run_id, "reader": arm, **validate(reader, record, arrays)})
                    print(json.dumps(validations[-1]), flush=True)
                w = plan["windows"]
                fname = hashlib.sha1(f"{arm}|{record.run_id}".encode()).hexdigest()[:20] + ".npz"
                meta = {"run_id": record.run_id, "arm": record.condition, "reader": arm, "own": own,
                        "problem_id": record.question_id, "group": lab["group"], "hack": lab["hack"],
                        "prompt_len": plan["P"], "completion_len": len(record.completion_token_ids),
                        "finish_reason": record.finish_reason, "think": w.think, "echo_free": w.echo_free,
                        "mention": w.mention, "think_end": w.think_end, "code_first": w.code_first,
                        "anchor_names": plan["anchor_names"], "anchor_pos": plan["anchors"],
                        "tracked": {"ids": reader.ids, "labels": reader.labels, "sets": reader.sets},
                        "adapter_reader": want_adapter, "file": fname, "seconds": round(dt, 3),
                        "vllm_absdiff": {"median": float(np.median(d)), "p99": float(np.quantile(d, 0.99)),
                                         "max": float(d.max())}}
                arrays["meta_json"] = np.array(json.dumps(meta))
                sizes.append(save_npz(out / fname, arrays))
                index.write(json.dumps(meta) + "\n")
                index.flush()
                n_done += 1
                if i % 25 == 0 or i == len(rs) - 1:
                    el = time.time() - t_start
                    print(f"[{arm} {i + 1}/{len(rs)}] len {len(record.token_ids)} {dt:.2f}s | "
                          f"{n_done / el * 60:.1f} runs/min | {np.mean(sizes) / 1e6:.2f} MB/run", flush=True)
            if hooks:
                hooks.remove()
    summary = {"n_new": n_done, "elapsed_s": round(time.time() - t_start, 1), "validations": validations,
               "disk_bytes_total_new": int(np.sum(sizes)) if sizes else 0}
    ok = True
    if diffs:
        dd, ss = np.concatenate(diffs), np.concatenate(shifted)
        med, p99 = float(np.median(dd)), float(np.quantile(dd, 0.99))
        summary["vllm_logprob_absdiff_own"] = {"n_tokens": int(dd.size), "median": med, "p99": p99,
                                               "max": float(dd.max()), "shifted_median": float(np.median(ss))}
        summary["m0_4_pass"] = {"median": med <= MEDIAN_MAX, "p99": p99 <= P99_MAX,
                                "shift": float(np.median(ss)) >= SHIFT_FACTOR * med}
        ok = all(summary["m0_4_pass"].values())
    path = out / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not ok:
        sys.exit("FAIL: teacher-forced log-probs disagree with vLLM beyond the M0.4 tolerance")


if __name__ == "__main__":
    main()
