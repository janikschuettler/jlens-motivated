"""Eval-awareness smoke test on Qwen3.6-27B (lens env, GPU + CPU offload). Exploratory.

  JMR_MODEL_PROFILE=qwen3.6-27b HF_HOME=/dev/shm/hf uv run --project envs/lens python \
      scripts/lens/evalaware_big.py --out data/explore/evalaware_big [--gpu-layers 42] \
      [--prompts-per-suite 14] [--runs data/explore/evalaware_big/api_runs.jsonl]

One process, one model load:
  0. checks: lens shapes and layer count match the model; the logit lens at the last block
     equals the model's own logits (rule 1 on mismatch);
  1. prompt only (no generation): the paper's eval score at the start of the model's turn
     for eval-like vs real-user prompts of the replication's suites (prompt texts from
     data/explore/evalaware/suites.json, re-rendered with this model's chat template);
     top-25 at every prompt position for a few prompts, for reading by eye;
  2. API runs (OpenRouter, text only): re-tokenized with the pinned tokenizer and the exact
     chat template (labelled "re-tokenized from API text"), teacher-forced; per-token
     log-probs for the fidelity check; tracked-token readout at 128 evenly spaced
     reasoning positions plus the anchors.

Rule 3: the full softmax is computed in memory only; stored are the tracked tokens'
log-probs, the mean log-prob of the 100 control nouns, and top-25 IDs/log-probs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jlens import JacobianLens, from_hf
from jlens.hooks import ActivationRecorder
from transformers import AutoConfig, AutoTokenizer

from jmr import config

EVAL_SUITES = ["redux_home", "redux_bench", "syco", "moral", "hack_task", "grader_agent", "grader_plain"]
DEPLOY_SUITES = ["oasst", "wildchat"]
TOPK = 25
N_COT = 128
LENSES = ("jlens", "logit")


def load_all(gpu_layers: int, lens_override: str | None = None):
    m = config.load("model")
    lc = config.load("lens")
    from transformers import Qwen3_5ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(m["hf_id"], revision=m["revision"])
    tc = cfg.text_config
    n = tc.num_hidden_layers
    if n != m["expected"]["num_hidden_layers"] or tc.hidden_size != m["expected"]["hidden_size"]:
        raise SystemExit(f"config {n} layers / d {tc.hidden_size} != expected (rule 1)")
    dm = {"model.visual": "cpu", "model.language_model.embed_tokens": 0, "model.language_model.norm": 0,
          "model.language_model.rotary_emb": 0, "lm_head": 0}
    for i in range(n):
        dm[f"model.language_model.layers.{i}"] = 0 if i < gpu_layers else "cpu"
    t0 = time.time()
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(m["hf_id"], revision=m["revision"], dtype=torch.bfloat16,
                                                         device_map=dm)
    hf.eval()
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    lm = from_hf(hf, tok, force_bos=False)
    print(f"model loaded in {time.time() - t0:.0f}s; {gpu_layers}/{n} blocks on GPU; "
          f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    lfile = lens_override or lc["jlens"]["file"]  # --lens-file: e.g. the J++ lens (same J-dict format)
    path = hf_hub_download(lc["jlens"]["repo"], lfile, revision=lc["jlens"]["revision"])
    print("lens file", lfile, flush=True)
    lens = JacobianLens.load(path)
    lo, hi = lc["jlens"]["expected_source_layers"]
    shapes = {tuple(J.shape) for J in lens.jacobians.values()}
    info = {"source_layers": [lens.source_layers[0], lens.source_layers[-1], len(lens.source_layers)],
            "d_model": lens.d_model, "shapes": sorted(shapes), "n_prompts": lens.n_prompts,
            "model_layers": lm.n_layers, "model_d": lm.d_model}
    print("lens", info, flush=True)
    if (lens.source_layers != list(range(lo, hi + 1)) or lens.d_model != tc.hidden_size
            or shapes != {(tc.hidden_size, tc.hidden_size)} or lm.n_layers != n or hi != n - 2):
        raise SystemExit(f"lens does not match the model: {info} (rule 1)")
    ws = lc["workspace_layers"]
    J = {l: lens.jacobians[l].to("cuda", torch.float32) for l in ws}
    return hf, tok, lm, J, ws, info, m, lc


def load_tokens(path, tok):
    cfg = json.loads(Path(path).read_text())
    for t in cfg["tokens"]:
        if tok(t["text"], add_special_tokens=False).input_ids != [t["id"]]:
            raise SystemExit(f"tracked token {t['text']!r} does not round-trip (rule 1)")
    tracked = [t for t in cfg["tokens"] if t["groups"] != ["control"]]
    control = [t for t in cfg["tokens"] if t["groups"] == ["control"]]
    assert len(control) == 100
    return tracked, control


class Reader:
    def __init__(self, lm, J, ws, tracked, control, chunk=256):
        self.lm, self.J, self.ws = lm, J, ws
        self.final = lm.n_layers - 1
        self.tt = torch.tensor([t["id"] for t in tracked], device="cuda")
        self.ct = torch.tensor([t["id"] for t in control], device="cuda")
        self.chunk = chunk

    @torch.no_grad()
    def forward(self, batch_ids):
        """Right-padded batch, no mask: causal attention and the causal recurrent layers
        never let a real token see the padding after it."""
        L = max(len(x) for x in batch_ids)
        pad = self.lm.tokenizer.pad_token_id or 0
        x = torch.tensor([list(b) + [pad] * (L - len(b)) for b in batch_ids], device=self.lm.input_device)
        with ActivationRecorder(self.lm.layers, at=[*self.ws, self.final]) as rec:
            self.lm.forward(x)
        return {l: rec.activations[l].detach().to("cuda") for l in [*self.ws, self.final]}

    @torch.no_grad()
    def lens_lp(self, h, layer, lens):
        h = h.float()
        if lens == "jlens":
            h = h @ self.J[layer].T
        return torch.log_softmax(self.lm.unembed(h).float(), dim=-1)

    @torch.no_grad()
    def read(self, acts, b, pos, top_pos=()):
        """Tracked-token log-probs + control mean at pos; top-25 at top_pos. Row b of the batch."""
        K = len(self.tt)
        lp = np.zeros((2, len(self.ws), len(pos), K + 1), np.float16)
        tid = np.zeros((2, len(self.ws), len(top_pos), TOPK), np.int32)
        tlp = np.zeros((2, len(self.ws), len(top_pos), TOPK), np.float16)
        for si, lens in enumerate(LENSES):
            for li, l in enumerate(self.ws):
                for want, kind in ((pos, "t"), (top_pos, "k")):
                    for s in range(0, len(want), self.chunk):
                        sub = list(want[s:s + self.chunk])
                        lps = self.lens_lp(acts[l][b, sub], l, lens)
                        if kind == "t":
                            lp[si, li, s:s + len(sub), :K] = lps[:, self.tt].half().cpu().numpy()
                            lp[si, li, s:s + len(sub), K] = lps[:, self.ct].mean(-1).half().cpu().numpy()
                        else:
                            v, i = lps.topk(TOPK, dim=-1)
                            tid[si, li, s:s + len(sub)] = i.int().cpu().numpy()
                            tlp[si, li, s:s + len(sub)] = v.half().cpu().numpy()
        return lp, tid, tlp

    @torch.no_grad()
    def next_token_lp(self, acts, b, ids, start):
        """Teacher-forced log-prob of ids[t] given ids[:t], for t >= start (final block -> unembed)."""
        out = []
        h = acts[self.final][b]
        for s in range(start - 1, len(ids) - 1, self.chunk):
            e = min(s + self.chunk, len(ids) - 1)
            lps = torch.log_softmax(self.lm.unembed(h[s:e]).float(), dim=-1)
            tgt = torch.tensor(ids[s + 1:e + 1], device="cuda")
            out.append(lps.gather(1, tgt[:, None])[:, 0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def turn_start(ids, tok):
    """Index of the newline after the last <|im_start|>assistant (the replication's
    'start of the model's turn')."""
    ims = tok.convert_tokens_to_ids("<|im_start|>")
    i = max(k for k, t in enumerate(ids) if t == ims)
    if tok.decode(ids[i + 1:i + 3]) != "assistant\n":
        raise SystemExit(f"unexpected turn start: {tok.decode(ids[i:i + 4])!r} (rule 1)")
    return i + 2


def check_logits(hf, tok, reader):
    msgs = [{"role": "user", "content": "What is the capital of France? Answer briefly."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=True)
    ids = list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
    with torch.no_grad():
        logits = hf(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0].float()
        acts = reader.forward([ids])
        mine = reader.lm.unembed(acts[reader.final][0]).float()
    d = (logits.to("cuda") - mine).abs().max().item()
    print(f"logit lens at the last block vs model logits: max |diff| {d:.4g} over {len(ids)} positions", flush=True)
    if d > 1e-3:
        raise SystemExit(f"logit lens at last block != model logits (max diff {d}) (rule 1)")
    return {"n_positions": len(ids), "max_absdiff": d, "rendered_prompt": tok.decode(ids)}


def chat_ids(tok, msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def step_prompts(reader, tok, out, per_suite, max_tokens_batch=4000):
    suites = json.loads(Path("data/explore/evalaware/suites.json").read_text())
    rows = []
    for s in EVAL_SUITES:
        rows += [(s, "eval", i, suites[s][i]) for i in range(min(per_suite, len(suites[s])))]
    rows.append(("blackmail", "eval", 0, suites["blackmail"][0]))
    n_eval = len(rows) - 1  # without the single blackmail prompt
    for s in DEPLOY_SUITES:
        n = n_eval // len(DEPLOY_SUITES)
        rows += [(s, "deploy", i, suites[s][i]) for i in range(min(n, len(suites[s])))]
    by_eye = {("blackmail", 0), ("grader_agent", 0), ("redux_bench", 0), ("syco", 0), ("redux_home", 0),
              ("oasst", 0), ("wildchat", 0), ("wildchat", 1)}
    items = [(s, kind, i, chat_ids(tok, msgs)) for s, kind, i, msgs in rows]
    order = sorted(range(len(items)), key=lambda k: len(items[k][3]))
    res, t0, done = [], time.time(), 0
    batches, cur, cur_max = [], [], 0
    for k in order:
        L = len(items[k][3])
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens_batch:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(k)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    tops = {}
    for bi, batch in enumerate(batches):
        acts = reader.forward([items[k][3] for k in batch])
        for b, k in enumerate(batch):
            s, kind, i, ids = items[k]
            ts = turn_start(ids, tok)
            pos = [ts, len(ids) - 1]
            top_pos = list(range(len(ids))) if (s, i) in by_eye else pos
            lp, tid, tlp = reader.read(acts, b, pos, top_pos)
            res.append({"suite": s, "kind": kind, "idx": i, "n_tokens": len(ids), "turn_start": ts, "lp": lp})
            if (s, i) in by_eye:
                tops[f"{s}/{i}"] = {"ids": ids, "top_ids": tid, "top_lp": tlp}
        del acts
        done += len(batch)
        if bi == 0:
            print(f"first prompt batch: {len(batch)} prompts, {time.time() - t0:.1f}s", flush=True)
    print(f"step 1: {len(res)} prompts in {time.time() - t0:.0f}s ({len(batches)} batches)", flush=True)
    np.savez_compressed(out / "prompts.npz", lp=np.stack([r["lp"] for r in res]))
    (out / "prompts_index.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "lp"} for r in res]))
    np.savez_compressed(out / "prompts_top.npz", **{f"{k}|{f}": np.asarray(v[f]) for k, v in tops.items()
                                                  for f in ("ids", "top_ids", "top_lp")})
    return time.time() - t0


def api_tokens(tok, run):
    """Re-tokenize an API run: prompt via the chat template; completion = the template's own
    rendering of the assistant turn (reasoning + response), tokenized without specials added."""
    msgs = run["messages"]
    p_ids = chat_ids(tok, msgs)
    p_str = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=False)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": run["content"] or "",
                                            "reasoning_content": run["reasoning"] or ""}],
                                   enable_thinking=True, tokenize=False)
    if not full.startswith(p_str):
        raise SystemExit(f"{run['run_id']}: rendered conversation does not extend the prompt (rule 1)")
    comp = full[len(p_str):]
    if comp.endswith("\n"):
        comp = comp[:-1]  # the trailing newline after <|im_end|> is not generated
    if (run["reasoning"] or "").strip()[:50] not in comp:
        raise SystemExit(f"{run['run_id']}: reasoning missing from the rendered turn (rule 1)")
    enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    return p_ids, list(enc.input_ids), comp, enc.offset_mapping


def step_runs(reader, tok, out, runs_path):
    runs = [json.loads(l) for l in open(runs_path)]
    think_end = tok.convert_tokens_to_ids("</think>")
    idx, t0 = [], time.time()
    arrays = {}
    for n, run in enumerate(runs):
        p_ids, c_ids, comp, offs = api_tokens(tok, run)
        ids = p_ids + c_ids
        P = len(p_ids)
        te = [k for k, t in enumerate(c_ids) if t == think_end]
        if len(te) != 1:
            raise SystemExit(f"{run['run_id']}: {len(te)} </think> tokens in the completion (rule 1)")
        cot = (P, P + te[0])  # reasoning token span
        r_txt = (run["reasoning"] or "").strip()
        r0 = comp.find(r_txt[:200])
        pos = sorted(set(np.linspace(cot[0], cot[1] - 1, min(N_COT, cot[1] - cot[0])).round().astype(int).tolist()))
        ts = turn_start(p_ids, tok)
        allpos = [ts, P - 1] + pos + [len(ids) - 2]
        tstart = time.time()
        acts = reader.forward([ids])
        tfw = time.time() - tstart
        lp, _, _ = reader.read(acts, 0, allpos)
        tf = reader.next_token_lp(acts, 0, ids, P)
        del acts
        key = f"r{n}"
        arrays[key + "_lp"] = lp
        arrays[key + "_tf"] = tf.astype(np.float32)
        idx.append({"key": key, "run_id": run["run_id"], "arm": run["arm"], "item_id": run["item_id"],
                    "sample": run["sample"], "answer": run["answer"], "hinted": run["hinted"],
                    "correct": run["correct"], "prompt_len": P, "completion_len": len(c_ids), "cot_span": cot,
                    "positions": allpos, "reasoning_char0": r0,
                    "pos_char": [None, None] + [int(offs[p - P][0]) for p in pos] + [None],
                    "provenance": "re-tokenized from API text", "forward_s": round(tfw, 1)})
        if n == 0:
            print(f"first run: {len(ids)} tokens, forward {tfw:.1f}s, total {time.time() - tstart:.1f}s", flush=True)
        if (n + 1) % 10 == 0 or n + 1 == len(runs):
            print(f"{n + 1}/{len(runs)} {time.time() - t0:.0f}s", flush=True)
            np.savez_compressed(out / "runs.npz", **arrays)  # partial results survive a stop
            (out / "runs_index.json").write_text(json.dumps(idx))
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-layers", type=int, default=42)
    ap.add_argument("--prompts-per-suite", type=int, default=14)
    ap.add_argument("--runs")
    ap.add_argument("--tokens", default="configs/evalaware_tokens_qwen3.6-27b.json")
    ap.add_argument("--skip-prompts", action="store_true")
    ap.add_argument("--lens-file", help="lens file in the same repo/revision instead of the profile's J-lens")
    ap.add_argument("--batch-tokens", type=int, default=4000, help="padded tokens per prompt batch")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hf, tok, lm, J, ws, linfo, m, lc = load_all(a.gpu_layers, a.lens_file)
    linfo["file"] = a.lens_file or lc["jlens"]["file"]
    tracked, control = load_tokens(a.tokens, tok)
    reader = Reader(lm, J, ws, tracked, control, chunk=lc.get("position_chunk", 256))
    chk = check_logits(hf, tok, reader)
    meta = {"model": m, "lens": lc, "lens_info": linfo, "logit_check": chk, "workspace_layers": ws,
            "tracked": tracked, "lenses": LENSES, "gpu_layers": a.gpu_layers, "git": config.git_state()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    timing = {}
    if not a.skip_prompts:
        timing["prompts_s"] = step_prompts(reader, tok, out, a.prompts_per_suite, a.batch_tokens)
    if a.runs:
        timing["runs_s"] = step_runs(reader, tok, out, a.runs)
    (out / "timing.json").write_text(json.dumps(timing))
    print("done", timing, flush=True)


if __name__ == "__main__":
    main()
