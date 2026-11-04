"""Blackmail-scenario rollouts with and without ablating the eval-awareness J-lens
directions (lens env, GPU, HF generate). Replication of the paper's ablation
(Gurnee, Sofroniew et al. 2026, "Strategic thinking and evaluation awareness in a
blackmail scenario": the J-lens directions of 28 eval-awareness tokens are projected
out of the residual stream at the workspace layers, at all token positions).

  uv run --project envs/lens python scripts/lens/evalaware_ablate.py \
      --prompt-run data/explore/evalaware/blackmail/runs.jsonl --out data/explore/evalaware/ablate \
      --conditions none eval control --n 60 --batch 15

J-lens vector of token t at block l (paper, "Technical details": logits are determined,
up to the norm's data-dependent scale, by <v_t, h_l>): v_t = J_l^T (g * W_U[t]), with g
the final RMSNorm weight and W_U the LM head. Ablation at block l: h <- h - Q Q^T h, Q
an orthonormal basis of span{v_t : t in the set}. Blocks: every block 14-32 (the
workspace range; the readouts use the even ones). Sets: "eval" = the eval_set group of
configs/evalaware_tokens.json (26 of the paper's 28 tokens map to single Qwen tokens);
"control" = the first 26 control nouns (same dimension; checks that removing any
26-dimensional J-lens subspace doesn't do the same). Prompt token IDs come from the
gen env's saved run (never re-tokenized here). Sampling: configs/sampling.yaml
(T 0.6, top-p 0.95, top-k 20), max new tokens 8,192, HF generate in bf16.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from jmr import config
from jmr.lens.batch import read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model

BLOCKS = list(range(14, 33))


def bases(hf_model, lens, token_ids, blocks=BLOCKS):
    g = hf_model.model.norm.weight.float()
    W = hf_model.lm_head.weight.float()[token_ids]  # [n, d]
    Qs = {}
    for l in blocks:
        J = lens.jacobians[l].to(W.device).float()  # transport: h @ J.T
        V = (W * g) @ J  # [n, d]: rows v_t = J^T (g * w_t)
        Q, R = torch.linalg.qr(V.T)  # [d, n]
        if (R.diagonal().abs() < 1e-6 * R.diagonal().abs().max()).any():
            raise ValueError(f"block {l}: J-lens vectors are (nearly) linearly dependent")
        Qs[l] = Q.contiguous()
    return Qs


class Ablation:
    def __init__(self, hf_model, Qs):
        self.handles = []
        for l, Q in Qs.items():
            self.handles.append(hf_model.model.layers[l].register_forward_hook(self._hook(Q)))

    @staticmethod
    def _hook(Q):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            h = t.float()
            h = h - (h @ Q) @ Q.T
            t.copy_(h.to(t.dtype))
            return output
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def check_ablation(hf_model, lens_model, lens, ids, Qs, token_ids, layer=24):
    """Sanity check on the prompt at block `layer` (default 24): J-lens log-prob of the set's tokens
    (mean, and max over positions) and the number of positions with one of them in
    the top-20, without and with the ablation hooks (rule 1 if nothing changes)."""
    from jmr.lens.readout import residuals
    tid = torch.tensor(token_ids, device=lens_model.input_device)
    out = {}
    for name in ("before", "after"):
        abl = Ablation(hf_model, Qs) if name == "after" else None
        try:
            r = residuals(lens_model, ids, [layer])[layer].float()
        finally:
            if abl:
                abl.remove()
        lp = torch.log_softmax(lens_model.unembed(lens.transport(r, layer)).float(), -1)
        top = lp.topk(20, -1).indices
        out[name] = {"mean_lp": float(lp[:, tid].mean()), "max_lp": float(lp[:, tid].max()),
                     "positions_in_top20": int(torch.isin(top, tid).any(-1).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=["none", "eval", "control"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-per", nargs="*", default=[], help="per-condition counts, e.g. none=30 control=30")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--tokens", default="configs/evalaware_tokens.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    rec, _ = read_runs_lenient(Path(a.prompt_run))[0]
    ids = rec.prompt_token_ids
    s = config.load("sampling")
    m = config.load("model")
    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    tk = json.loads(Path(a.tokens).read_text())["tokens"]
    sets = {"eval": [t["id"] for t in tk if "eval_set" in t["groups"]]}
    sets["control"] = [t["id"] for t in tk if t["groups"] == ["control"]][:len(sets["eval"])]
    for k, v in sets.items():
        for t in tk:
            if t["id"] in v and tok.decode([t["id"]]) != t["text"]:
                raise ValueError(f"token {t} does not round-trip (rule 1)")
    Q = {k: bases(hf_model, lens, v) for k, v in sets.items()}
    checks = {k: check_ablation(hf_model, lens_model, lens, ids, Q[k], sets[k]) for k in sets}
    checks["eval_tokens_under_control_ablation"] = check_ablation(hf_model, lens_model, lens, ids, Q["control"],
                                                                  sets["eval"])
    print("ablation check (block 24, prompt):", json.dumps(checks), flush=True)
    for k in ("eval", "control"):
        if checks[k]["after"]["max_lp"] >= checks[k]["before"]["max_lp"] - 1e-3:
            raise SystemExit(f"ablation of the {k} set does not lower its tokens' J-lens log-prob (rule 1): {checks[k]}")
    (out / "ablation_check.json").write_text(json.dumps({"checks": checks, "sets": sets, "blocks": BLOCKS}, indent=1))
    path = out / "runs.jsonl"
    done = {}
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            done[r["condition"]] = done.get(r["condition"], 0) + 1
    n_per = {k: int(v) for k, v in (x.split("=") for x in a.n_per)}
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    with open(path, "a") as f:
        for cond in a.conditions:
            k0 = done.get(cond, 0)
            n_c = n_per.get(cond, a.n)
            while k0 < n_c:
                b = min(a.batch, n_c - k0)
                abl = Ablation(hf_model, Q[cond]) if cond != "none" else None
                seed = 1000 * (["none", "eval", "control"].index(cond) + 1) + k0
                torch.manual_seed(seed)
                x = torch.tensor([ids] * b, device=lens_model.input_device)
                t0 = time.time()
                gen = hf_model.generate(x, attention_mask=torch.ones_like(x), do_sample=True,
                                        temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                                        max_new_tokens=a.max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id)
                if abl:
                    abl.remove()
                dt = time.time() - t0
                for j in range(b):
                    c = gen[j, len(ids):].tolist()
                    fin = "length"
                    for p, t in enumerate(c):
                        if t in eos:
                            c, fin = c[:p + 1], "stop"
                            break
                    f.write(json.dumps({"run_id": f"evalaware_ablate/{cond}/{k0 + j}", "condition": cond,
                                        "sample_idx": k0 + j, "batch_seed": seed, "prompt_token_ids": ids,
                                        "completion_token_ids": c, "completion_text": tok.decode(c),
                                        "finish_reason": fin, "model": {"hf_id": m["hf_id"], "revision": m["revision"]},
                                        "sampling": {k: s[k] for k in ("temperature", "top_p", "top_k")} | {
                                            "max_new_tokens": a.max_new, "engine": "hf_generate"}}) + "\n")
                f.flush()
                print(f"{cond} {k0 + b}/{n_c} batch {dt:.0f}s max_len {max(len(gen[j]) for j in range(b)) - len(ids)}",
                      flush=True)
                k0 += b


if __name__ == "__main__":
    main()
