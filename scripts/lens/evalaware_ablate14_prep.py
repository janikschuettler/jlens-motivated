"""Eval-awareness ablation on Qwen3-14B, step 1 (lens env, runs on CPU): the ablation bases
and the HF reference log-probs that the vLLM hook is checked against.

  JMR_MODEL_PROFILE=qwen3-14b HF_HOME=/dev/shm/hf CUDA_VISIBLE_DEVICES= \
  uv run --project envs/lens python scripts/lens/evalaware_ablate14_prep.py --out data/explore/evalaware_ablate14

Bases (each [d_model, 26], orthonormal, float32, per block 16-34 = Qwen3-14B's workspace range,
the replication's "every block of the workspace range"):
  eval_J      J-lens vectors of the 26 eval tokens (the paper's 28-token list; scripts/lens/
              evalaware_ablate.py bases(): v_t = J_l^T (g * W_U[t]))
  control_J   J-lens vectors of the first 26 control nouns (same dimension)
  eval_logit  logit-lens vectors of the 26 eval tokens (J = I: v_t = g * W_U[t], same at every block)
  random      26 Gaussian directions per block (seed 1000 + block), orthonormalized
Reference: the 3 runs in refs.jsonl (saved prompt + completion IDs, never re-tokenized) are
teacher-forced through the HF model in bf16 on CPU, unablated and with each basis projected out
by the replication's hook (evalaware_ablate.Ablation), and the log-prob of every actual next
token is stored (rule 3: no distributions). Also the replication's mechanical check at block 26.
"""

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalaware_ablate import Ablation, bases, check_ablation  # noqa: E402

from jmr import config  # noqa: E402
from jmr.lens.load import lens_file, load_lens, load_model  # noqa: E402
from jmr.tokens import fingerprint  # noqa: E402

BLOCKS = list(range(16, 35))
CHECK_LAYER = 26
N_DIRS = 26


@torch.no_grad()
def next_token_logprobs(hf_model, ids):
    x = torch.tensor([ids])
    logits = hf_model(input_ids=x).logits[0].float()
    lp = torch.log_softmax(logits[:-1], -1)
    return lp.gather(-1, x[0, 1:, None]).squeeze(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="configs/evalaware_tokens.json")
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    m = config.load("model")
    if m["hf_id"] != "Qwen/Qwen3-14B":
        raise SystemExit(f"profile is {m['hf_id']}, expected Qwen/Qwen3-14B (set JMR_MODEL_PROFILE=qwen3-14b)")
    t0 = time.time()
    hf_model, tok, lens_model = load_model("cpu")
    lens = load_lens(lens_file())
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    tk = json.loads(Path(a.tokens).read_text())["tokens"]
    sets = {"eval": [t["id"] for t in tk if "eval_set" in t["groups"]]}
    sets["control"] = [t["id"] for t in tk if t["groups"] == ["control"]][:len(sets["eval"])]
    if len(sets["eval"]) != N_DIRS or len(sets["control"]) != N_DIRS:
        raise SystemExit(f"token sets have {len(sets['eval'])}/{len(sets['control'])} tokens, expected {N_DIRS}")
    for v in sets.values():
        for t in tk:
            if t["id"] in v and tok.decode([t["id"]]) != t["text"]:
                raise SystemExit(f"token {t} does not round-trip (rule 1)")
    Q = {"eval_J": bases(hf_model, lens, sets["eval"], BLOCKS), "control_J": bases(hf_model, lens, sets["control"], BLOCKS)}
    g = hf_model.model.norm.weight.float()
    Wg = hf_model.lm_head.weight.float()[sets["eval"]] * g
    Ql, R = torch.linalg.qr(Wg.T)
    if (R.diagonal().abs() < 1e-6 * R.diagonal().abs().max()).any():
        raise SystemExit("logit-lens vectors are (nearly) linearly dependent")
    Q["eval_logit"] = {l: Ql.contiguous() for l in BLOCKS}
    d = Wg.shape[1]
    Q["random"] = {}
    for l in BLOCKS:
        gen = torch.Generator().manual_seed(1000 + l)
        Qr, _ = torch.linalg.qr(torch.randn(d, N_DIRS, generator=gen))
        Q["random"][l] = Qr.contiguous()
    # overlap between the eval J-lens subspace and the others (mean squared cosine of principal angles)
    overlap = {k: {l: float((Q["eval_J"][l].T @ Q[k][l]).pow(2).sum() / N_DIRS) for l in (16, 26, 34)}
               for k in ("control_J", "eval_logit", "random")}
    buf = io.BytesIO()
    torch.save({k: {l: q.cpu() for l, q in v.items()} for k, v in Q.items()}, buf)
    (out / "bases.pt").write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    print("bases saved", sha[:12], "overlap with eval_J:", json.dumps(overlap), flush=True)

    refs = [json.loads(l) for l in open(out / "refs.jsonl")]
    fp = fingerprint(tok)
    for r in refs:
        if r["tokenizer_fingerprint"] != fp or r["model"]["revision"] != m["revision"]:
            raise SystemExit(f"{r['run_id']}: tokenizer or revision mismatch (rule 1)")
    mech = {k: check_ablation(hf_model, lens_model, lens, refs[0]["prompt_token_ids"], Q[k],
                              sets["eval" if k != "control_J" else "control"], layer=CHECK_LAYER)
            for k in ("eval_J", "control_J", "eval_logit", "random")}
    print("mechanical check, block", CHECK_LAYER, json.dumps(mech), flush=True)
    ref_lp = {}
    for r in refs:
        ids = r["prompt_token_ids"] + r["completion_token_ids"]
        ref_lp[r["run_id"]] = {}
        for arm in ("none", "eval_J", "control_J", "eval_logit", "random"):
            t1 = time.time()
            abl = Ablation(hf_model, Q[arm]) if arm != "none" else None
            try:
                ref_lp[r["run_id"]][arm] = [round(x, 5) for x in next_token_logprobs(hf_model, ids)]
            finally:
                if abl:
                    abl.remove()
            print(f"{r['run_id']} {arm} {len(ids)} tokens {time.time() - t1:.0f}s", flush=True)
        # CPU-bf16 HF vs the original vLLM sampled-token log-probs (completion part)
        P = len(r["prompt_token_ids"])
        hf = ref_lp[r["run_id"]]["none"][P - 1:]
        dv = sorted(abs(x - y) for x, y in zip(hf, r["completion_logprobs"]))
        print(f"  HF-cpu vs original vLLM, completion: median {dv[len(dv) // 2]:.4f} p99 {dv[int(.99 * len(dv))]:.3f}")
    summary = {}
    for rid, v in ref_lp.items():
        summary[rid] = {}
        for arm in ("eval_J", "control_J", "eval_logit", "random"):
            dd = sorted(abs(x - y) for x, y in zip(v[arm], v["none"]))
            summary[rid][arm] = {"median_abs_change_vs_none": dd[len(dd) // 2], "p99": dd[int(.99 * len(dd))],
                                 "mean": sum(dd) / len(dd)}
    print("ablation effect on next-token log-probs (HF):", json.dumps(summary, indent=1))
    (out / "ref_logprobs.json").write_text(json.dumps({
        "blocks": BLOCKS, "sets": sets, "bases_sha256": sha, "overlap_with_eval_J": overlap,
        "mechanical_check": mech, "effect_summary": summary, "model": m, "engine": "HF transformers bf16, CPU",
        "logprobs": ref_lp}, indent=None))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
