"""Is the base-model J-lens still valid on the LoRA-adapted models? (lens env, GPU)

On neutral text (the problem statements of the first --n-texts hack items, as plain
text without any chat template or hack prompt, cut to --max-tokens tokens each),
compare each adapter arm with base Qwen3-8B, per workspace layer and lens:

- top25_overlap_vs_base: |top-25(base lens) ∩ top-25(arm lens)| / 25, mean over positions;
- kl_base_to_arm: KL(base lens || arm lens) in nats, mean over positions;
- faithfulness: KL(own final output || own lens readout) and top-1 agreement of the
  lens with the own final output, for base and for the arm; the ratio arm / base;
- final_kl_base_to_arm: KL(base output || arm output), how much the adapter changes
  the model's output on this text at all;
- logit lens at block 35 vs the arm's own HF logits (log-softmax), max |diff|.

Criterion fixed before running (2026-11-03): the lens counts as clearly broken on an
arm if (a) the logit lens at block 35 differs from the arm's own log-probs by more than
1e-3, or (b) at any workspace layer the arm's J-lens faithfulness KL exceeds 1.5x
base's. Either is a rule-1 stop.

uv run --project envs/lens python scripts/lens/hack_lens_validity.py --items data/hack/items.jsonl \
    --out data/hack/lens_validity.json
"""

import argparse
import json
from pathlib import Path

import torch

from jmr import config
from jmr.lens import readout as single
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.lens.lora import LoRAHooks, adapter_dir


@torch.no_grad()
def dists(lens_model, lens, ids, layers, final):
    """{(lens, layer): log-probs [n, V] fp32 on GPU} for the workspace layers, plus
    ('model', final) = the model's own output, and the HF logits' log-softmax."""
    resid = single.residuals(lens_model, ids, layers + [final])
    out = {}
    for layer in layers:
        h = resid[layer].float()
        out[("logit", layer)] = torch.log_softmax(lens_model.unembed(h).float(), -1)
        out[("jlens", layer)] = torch.log_softmax(lens_model.unembed(lens.transport(h, layer)).float(), -1)
    out[("model", final)] = torch.log_softmax(lens_model.unembed(resid[final].float()).float(), -1)
    return out


def kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-texts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=150)
    args = ap.parse_args()
    cfg, lcfg = config.load("hack_organism"), config.load("lens")
    layers, top_k = sorted(lcfg["workspace_layers"]), lcfg["top_k"]
    hf_model, tok, lens_model = load_model()
    lens = load_lens(lens_file())
    final = lens_model.n_layers - 1
    items = [json.loads(l) for l in open(args.items)][: args.n_texts]
    texts = [tok(it["description"], add_special_tokens=False).input_ids[: args.max_tokens] for it in items]
    arms = [a for a in cfg["arms"] if cfg["arms"][a] is not None]
    acc = {a: {} for a in ["base"] + arms}
    n_pos = sum(len(t) for t in texts)

    def add(arm, key, value):
        acc[arm][key] = acc[arm].get(key, 0.0) + float(value)

    max_block35 = {}
    for ids in texts:
        base = dists(lens_model, lens, ids, layers, final)
        hf_lp = torch.log_softmax(hf_model(torch.tensor([ids], device=lens_model.input_device)).logits[0].float(), -1)
        max_block35["base"] = max(max_block35.get("base", 0.0), float((hf_lp - base[("model", final)]).abs().max()))
        for (name, layer), lp in base.items():
            if name == "model":
                continue
            add("base", f"faith_kl|{name}|{layer}", kl(base[("model", final)], lp).sum())
            add("base", f"faith_top1|{name}|{layer}", (lp.argmax(-1) == base[("model", final)].argmax(-1)).sum())
        for arm in arms:
            hooks = LoRAHooks(hf_model, adapter_dir(arm))
            d = dists(lens_model, lens, ids, layers, final)
            hf_lp = torch.log_softmax(hf_model(torch.tensor([ids], device=lens_model.input_device)).logits[0].float(), -1)
            hooks.remove()
            max_block35[arm] = max(max_block35.get(arm, 0.0), float((hf_lp - d[("model", final)]).abs().max()))
            add(arm, "final_kl_base_to_arm", kl(base[("model", final)], d[("model", final)]).sum())
            tb = base[("model", final)].topk(top_k, -1).indices
            ta = d[("model", final)].topk(top_k, -1).indices
            add(arm, "final_top25_overlap", sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ta, tb)) / top_k)
            for (name, layer), lp in d.items():
                if name == "model":
                    continue
                bl = base[(name, layer)]
                add(arm, f"kl_base_to_arm|{name}|{layer}", kl(bl, lp).sum())
                tb, ta = bl.topk(top_k, -1).indices, lp.topk(top_k, -1).indices
                add(arm, f"top25_overlap_vs_base|{name}|{layer}",
                    sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ta, tb)) / top_k)
                add(arm, f"faith_kl|{name}|{layer}", kl(d[("model", final)], lp).sum())
                add(arm, f"faith_top1|{name}|{layer}", (lp.argmax(-1) == d[("model", final)].argmax(-1)).sum())
            del d
        del base
        torch.cuda.empty_cache()

    res = {"n_positions": n_pos, "n_texts": len(texts), "layers": layers,
           "block35_logit_lens_vs_hf_logprobs_max_absdiff": max_block35, "arms": {}}
    broken = []
    for arm, vals in acc.items():
        res["arms"][arm] = {k: v / n_pos for k, v in vals.items()}
    for arm in arms:
        a, b = res["arms"][arm], res["arms"]["base"]
        ratios = {f"{name}|{l}": a[f"faith_kl|{name}|{l}"] / b[f"faith_kl|{name}|{l}"] for name in ("jlens", "logit")
                  for l in layers}
        res["arms"][arm]["faith_kl_ratio_vs_base"] = ratios
        if max_block35[arm] > 1e-3:
            broken.append(f"{arm}: block-35 logit lens differs from own logits by {max_block35[arm]:.2e}")
        bad = {k: r for k, r in ratios.items() if k.startswith("jlens") and r > 1.5}
        if bad:
            broken.append(f"{arm}: J-lens faithfulness KL ratio > 1.5 at {bad}")
    res["broken"] = broken
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if broken:
        raise SystemExit("LENS VALIDITY FAILED (rule 1): " + "; ".join(broken))


if __name__ == "__main__":
    main()
