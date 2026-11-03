"""M0.2 (lens env): download the Neuronpedia J-lens, check it matches Qwen3-8B,
check our token-ID readout against jlens's own lens.apply, and check the
logit lens at the last layer against the model's own logits.

uv run --project envs/lens python scripts/lens/m0_check_lens.py
"""

import torch

from jmr import config
from jmr.checks import Checks
from jmr.lens.load import inspect_lens_file, lens_file, load_lens, load_model
from jmr.lens.readout import readout, residuals

PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is",
    "The capital of the state containing Dallas is",
    "<|im_start|>user\nName a primary colour.<|im_end|>\n<|im_start|>assistant\n",
]


def main() -> None:
    checks = Checks("m0_2_lens")
    lens_cfg = config.load("lens")
    spec = lens_cfg["jlens"]
    path = lens_file()
    info = inspect_lens_file(path)
    checks.info["lens_file"] = path
    checks.info["raw"] = {k: v for k, v in info.items() if k != "J_layers"}

    first, last = spec["expected_source_layers"]
    expected_layers = list(range(first, last + 1))
    checks.check("keys include J, n_prompts, d_model", {"J", "n_prompts", "d_model"} <= set(info["keys"]), info["keys"])
    checks.check(f"d_model == {spec['expected_d_model']}", info["d_model"] == spec["expected_d_model"], info["d_model"])
    checks.check(f"J covers blocks {first}..{last}", info["J_layers"] == expected_layers,
                 f"{info['J_layers'][0]}..{info['J_layers'][-1]} ({len(info['J_layers'])})")
    checks.check("every J is d_model × d_model", info["J_shapes"] == [(spec["expected_d_model"], spec["expected_d_model"])], info["J_shapes"])
    checks.check("workspace layers are all fitted",
                 set(lens_cfg["workspace_layers"]) <= set(info["J_layers"]))
    provenance = info["provenance"] or {}
    if "capture_point" in provenance:
        checks.check("provenance capture point recorded", True, provenance["capture_point"])
    else:
        checks.info["capture_point_note"] = (
            "Not in the file. jlens.fitting records block outputs (forward hooks on "
            "model.layers[i]) for sources and target; readouts here use the same hooks."
        )

    hf_model, tokenizer, lens_model = load_model()
    lens = load_lens(path)
    layers = lens_cfg["workspace_layers"]
    final = lens_model.n_layers - 1
    top_k = lens_cfg["top_k"]

    # The LM head outputs bf16 logits, so exact ties are common (about 19 distinct
    # values among a top-25), and CPU and GPU topk break ties differently.
    # Compare logits directly, and top-k sets up to ties.
    logit_diffs, top1_ok, topk_ok, logit_lens_diffs, logit_lens_top1 = [], [], [], [], []
    for prompt in PROMPTS:
        # Comparison with jlens: tokenizing text is fine here, since both sides
        # read the same input_ids.
        ref_logits, _, input_ids = lens.apply(lens_model, prompt, layers=layers)
        ids = input_ids[0].tolist()
        resid = residuals(lens_model, ids, layers + [final])
        ours = readout(lens_model, {l: resid[l] for l in layers}, lens=lens, top_k=top_k)
        for li, layer in enumerate(layers):
            ref = ref_logits[layer].float()
            with torch.no_grad():
                our_logits = lens_model.unembed(lens.transport(resid[layer].float(), layer)).float().cpu()
            logit_diffs.append(float((our_logits - ref).abs().max()))
            our_ids = torch.from_numpy(ours["top_ids"][li]).long()
            kth = ref.topk(top_k, dim=-1).values[:, -1:]
            top1_ok.append(bool((ref.gather(1, our_ids[:, :1]) == ref.max(-1, keepdim=True).values).all()))
            topk_ok.append(bool((ref.gather(1, our_ids) >= kth).all()))

        # Logit lens (J = I) at the last block vs the HF model's own forward.
        with torch.no_grad():
            hf_lp = torch.log_softmax(hf_model(input_ids=input_ids).logits[0].float(), dim=-1).cpu()
            ours_lp = torch.log_softmax(lens_model.unembed(resid[final]).float(), dim=-1).cpu()
        logit_lens_top1.append(bool((hf_lp.argmax(-1) == ours_lp.argmax(-1)).all()))
        logit_lens_diffs.append(float((hf_lp - ours_lp).abs().max()))
        if prompt == PROMPTS[0]:
            checks.info["example_readout_prompt0_pos-2"] = {
                layer: [tokenizer.decode([t]) for t in ours["top_ids"][li][-2][:10]]
                for li, layer in enumerate(layers)
            }

    checks.check("our readout: logits identical to lens.apply (max |Δ| over the full vocab ≤ 1e-4)",
                 max(logit_diffs) <= 1e-4, max(logit_diffs))
    checks.check("our readout: top-1 has lens.apply's maximal logit at every layer and position (ties allowed)", all(top1_ok))
    checks.check("our readout: every top-25 token is in lens.apply's top-25 up to ties", all(topk_ok))
    checks.check("logit lens at last block: same top-1 as the model at every position", all(logit_lens_top1))
    checks.check("logit lens at last block: max |Δ log-prob| ≤ 0.01 nats", max(logit_lens_diffs) <= 0.01,
                 round(max(logit_lens_diffs), 5))
    checks.info["neuronpedia_agreement"] = "not run: compare manually at neuronpedia.org/jlens if it has Qwen3-8B"
    checks.finish()


if __name__ == "__main__":
    main()
