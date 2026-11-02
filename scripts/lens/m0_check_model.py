"""M0.1 (lens env): load Qwen3-8B, check its config, fingerprint the tokenizer,
measure peak GPU memory for an 8k-token pass with hooks at the workspace layers.

uv run --project envs/lens python scripts/lens/m0_check_model.py
"""

import torch
from transformers import AutoConfig

from jmr import config
from jmr.checks import Checks
from jmr.lens.load import load_model
from jmr.lens.readout import residuals
from jmr.tokens import fingerprint


def main() -> None:
    checks = Checks("m0_1_model_lens_env")
    model_cfg = config.load("model")
    hf_config = AutoConfig.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"])
    for key, expected in model_cfg["expected"].items():
        actual = getattr(hf_config, key)
        checks.check(f"config.{key} == {expected}", actual == expected, actual)

    hf_model, tokenizer, lens_model = load_model()
    checks.info["torch"] = torch.__version__
    checks.info["transformers"] = __import__("transformers").__version__
    checks.info["tokenizer_fingerprint"] = fingerprint(tokenizer)
    checks.info["think_token_ids"] = {
        name: tokenizer.convert_tokens_to_ids(tok) for name, tok in model_cfg["special_tokens"].items()
    }
    checks.info["bos_token_id"] = tokenizer.bos_token_id
    checks.check("lm_head is untied from the embedding",
                 hf_model.lm_head.weight.data_ptr() != hf_model.model.embed_tokens.weight.data_ptr())
    checks.check("model dtype is bf16", next(hf_model.parameters()).dtype == torch.bfloat16)
    checks.info["gpu_memory_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)

    layers = config.load("lens")["workspace_layers"] + [lens_model.n_layers - 1]
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 150000, (8192,)).tolist()
    resid = residuals(lens_model, ids, layers)
    checks.check("8k-token pass returns [8192, 4096] residuals",
                 all(tuple(r.shape) == (8192, 4096) for r in resid.values()))
    peak = torch.cuda.max_memory_allocated() / 1e9
    checks.info["peak_gpu_memory_8k_pass_gb"] = round(peak, 2)
    checks.check("8k-token pass fits in 22 GB", peak < 22, f"{peak:.2f} GB")
    checks.finish()


if __name__ == "__main__":
    main()
