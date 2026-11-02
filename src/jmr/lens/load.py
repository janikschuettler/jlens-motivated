"""Loading the model and the lens in the lens env."""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from jlens import HFLensModel, JacobianLens, from_hf
from transformers import AutoModelForCausalLM, AutoTokenizer

from jmr import config


def load_model(device: str = "cuda"):
    """The pinned Qwen3-8B in bf16, plus its tokenizer and a jlens wrapper.

    jlens's wrapper sets requires_grad=False on all parameters; the HF model is
    still usable for ordinary forward passes."""
    cfg = config.load("model")
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]), device_map=device
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"])
    lens_model = from_hf(hf_model, tokenizer, force_bos=False)
    return hf_model, tokenizer, lens_model


def lens_file() -> str:
    cfg = config.load("lens")["jlens"]
    return hf_hub_download(cfg["repo"], cfg["file"], revision=cfg["revision"])


def inspect_lens_file(path: str) -> dict:
    """Describe the raw checkpoint without going through jlens."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    J = raw.get("J", {})
    return {
        "keys": sorted(raw),
        "n_prompts": raw.get("n_prompts"),
        "d_model": raw.get("d_model"),
        "source_layers": raw.get("source_layers"),
        "J_layers": sorted(J),
        "J_shapes": sorted({tuple(t.shape) for t in J.values()}),
        "J_dtypes": sorted({str(t.dtype) for t in J.values()}),
        "provenance": raw.get("provenance"),
    }


def load_lens(path: str) -> JacobianLens:
    return JacobianLens.load(path)


__all__ = ["HFLensModel", "inspect_lens_file", "lens_file", "load_lens", "load_model"]
