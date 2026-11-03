"""Apply a PEFT LoRA adapter to the loaded HF model at load time, without peft and
without merging: a forward hook on each target Linear adds
scale * B(A(x)) to its output, computed in the model dtype (bf16), the way vLLM
applies LoRA (unmerged, in the model dtype). Nothing is written to disk.

Only plain LoRA is supported (no DoRA, rsLoRA, bias, modules_to_save); anything
else raises (rule 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


class LoRAHooks:
    def __init__(self, hf_model: torch.nn.Module, adapter_dir: str | Path) -> None:
        adapter_dir = Path(adapter_dir)
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        bad = {k: cfg.get(k) for k in ("use_dora", "use_rslora", "modules_to_save", "layers_to_transform",
                                         "rank_pattern", "alpha_pattern")
               if cfg.get(k) not in (None, False, {}, [])}
        if cfg.get("peft_type") != "LORA" or cfg.get("bias", "none") != "none" or bad:
            raise ValueError(f"unsupported adapter config: peft_type {cfg.get('peft_type')}, bias {cfg.get('bias')}, {bad}")
        self.scale = cfg["lora_alpha"] / cfg["r"]
        self.config = cfg
        weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
        modules = dict(hf_model.named_modules())
        dtype = next(hf_model.parameters()).dtype
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, w in weights.items():
            if not key.startswith("base_model.model.") or not key.endswith((".lora_A.weight", ".lora_B.weight")):
                raise ValueError(f"unexpected adapter key {key}")
            name, ab = key[len("base_model.model."):].rsplit(".lora_", 1)
            pairs.setdefault(name, {})[ab[0]] = w
        self.handles = []
        self.n_modules = 0
        for name, ab in sorted(pairs.items()):
            mod = modules.get(name)
            if mod is None or not isinstance(mod, torch.nn.Linear) or set(ab) != {"A", "B"}:
                raise ValueError(f"adapter module {name} not found as a Linear in the model, or A/B missing")
            dev = mod.weight.device
            A = ab["A"].to(device=dev, dtype=dtype)  # [r, in]
            B = ab["B"].to(device=dev, dtype=dtype)  # [out, r]
            if A.shape[1] != mod.in_features or B.shape[0] != mod.out_features or A.shape[0] != cfg["r"]:
                raise ValueError(f"{name}: LoRA shapes {tuple(A.shape)}, {tuple(B.shape)} don't fit {mod}")
            self.handles.append(mod.register_forward_hook(self._hook(A, B)))
            self.n_modules += 1
        targets = {n.rsplit(".", 1)[-1] for n in pairs}
        if targets != set(cfg["target_modules"]):
            raise ValueError(f"adapter modules {targets} != target_modules {cfg['target_modules']}")

    def _hook(self, A: torch.Tensor, B: torch.Tensor):
        scale = self.scale

        def hook(module, inputs, output):
            x = inputs[0]
            delta = (x @ A.T) @ B.T
            return output + (delta * scale if scale != 1.0 else delta)
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def adapter_dir(arm: str) -> str | None:
    """Local path of an arm's adapter in the HF cache (downloads if missing); None for base."""
    from huggingface_hub import snapshot_download

    from jmr import config
    cfg = config.load("hack_organism")
    sub = cfg["arms"][arm]
    if sub is None:
        return None
    root = snapshot_download(cfg["adapters_repo"], revision=cfg["adapters_revision"],
                             allow_patterns=[f"{sub}/adapter_config.json", f"{sub}/adapter_model.safetensors"])
    return f"{root}/{sub}"
