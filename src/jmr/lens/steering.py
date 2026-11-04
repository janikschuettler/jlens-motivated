"""Adding steering vectors to block outputs (lens env, HF transformers).

The vector is added to the output of block l (model.model.layers[l]), the capture
point the J-lens was fit at (checked against jlens's ActivationRecorder by
scripts/lens/steer_answer.py before any steering). Per-row vectors, so one batch can
hold different directions. Where it is added:
  cols = "last": the last column of every forward call. In a prefill this is the last
                 prompt position; in a one-token decode step it is the new position.
                 So during generation every position whose output produces a generated
                 token is steered, and no prompt position before the last one is.
  cols = mask:   a bool tensor [B, T] for a full forward (forced answers).
"""

from __future__ import annotations

import torch


class Steerer:
    def __init__(self, hf_model, layers: list[int]):
        self.vec: dict[int, torch.Tensor | None] = {l: None for l in layers}
        self.cols: str | torch.Tensor = "last"
        self.handles = [hf_model.model.layers[l].register_forward_hook(self._hook(l)) for l in layers]

    def set(self, vecs: dict[int, torch.Tensor | None], cols: str | torch.Tensor = "last") -> None:
        for l in self.vec:
            self.vec[l] = vecs.get(l)
        self.cols = cols

    def clear(self) -> None:
        self.set({}, "last")

    def _hook(self, l: int):
        def hook(module, inputs, output):
            v = self.vec[l]
            if v is None:
                return output
            t = output if torch.is_tensor(output) else output[0]
            if v.shape[0] != t.shape[0]:
                raise ValueError(f"block {l}: {v.shape[0]} steering rows for batch {t.shape[0]}")
            if isinstance(self.cols, str):
                t[:, -1, :] += v.to(t.dtype)
            else:
                m = self.cols.to(t.device)
                t += (m[:, :, None].to(t.dtype) * v[:, None, :].to(t.dtype))
            return output
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
