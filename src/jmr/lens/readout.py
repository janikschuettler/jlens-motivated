"""Teacher-forced passes from saved token IDs, and lens readouts.

Residuals are block outputs, captured with jlens's own ActivationRecorder, so
they match the capture point the lens was fit at. Readouts are computed in
position chunks; only top-k and selected token log-probs leave this module
(CLAUDE.md rule 3).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from jlens import ActivationRecorder, HFLensModel, JacobianLens


@torch.no_grad()
def residuals(lens_model: HFLensModel, ids: Sequence[int], layers: Sequence[int]) -> dict[int, torch.Tensor]:
    """Block outputs [seq_len, d_model] at the given layers for exactly these IDs."""
    input_ids = torch.tensor([list(ids)], device=lens_model.input_device)
    with ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(input_ids)
        return {layer: recorder.activations[layer][0].detach() for layer in layers}


@torch.no_grad()
def _logprob_chunks(
    lens_model: HFLensModel,
    residual: torch.Tensor,
    *,
    lens: JacobianLens | None,
    layer: int,
    chunk: int,
):
    """Yield (start, log-probs [n, vocab] fp32) over position chunks. With lens
    None this is the logit lens (J = I)."""
    for start in range(0, residual.shape[0], chunk):
        h = residual[start : start + chunk].float()
        if lens is not None:
            h = lens.transport(h, layer)
        logits = lens_model.unembed(h).float()
        yield start, torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def readout(
    lens_model: HFLensModel,
    resid: dict[int, torch.Tensor],
    *,
    lens: JacobianLens | None,
    top_k: int,
    token_ids: Sequence[int] = (),
    chunk: int = 512,
    dtype=np.float16,
) -> dict[str, np.ndarray]:
    """Top-k tokens and selected token log-probs at every position and layer.

    Returns top_ids int32 [L, T, k], top_logprobs fp16 [L, T, k] and
    token_logprobs fp16 [L, T, len(token_ids)], layers in sorted order.
    dtype sets the stored log-prob precision (fp16 default; fp32 for validation)."""
    layers = sorted(resid)
    seq_len = resid[layers[0]].shape[0]
    top_ids = np.zeros((len(layers), seq_len, top_k), dtype=np.int32)
    top_lp = np.zeros((len(layers), seq_len, top_k), dtype=dtype)
    sel_lp = np.zeros((len(layers), seq_len, len(token_ids)), dtype=dtype)
    select = torch.tensor(list(token_ids), dtype=torch.long)
    for li, layer in enumerate(layers):
        for start, lp in _logprob_chunks(lens_model, resid[layer], lens=lens, layer=layer, chunk=chunk):
            end = start + lp.shape[0]
            values, indices = lp.topk(top_k, dim=-1)
            top_ids[li, start:end] = indices.cpu().numpy()
            top_lp[li, start:end] = values.cpu().numpy()
            if len(token_ids):
                sel_lp[li, start:end] = lp[:, select.to(lp.device)].cpu().numpy()
    return {"layers": np.array(layers), "top_ids": top_ids, "top_logprobs": top_lp, "token_logprobs": sel_lp}


@torch.no_grad()
def next_token_logprobs(
    lens_model: HFLensModel, final_resid: torch.Tensor, ids: Sequence[int], *, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Model log-prob of ids[t+1] given ids[:t+1], and the argmax, for every t.

    final_resid is the last block's output; unembedding it is the model's own
    forward (final norm + LM head). Arrays have length len(ids) - 1."""
    targets = torch.tensor(list(ids[1:]), device=final_resid.device)
    out_lp, out_argmax = [], []
    for start in range(0, len(ids) - 1, chunk):
        end = min(start + chunk, len(ids) - 1)
        logits = lens_model.unembed(final_resid[start:end]).float()
        lp = torch.log_softmax(logits, dim=-1)
        out_lp.append(lp.gather(1, targets[start:end, None])[:, 0].cpu())
        out_argmax.append(lp.argmax(-1).cpu())
    return torch.cat(out_lp).numpy(), torch.cat(out_argmax).numpy()
