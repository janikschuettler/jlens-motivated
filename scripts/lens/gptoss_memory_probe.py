"""Memory probe for gpt-oss-20b in the lens env (JMR_MODEL_PROFILE=gpt-oss-20b): weights after
load, parameter dtypes (MXFP4 kept or dequantised), attention implementation, and the peak GPU
memory of one teacher-forced pass at several lengths, using the readout's own residual capture.

  JMR_MODEL_PROFILE=gpt-oss-20b HF_HUB_CACHE=/dev/shm/hf/hub uv run --project envs/lens \
      python scripts/lens/gptoss_memory_probe.py --lengths 1024 2048 4096 6144 8192
"""

import argparse
import json

import torch

from jmr import config
from jmr.lens.load import load_model
from jmr.lens.readout import residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    a = ap.parse_args()
    hf_model, tok, lens_model = load_model()
    info = {
        "allocated_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "param_dtypes": sorted({str(p.dtype) for p in hf_model.parameters()}),
        "attn_implementation": getattr(hf_model.config, "_attn_implementation", None),
        "quantization_config": str(getattr(hf_model.config, "quantization_config", None))[:300],
        "expert_module": type(hf_model.model.layers[0].mlp.experts).__name__,
        "expert_attrs": sorted(k for k in vars(hf_model.model.layers[0].mlp.experts) if not k.startswith("_"))[:20],
    }
    print(json.dumps(info, indent=1), flush=True)
    layers = config.load("lens")["workspace_layers"] + [lens_model.n_layers - 1]
    for n in a.lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 199000, (n,)).tolist()
        try:
            residuals(lens_model, ids, layers)
            print(f"len {n}: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
        except torch.OutOfMemoryError as e:
            print(f"len {n}: OOM ({str(e)[:120]})", flush=True)
            break


if __name__ == "__main__":
    main()
