"""Run M&B's main.py unchanged, after seeding torch/numpy/random (their code sets no seed).

Usage (cwd must be the M&B clone, because their code reads core/configs/*.json relatively):
  cd third_party/motivated-reasoning && MOTIVATION_HOME=... MB_LOCAL_DATASET=... \
  MB_MODEL_REVISION=... .venv/bin/python -I /root/jlens-motivated/scripts/mb/seeded_main.py \
      --seed 0 -- --model qwen-3-8b --dataset mmlu --generate --bias expert --hint_idx 0 ...
"""
import json
import os
import random
import runpy
import sys
import time

argv = sys.argv[1:]
assert argv[0] == "--seed" and argv[2] == "--", argv
seed = int(argv[1])
main_args = argv[3:]
repo = os.getcwd()
assert os.path.exists(os.path.join(repo, "core", "motivated_reasoning.py")), repo
sys.path.insert(0, repo)

import numpy as np  # noqa: E402
import torch  # noqa: E402

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
t0 = time.time()
sys.argv = [os.path.join(repo, "main.py")] + main_args
runpy.run_path(sys.argv[0], run_name="__main__")
elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
rec = {"args": main_args, "seed": seed, "elapsed_s": round(elapsed, 1), "peak_mem_gib": round(peak, 2),
       "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(os.environ["MOTIVATION_HOME"], "timing.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")
print("TIMING", json.dumps(rec))
