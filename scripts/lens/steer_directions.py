"""Build the steering directions (CPU, lens env): J-lens, logit-lens, probe and random
directions for the four option letters at blocks 16, 20, 24, 28, plus the norm scale
r_l. Definitions in jmr.steer. No outcome data is read here: the probe is trained on
unhinted runs of known items outside the steering test set (block outputs at the
answer position, from existing readouts).

  uv run --project envs/lens python scripts/lens/steer_directions.py --out data/steer

Writes data/steer/directions.pt (directions only, derived from the weights and the lens;
gitignored) and data/steer/directions_meta.json (cosines, how well each direction
*reads* the answer letter at the answer position, r_l, counts).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from jmr import config, steer
from jmr.lens.load import lens_file, load_lens

READOUTS = ["data/readouts/instr_train_extra", "data/readouts/instr_ct_core", "data/readouts/instr_nc_core"]


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def centre(v: torch.Tensor) -> torch.Tensor:
    return v - v.mean(0, keepdim=True)


def load_probe_data(layers, exclude: set[str], known: set[str]):
    """Block outputs at pre_answer (and CoT anchors, for the norm) of unhinted runs whose
    final answer is the correct letter, on known items outside `exclude`."""
    H = {l: [] for l in layers}
    cot_norms = {l: [] for l in layers}
    ys, items, seen = [], [], set()
    for d in READOUTS:
        for line in open(Path(d) / "index.jsonl"):
            m = json.loads(line)
            q = m["question_id"]
            if (m["condition"] != "unhinted" or m["run_id"] in seen or q in exclude or q not in known
                    or m.get("answer") != m["correct_letter"]):
                continue
            z = np.load(Path(d) / m["file"])
            k = m["resid_anchor_names"].index("pre_answer")
            if z["resid_pos"][k] < 0:
                continue
            seen.add(m["run_id"])
            for l in layers:
                H[l].append(z["resid"][l, k].astype(np.float32))
            if "cot_resid" in z.files:
                cl = list(m["cot_resid_layers"])
                ok = z["cot_resid_pos"] >= 0
                for l in layers:
                    cot_norms[l] += list(np.linalg.norm(z["cot_resid"][cl.index(l)][ok].astype(np.float32), axis=-1))
            ys.append(steer.LETTERS.index(m["correct_letter"]))
            items.append(q)
    return {l: np.stack(v) for l, v in H.items()}, np.array(ys), np.array(items), cot_norms


def diff_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = np.stack([X[y == k].mean(0) for k in range(4)])
    return mu - mu.mean(0, keepdims=True)


def read_acc(D: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(((X @ D.T).argmax(1) == y).mean())


def cv_acc(X: np.ndarray, y: np.ndarray, items: np.ndarray, k: int = 5) -> float:
    uniq = np.array(sorted(set(items)))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fold = {q: i % k for i, q in enumerate(uniq)}
    f = np.array([fold[q] for q in items])
    hit = np.zeros(len(y), bool)
    for i in range(k):
        D = diff_means(X[f != i], y[f != i])
        hit[f == i] = (X[f == i] @ D.T).argmax(1) == y[f == i]
    # item-level mean (each item counts once)
    by = defaultdict(list)
    for q, h in zip(items, hit):
        by[q].append(h)
    return float(np.mean([np.mean(v) for v in by.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/steer")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = sorted(set(steer.ANSWER_LAYERS) | {steer.GEN_LAYER})

    pool = [json.loads(l) for l in open(steer.POOL_FILE)]
    known = steer.known_ids(pool)
    test_items = [json.loads(l)["id"] for l in open(steer.ITEMS_FILE)]
    if test_items != known[:len(test_items)]:
        raise ValueError("test items are not the first known pool items (rule 1)")
    exclude = set(known[:steer.N_EXCLUDE_FROM_PROBE])

    m = config.load("model")
    model = AutoModelForCausalLM.from_pretrained(m["hf_id"], revision=m["revision"], dtype=torch.bfloat16,
                                                 device_map="cpu")
    g = model.model.norm.weight.float()
    W = model.lm_head.weight.float()
    ids = config.load("lens")["tracked_tokens"]["ids"]
    U = torch.stack([torch.stack([W[ids[x][f]] for f in ("space", "bare", "paren")]).mean(0) for x in steer.LETTERS])
    lens = load_lens(lens_file())

    H, y, items, cot_norms = load_probe_data(layers, exclude, set(known))
    print(f"probe data: {len(y)} runs on {len(set(items))} items; letters {np.bincount(y, minlength=4).tolist()}",
          flush=True)

    dirs, meta = {}, {"layers": layers, "n_probe_runs": int(len(y)), "n_probe_items": len(set(items)),
                      "probe_letter_counts": np.bincount(y, minlength=4).tolist(), "r": {}, "cos": {},
                      "read_acc": {}, "probe_cv_item_acc": {}, "norm_convention":
                      "jlens = J_l^T (g * u_X), logit = g * u_X; u_X = mean LM-head row over ' X', 'X', '(X'; "
                      "centred over the four letters, unit norm"}
    for l in layers:
        J = lens.jacobians[l].float()
        vj = unit(centre((U * g) @ J))  # rows J^T (g u_X)
        vl = unit(centre(U * g))
        vp = unit(torch.from_numpy(diff_means(H[l], y)).float())
        gen = torch.Generator().manual_seed(1000 + l)
        vr = unit(torch.randn(steer.N_RANDOM, W.shape[1], generator=gen))
        dirs[l] = {"jlens": vj, "logit": vl, "probe": vp, "random": vr}
        r = float(np.median(cot_norms[l]))
        meta["r"][l] = r
        meta["cos"][l] = {f"{a_}-{b_}": [float((dirs[l][a_][i] * dirs[l][b_][i]).sum()) for i in range(4)]
                          for a_, b_ in (("jlens", "logit"), ("jlens", "probe"), ("logit", "probe"))}
        meta["cos"][l]["random-jlens_max_abs"] = float((vr @ vj.T).abs().max())
        Xl = H[l]
        meta["read_acc"][l] = {t: read_acc(dirs[l][t].numpy(), Xl, y) for t in ("jlens", "logit", "probe")}
        meta["read_acc"][l]["probe_note"] = "probe read_acc is in-sample; see probe_cv_item_acc"
        meta["probe_cv_item_acc"][l] = cv_acc(Xl, y, items)
        print(l, "r", round(r, 1), "cos", {k: np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)
                                           for k, v in meta["cos"][l].items()},
              "read", meta["read_acc"][l], "probe cv", round(meta["probe_cv_item_acc"][l], 3), flush=True)
    meta["n_cot_norms"] = {l: len(v) for l, v in cot_norms.items()}
    meta["model"] = {"hf_id": m["hf_id"], "revision": m["revision"]}
    meta["lens"] = config.load("lens")["jlens"]
    torch.save({"dirs": dirs, "r": meta["r"], "layers": layers}, out / "directions.pt")
    (out / "directions_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("wrote", out / "directions.pt")


if __name__ == "__main__":
    main()
