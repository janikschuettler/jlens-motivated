"""Part 1 of the steering test (GPU, lens env): steer at the answer position of
existing traces and read the model's answer letter.

For each selected run (jmr.steer.select_answer_runs: followed runs of the instructed,
hint and planted-claim groups, correct unhinted runs), teacher-force exactly the saved
token IDs up to the position whose next token is the final answer letter (pre_answer),
then add alpha * r_l * d at block l at that one position and read the next-token
distribution. Every (layer, direction, target, alpha) is one row of a batch that shares
the prefix's KV cache. Saved per variant: the four letters' log-probs (log-sum-exp over
" X", "X", "(X") and the top-1 token. No lens readout and no full-vocabulary output is
stored (rule 3 does not apply to model outputs, but nothing larger is kept anyway).

  uv run --project envs/lens python scripts/lens/steer_answer.py --dirs data/steer/directions.pt \
      --out data/steer/answer

Checks (rule 1, stop on failure): tokenizer fingerprint and model revision of every run;
anchors decode as in M0.4; the hook's capture point equals jlens's ActivationRecorder
(first run); the unsteered next-token log-prob of the saved answer token agrees with
vLLM's (median |diff| <= 0.05 over runs); the unsteered argmax letter equals the saved
answer in >= 95% of runs.
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from jmr import config, steer
from jmr.lens.batch import check_plan, plan_run, read_runs_lenient
from jmr.lens.load import load_model
from jmr.lens.readout import residuals
from jmr.lens.steering import Steerer
from jmr.tokens import fingerprint

KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2  # Qwen3-8B, bf16
KV_BUDGET = 8e9  # with the weights and the allocator cache this stays under 40 GB
MAX_ROWS = 32


def variants():
    """(layer, dir_type, primary, alpha) for every steered row; alpha 0 row first."""
    out = [(steer.ANSWER_PRIMARY_LAYER, "none", True, 0.0)]
    for l in steer.ANSWER_LAYERS:
        for t in steer.DIR_TYPES:
            for primary in (True, False):
                for a in steer.ANSWER_ALPHAS:
                    out.append((l, t, primary, a))
    return out


def variants_other():
    """Exploratory (added after part 1's results): push toward the *other* unchosen wrong
    letter at block 24, to compare with pushing toward the correct letter. primary is
    the string "other" in these rows."""
    out = [(steer.ANSWER_PRIMARY_LAYER, "none", "other", 0.0)]
    for t in ("jlens", "logit", "probe"):
        for a in steer.ANSWER_ALPHAS:
            out.append((steer.ANSWER_PRIMARY_LAYER, t, "other", a))
    return out


def other_letter(qid: str, letters, correct: str, hinted: str) -> str:
    """The unchosen wrong letter used as the 'other' target: one of the two letters that
    are neither correct nor B, fixed by the item ID."""
    cands = [x for x in letters if x not in (correct, hinted)]
    return cands[zlib.crc32(qid.encode()) % len(cands)]


def random_k(qid: str) -> int:
    return zlib.crc32(qid.encode()) % steer.N_RANDOM


def select(groups, items):
    runs = []
    for g in groups:
        d, _ = steer.SOURCES[g]
        labels = [json.loads(l) for l in open(Path(d) / "labels.jsonl")]
        want = set(steer.select_answer_runs(labels, g, items))
        for rec, extra in read_runs_lenient(Path(d) / "runs.jsonl"):
            if rec.run_id in want:
                runs.append((g, rec, extra))
        print(f"{g}: {sum(1 for x in runs if x[0] == g)} runs", flush=True)
    return runs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="data/steer/directions.pt")
    ap.add_argument("--out", default="data/steer/answer")
    ap.add_argument("--groups", nargs="+", default=["instructed", "hint", "planted", "unhinted"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["main", "other"], default="main")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config.snapshot(out)
    D = torch.load(a.dirs)
    item_rows = {json.loads(l)["id"]: json.loads(l) for l in open(steer.ITEMS_FILE)}
    items = set(item_rows)
    runs = select(a.groups, items)
    if a.limit:
        runs = runs[:a.limit]
    V = variants() if a.mode == "main" else variants_other()
    sfx = "" if a.mode == "main" else "_other"
    (out / f"variants{sfx}.json").write_text(json.dumps(V))

    m = config.load("model")
    hf_model, tok, lens_model = load_model(a.device)
    fp = fingerprint(tok)
    think_ids = (tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>"))
    tr = config.load("lens")["tracked_tokens"]["ids"]
    letter_ids = torch.tensor([[tr[x][f] for f in ("space", "bare", "paren")] for x in steer.LETTERS])
    id2letter = {tr[x][f]: x for x in steer.LETTERS for f in ("space", "bare", "paren")}
    layers = steer.ANSWER_LAYERS
    dev = hf_model.device
    dirs = {l: {t: D["dirs"][l][t].to(dev) for t in steer.DIR_TYPES} for l in layers}
    r = {l: float(D["r"][l]) for l in layers}
    st = Steerer(hf_model, layers)

    path = out / f"results{sfx}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["run_id"] for l in open(path)}
    lp_diffs, agree, t0 = [], [], time.time()
    checked_capture = False
    with open(path, "a") as f:
        for n, (g, rec, extra) in enumerate(runs):
            if rec.run_id in done:
                continue
            if rec.tokenizer_fingerprint != fp or rec.model.get("revision") != m["revision"]:
                raise ValueError(f"{rec.run_id}: tokenizer or model revision mismatch (rule 1)")
            plan = plan_run(rec, tok, think_ids=think_ids, prompt_tail=64, n_cot=16, resid_anchors=[])
            probs = check_plan(plan, tok)
            if probs:
                raise ValueError(f"{rec.run_id}: anchor problems {probs} (rule 1)")
            ids, P = plan.ids, plan.prompt_len
            # unhinted runs: B = the item's (unshown) hinted option, as in the readouts
            hinted = rec.hint_letter or item_rows[rec.question_id]["hinted"]
            if rec.hint_letter is not None and rec.hint_letter != item_rows[rec.question_id]["hinted"]:
                raise ValueError(f"{rec.run_id}: hint letter differs from the item's hinted option (rule 1)")
            p = plan.named["pre_answer"]
            if p < 0 or ids[p + 1] not in id2letter or id2letter[ids[p + 1]] != plan.answer:
                raise ValueError(f"{rec.run_id}: answer token at pre_answer+1 is not the tracked answer letter")
            if not checked_capture:
                # Capture point: our hook sees the same block outputs as jlens's recorder.
                cap = {}
                hs = [hf_model.model.layers[l].register_forward_hook(
                    lambda mod, i, o, l=l: cap.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0].clone()))
                    for l in layers]
                hf_model(input_ids=torch.tensor([ids[:p + 1]], device=dev), logits_to_keep=1)
                for h in hs:
                    h.remove()
                ref = residuals(lens_model, ids[:p + 1], layers)
                diff = max(float((cap[l].float() - ref[l].float()).abs().max()) for l in layers)
                if diff != 0.0:
                    raise ValueError(f"hook capture point differs from jlens's recorder: max |diff| {diff} (rule 1)")
                print("capture point check: hook == ActivationRecorder (max |diff| 0.0)", flush=True)
                checked_capture = True

            cache = DynamicCache()
            st.clear()
            hf_model(input_ids=torch.tensor([ids[:p]], device=dev), past_key_values=cache, use_cache=True,
                     logits_to_keep=1)
            rows = int(max(1, min(MAX_ROWS, KV_BUDGET // (p * KV_BYTES_PER_TOKEN))))
            cache.batch_repeat_interleave(rows)
            x = torch.full((rows, 1), ids[p], device=dev)
            res = []
            for c0 in range(0, len(V), rows):
                chunk = V[c0:c0 + rows]
                vecs = {l: torch.zeros(rows, hf_model.config.hidden_size, device=dev) for l in layers}
                for j, (l, t, primary, alpha) in enumerate(chunk):
                    if t == "none":
                        continue
                    if t == "random":
                        v = dirs[l]["random"][random_k(rec.question_id)] * (1 if primary else -1)
                    else:
                        if primary == "other":
                            letter, sign = other_letter(rec.question_id, rec.letters, rec.correct_letter, hinted), 1
                        else:
                            letter, sign = steer.target(g, rec.correct_letter, hinted, primary)
                        v = sign * dirs[l][t][steer.LETTERS.index(letter)]
                    vecs[l][j] = alpha * r[l] * v
                st.set(vecs)
                o = hf_model(input_ids=x, past_key_values=cache, use_cache=True)
                st.clear()
                cache.crop(p)
                lp = torch.log_softmax(o.logits[:, -1].float(), -1)
                let = torch.logsumexp(lp[:, letter_ids.to(dev)], -1)  # [rows, 4]
                top = lp.argmax(-1)
                for j in range(len(chunk)):
                    res.append([round(float(v_), 4) for v_ in let[j]] + [int(top[j])])
                if c0 == 0:
                    tf_lp = float(lp[0, ids[p + 1]])
            del cache
            vl = rec.completion_logprobs[p + 1 - P]
            lp_diffs.append(abs(tf_lp - vl))
            base = res[0][:4]
            agree.append(steer.LETTERS[int(np.argmax(base))] == plan.answer)
            f.write(json.dumps({"run_id": rec.run_id, "group": g, "arm": extra.get("arm"),
                                "question_id": rec.question_id, "correct": rec.correct_letter,
                                "hinted": hinted, "answer": plan.answer, "pre_answer": p,
                                "seq_len": len(ids), "tf_vs_vllm_answer_lp": tf_lp - vl, "rows": rows,
                                "random_k": random_k(rec.question_id), "res": res,
                                "other": other_letter(rec.question_id, rec.letters, rec.correct_letter, hinted)}) + "\n")
            f.flush()
            if len(lp_diffs) == 20 and (np.median(lp_diffs) > 0.05 or np.mean(agree) < 0.95):
                raise SystemExit(f"unsteered answer check failed after 20 runs: median |tf-vllm| "
                                 f"{np.median(lp_diffs):.4f}, agree {np.mean(agree):.3f} (rule 1)")
            if len(lp_diffs) % 20 == 0:
                print(f"{n + 1}/{len(runs)} {time.time() - t0:.0f}s median |tf-vllm| {np.median(lp_diffs):.4f} "
                      f"agree {np.mean(agree):.3f}", flush=True)
    summ = {"n": len(lp_diffs), "median_abs_tf_vllm": float(np.median(lp_diffs)) if lp_diffs else None,
            "p99_abs_tf_vllm": float(np.quantile(lp_diffs, 0.99)) if lp_diffs else None,
            "argmax_agrees_with_answer": float(np.mean(agree)) if agree else None, "seconds": time.time() - t0,
            "r": r, "model": {"hf_id": m["hf_id"], "revision": m["revision"]}}
    (out / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)
    if lp_diffs and (summ["median_abs_tf_vllm"] > 0.05 or summ["argmax_agrees_with_answer"] < 0.95):
        raise SystemExit("unsteered answer check failed (rule 1)")


if __name__ == "__main__":
    main()
