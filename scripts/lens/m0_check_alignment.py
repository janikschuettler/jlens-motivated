"""M0.4 (lens env): token alignment between vLLM generation and teacher forcing.

For every run in the run set: teacher-force exactly the saved IDs, compare the
model's log-prob of each sampled token with vLLM's, locate the anchors, and check
that they decode to the intended tokens. Also reports per hint type (M0.5).

uv run --project envs/lens python scripts/lens/m0_check_alignment.py --run-set m0
"""

import argparse

import numpy as np

from jmr import config
from jmr.anchors import find_anchors
from jmr.checks import Checks
from jmr.label import extract_answer
from jmr.lens.load import load_model
from jmr.lens.readout import next_token_logprobs, residuals
from jmr.records import read_runs, runs_path
from jmr.tokens import fingerprint, token_bytes

MEDIAN_MAX, P99_MAX, SHIFT_FACTOR = 0.05, 0.5, 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-set", default="m0")
    args = parser.parse_args()
    records = read_runs(runs_path(config.run_set_dir(args.run_set)))

    checks = Checks("m0_4_alignment")
    model_cfg = config.load("model")
    hf_model, tokenizer, lens_model = load_model()
    final = lens_model.n_layers - 1
    think_start = tokenizer.convert_tokens_to_ids(model_cfg["special_tokens"]["think_start"])
    think_end = tokenizer.convert_tokens_to_ids(model_cfg["special_tokens"]["think_end"])

    fp = fingerprint(tokenizer)
    gen_fps = {r.tokenizer_fingerprint for r in records}
    checks.check("tokenizer fingerprint matches the gen env's", gen_fps == {fp}, {"lens": fp, "gen": sorted(gen_fps)})

    all_diffs, all_shifted, per_run = [], [], []
    bytes_mismatch, retok_mismatch, argmax_agree = 0, 0, []
    for r in records:
        ids = r.token_ids
        P = len(r.prompt_token_ids)
        if b"".join(token_bytes(tokenizer, ids)) != tokenizer.decode(ids, skip_special_tokens=False).encode():
            bytes_mismatch += 1
        if tokenizer(tokenizer.decode(ids, skip_special_tokens=False), add_special_tokens=False).input_ids != ids:
            retok_mismatch += 1

        final_resid = residuals(lens_model, ids, [final])[final]
        tf_lp, tf_argmax = next_token_logprobs(lens_model, final_resid, ids)
        n = len(r.completion_token_ids)
        vllm_lp = np.array(r.completion_logprobs)
        tf = tf_lp[P - 1 : P - 1 + n]  # log-prob of completion token i comes from position P - 1 + i
        diffs = np.abs(tf - vllm_lp)
        shifted = np.abs(tf_lp[P : P - 1 + n] - vllm_lp[: n - 1])  # paired with the next token's log-prob
        argmax_agree.append(float((tf_argmax[P - 1 : P - 1 + n] == np.array(r.completion_token_ids)).mean()))
        all_diffs.append(diffs)
        all_shifted.append(shifted)

        answer = extract_answer(r)
        anchors = find_anchors(tokenizer, ids, P, think_start_id=think_start, think_end_id=think_end, final_letter=answer)

        def dec(i):
            return None if i is None else tokenizer.decode([ids[i]])

        problems = []
        if dec(anchors.think_start) != "<think>":
            problems.append(f"think_start decodes to {dec(anchors.think_start)!r}")
        if r.finish_reason == "stop" and dec(anchors.think_end) != "</think>":
            problems.append(f"think_end decodes to {dec(anchors.think_end)!r}")
        if answer is not None and (anchors.final_answer is None or answer not in dec(anchors.final_answer)):
            problems.append(f"final_answer decodes to {dec(anchors.final_answer)!r}, answer {answer}")
        if anchors.answer_first_stated is not None and answer not in dec(anchors.answer_first_stated):
            problems.append(f"answer_first_stated decodes to {dec(anchors.answer_first_stated)!r}")
        context = {}
        for name, i in anchors.as_dict().items():
            if name != "prompt_len" and i is not None:
                context[name] = tokenizer.decode(ids[max(0, i - 6) : i + 3])
        run_ok = (not problems) and np.median(diffs) <= MEDIAN_MAX and np.quantile(diffs, 0.99) <= P99_MAX
        per_run.append({
            "run_id": r.run_id,
            "condition": r.condition,
            "finish_reason": r.finish_reason,
            "n_completion": n,
            "median_diff": float(np.median(diffs)),
            "p99_diff": float(np.quantile(diffs, 0.99)),
            "max_diff": float(diffs.max()),
            "anchors": anchors.as_dict(),
            "anchor_context": context,
            "problems": problems,
            "ok": bool(run_ok),
        })

    diffs = np.concatenate(all_diffs)
    shifted = np.concatenate(all_shifted)
    median, p99 = float(np.median(diffs)), float(np.quantile(diffs, 0.99))
    checks.info["logprob_diff"] = {
        "n_tokens": int(diffs.size), "median": median, "mean": float(diffs.mean()), "p99": p99, "max": float(diffs.max()),
        "shifted_median": float(np.median(shifted)), "tf_argmax_equals_sampled": float(np.mean(argmax_agree)),
    }
    checks.check("token bytes reproduce tokenizer.decode for every run", bytes_mismatch == 0, bytes_mismatch)
    checks.info["retokenization_mismatch_runs"] = f"{retok_mismatch}/{len(records)}"
    checks.check(f"median |Δ log-prob| ≤ {MEDIAN_MAX}", median <= MEDIAN_MAX, round(median, 5))
    checks.check(f"99th percentile |Δ log-prob| ≤ {P99_MAX}", p99 <= P99_MAX, round(p99, 4))
    checks.check(f"one-position shift makes the median ≥ {SHIFT_FACTOR:.0f}× worse",
                 np.median(shifted) >= SHIFT_FACTOR * median, round(float(np.median(shifted)), 4))
    anchor_problems = [p for p in per_run if p["problems"]]
    checks.check("anchors decode to the intended tokens in every run", not anchor_problems,
                 [(p["run_id"], p["problems"]) for p in anchor_problems][:10])
    for condition in sorted({r.condition for r in records}):
        ok = [p for p in per_run if p["condition"] == condition and p["ok"]]
        checks.check(f"{condition}: at least one run passes all alignment checks", bool(ok),
                     f"{len(ok)}/{sum(p['condition'] == condition for p in per_run)}")
    checks.info["per_run"] = per_run
    checks.finish()


if __name__ == "__main__":
    main()
