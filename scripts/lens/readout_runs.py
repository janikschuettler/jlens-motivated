"""Batch J-lens / logit-lens readouts over a runs.jsonl (lens env).

Teacher-forces exactly the saved prompt + completion token IDs of every run
and writes one compact .npz per run plus index.jsonl and summary.json to the
output directory (format: jmr.lens.batch, loader: jmr.analysis.readouts).
Resumable: runs already in index.jsonl with their .npz present are skipped.
Stops on the first failing run (rule 1); nothing is skipped or truncated.

uv run --project envs/lens python scripts/lens/readout_runs.py \
    --runs data/runs/temp_check/runs.jsonl --out data/readouts/temp_check [--validate 3]
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from jmr import config
from jmr.lens import readout as single
from jmr.lens.batch import LENSES, BatchReader, check_plan, read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.tokens import fingerprint

# M0.4 tolerances (roadmap).
MEDIAN_MAX, P99_MAX, SHIFT_FACTOR = 0.05, 0.5, 10.0


def file_name(run_id: str) -> str:
    return hashlib.sha1(run_id.encode()).hexdigest()[:20] + ".npz"


def save_npz(path: Path, arrays: dict, compress: bool) -> int:
    tmp = path.with_name(path.name + ".tmp.npz")
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)
    return path.stat().st_size


def topk_equal_up_to_ties(ids_a, lp_a, ids_b, lp_b) -> bool:
    """Same sorted values, and the same IDs above the smallest kept value."""
    if not np.array_equal(np.sort(lp_a), np.sort(lp_b)):
        return False
    cut = lp_a.min()
    return set(ids_a[lp_a > cut].tolist()) == set(ids_b[lp_b > cut].tolist())


def validate_against_single(reader: BatchReader, lens_model, lens, record, plan, arrays) -> dict:
    """Recompute this run with the M0.2-validated single-run readout
    (jmr.lens.readout) and compare."""
    ids, lo = plan.ids, plan.stored_lo
    layers = reader.layers
    resid = single.residuals(lens_model, ids, layers + [reader.final])
    res = {}
    for si, name in enumerate(LENSES):
        ref = single.readout(lens_model, {l: resid[l] for l in layers}, lens=lens if name == "jlens" else None,
                             top_k=reader.top_k, token_ids=reader.tracked.ids, chunk=reader.chunk)
        ref_lp = ref["token_logprobs"][:, lo:].astype(np.float32)
        ours = arrays["lp"][si].astype(np.float32)
        top_ok = top_n = 0
        for a, pos in enumerate(arrays["top_pos"]):
            if pos < 0:
                continue
            for li in range(len(layers)):
                top_n += 1
                top_ok += topk_equal_up_to_ties(arrays["top_ids"][si, li, a], arrays["top_lp"][si, li, a],
                                                ref["top_ids"][li, pos], ref["top_logprobs"][li, pos])
        # Ranks: recompute with one plain masked count per tracked token.
        rank_mismatch = 0
        tracked = torch.tensor(reader.tracked.ids, device=reader.device)
        for li, layer in enumerate(layers):
            r = resid[layer]
            for start in range(0, r.shape[0], reader.chunk):
                end = min(start + reader.chunk, r.shape[0])
                if end <= lo:
                    continue
                h = r[start:end].float()
                if name == "jlens":
                    h = lens.transport(h, layer)
                lp = torch.log_softmax(lens_model.unembed(h).float(), dim=-1)[max(lo, start) - start:]
                tl = lp[:, tracked]
                eager = torch.stack([(lp > tl[:, k:k + 1]).sum(-1) + 1 for k in range(tl.shape[1])], 1)
                stored = torch.from_numpy(arrays["rank"][si, li, max(lo, start) - lo:end - lo]).to(eager.device)
                rank_mismatch += int((eager != stored).sum())
        if "x_ids" in arrays:  # per-run content tokens: same checks
            xids = arrays["x_ids"].tolist()
            ref_x = single.readout(lens_model, {l: resid[l] for l in layers}, lens=lens if name == "jlens" else None,
                                   top_k=reader.top_k, token_ids=xids, chunk=reader.chunk)
            x_diff = float(np.abs(ref_x["token_logprobs"][:, lo:].astype(np.float32)
                                  - arrays["x_lp"][si].astype(np.float32)).max())
            x_mis = 0
            xt = torch.tensor(xids, device=reader.device)
            for li, layer in enumerate(layers):
                r = resid[layer]
                for start in range(0, r.shape[0], reader.chunk):
                    end = min(start + reader.chunk, r.shape[0])
                    if end <= lo:
                        continue
                    h = r[start:end].float()
                    if name == "jlens":
                        h = lens.transport(h, layer)
                    lp = torch.log_softmax(lens_model.unembed(h).float(), dim=-1)[max(lo, start) - start:]
                    tl = lp[:, xt]
                    eager = torch.stack([(lp > tl[:, k:k + 1]).sum(-1) + 1 for k in range(tl.shape[1])], 1)
                    stored = torch.from_numpy(arrays["x_rank"][si, li, max(lo, start) - lo:end - lo]).to(eager.device)
                    x_mis += int((eager != stored).sum())
        res[name] = {"rank_mismatches_vs_eager_count": rank_mismatch,
                     "tracked_lp_max_absdiff": float(np.abs(ref_lp - ours).max()),
                     "tracked_lp_frac_identical": float((ref_lp == ours).mean()),
                     "top25_equal_up_to_ties": f"{top_ok}/{top_n}"}
        if "x_ids" in arrays:
            res[name].update({"content_lp_max_absdiff": x_diff, "content_rank_mismatches_vs_eager_count": x_mis})
    # Logit lens at the last block = the model's own distribution.
    ref = single.readout(lens_model, {reader.final: resid[reader.final]}, lens=None, top_k=reader.top_k,
                         token_ids=reader.tracked.ids, chunk=reader.chunk)
    d = np.abs(ref["token_logprobs"][0, lo:].astype(np.float32) - arrays["model_lp"].astype(np.float32))
    tf_lp, _ = single.next_token_logprobs(lens_model, resid[reader.final], ids, chunk=reader.chunk)
    P = plan.prompt_len
    d2 = np.abs(tf_lp[P - 1:] - arrays["completion_tf_logprob"])
    res["model"] = {"tracked_lp_max_absdiff_vs_logit_lens_block35": float(d.max()),
                    "sampled_token_lp_max_absdiff_vs_next_token_logprobs": float(d2.max())}
    # Stored anchor block outputs (own hooks) = jlens's ActivationRecorder block
    # outputs at the same positions, for the layers captured here.
    rd, n = 0.0, 0
    for j, pos in enumerate(arrays["resid_pos"]):
        if pos < 0:
            continue
        for layer in layers + [reader.final]:
            want = resid[layer][int(pos)].half().cpu().numpy().astype(np.float32)
            rd = max(rd, float(np.abs(arrays["resid"][layer, j].astype(np.float32) - want).max()))
            n += 1
    if "cot_resid" in arrays:
        for k, pos in enumerate(arrays["cot_resid_pos"]):
            if pos < 0:
                continue
            for li, layer in enumerate(layers):
                want = resid[layer][int(pos)].half().cpu().numpy().astype(np.float32)
                rd = max(rd, float(np.abs(arrays["cot_resid"][li, k].astype(np.float32) - want).max()))
                n += 1
    res["resid"] = {"n_checked": n, "max_absdiff_vs_activation_recorder": rd}
    return res


def rank_vs_top(arrays, tracked_ids, prefix: str = "") -> tuple[int, int, int]:
    """Where a tracked token is in the stored top-25, its rank must match its
    place there (min-rank rule; the top-25 values are fp16, so fp16 ties give a
    band). Returns (n overlaps, n exact, n within the fp16 tie band); tracked
    tokens outside the top-25 must have rank > number of strictly-higher entries.
    prefix "x_": the per-run content tokens (x_rank, x_lp, ...) instead."""
    n = exact = band = 0
    R, LP = arrays[prefix + "rank"], arrays[prefix + "lp"]
    sets = [(arrays["top_ids"][si, li], arrays["top_lp"][si, li], R[si, li], LP[si, li])
            for si in range(arrays["top_ids"].shape[0]) for li in range(arrays["top_ids"].shape[1])]
    sets.append((arrays["model_top_ids"], arrays["model_top_lp"], arrays[prefix + "model_rank"],
                 arrays[prefix + "model_lp"]))
    lo = int(arrays["positions_lo"])
    for top_ids, top_lp, rank, lp in sets:
        for a, pos in enumerate(arrays["top_pos"]):
            if pos < 0:
                continue
            t = pos - lo
            for k, tid in enumerate(tracked_ids):
                hit = np.where(top_ids[a] == tid)[0]
                r = int(rank[t, k])
                if len(hit):
                    v = top_lp[a, hit[0]]
                    if v != lp[t, k]:
                        raise ValueError(f"tracked log-prob {lp[t, k]} != top-25 value {v} at position {pos}")
                    lo_r = 1 + int((top_lp[a] > v).sum())
                    hi_r = int((top_lp[a] >= v).sum())
                    n += 1
                    exact += r == lo_r
                    band += lo_r <= r <= hi_r
                elif r <= int((top_lp[a] > lp[t, k]).sum()):
                    raise ValueError(f"tracked token {tid} rank {r} but not in the top-25 at position {pos}")
    return n, exact, band


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="path to a runs.jsonl")
    parser.add_argument("--out", required=True, help="output directory (under data/)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N runs")
    parser.add_argument("--validate", type=int, default=0,
                        help="compare the first N new runs with the single-run readout (jmr.lens.readout)")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--content-tokens", default=None,
                        help="per-item content-word tracked tokens (e.g. configs/content_tokens.json); stored in "
                             "separate x_* arrays, default arrays unchanged; runs on items not in the file stop the run")
    parser.add_argument("--concept-tokens", default=None,
                        help="global concept tokens tracked in every run (e.g. configs/concept_tokens_grader.json); "
                             "appended to the x_* arrays with x_opt = -1")
    parser.add_argument("--cot-resid", action="store_true",
                        help="also store block outputs at the CoT anchors for the workspace layers (fp16)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = read_runs_lenient(Path(args.runs))
    if args.limit:
        runs = runs[: args.limit]
    index_path = out / "index.jsonl"
    done = set()
    if index_path.exists():
        for line in open(index_path):
            m = json.loads(line)
            if (out / m["file"]).exists():
                done.add(m["run_id"])
    todo = [(r, e) for r, e in runs if r.run_id not in done]
    if len({r.run_id for r, _ in runs}) != len(runs):
        raise ValueError("duplicate run_id in the runs file")
    print(f"{len(runs)} runs, {len(done)} already done, {len(todo)} to do → {out}", flush=True)
    config.snapshot(out)

    model_cfg = config.load("model")
    hf_model, tokenizer, lens_model = load_model()
    lens = load_lens(lens_file())
    reader = BatchReader(lens_model, lens, tokenizer, cot_resid=args.cot_resid, content_tokens=args.content_tokens,
                         concept_tokens=args.concept_tokens)
    fp = fingerprint(tokenizer)
    print(f"tracked tokens: {list(zip(reader.tracked.labels, reader.tracked.ids))}", flush=True)

    pool = ThreadPoolExecutor(max_workers=2)
    pending = []
    diffs, shifted, n_tok, sizes, peaks, times, validations = [], [], 0, [], [], [], []
    overlap, x_overlap = [0, 0, 0], [0, 0, 0]
    t_start = time.time()
    with open(index_path, "a") as index:
        for i, (record, extras) in enumerate(todo):
            # Rule 1 checks before reading.
            if record.tokenizer_fingerprint != fp:
                raise ValueError(f"{record.run_id}: tokenizer fingerprint {record.tokenizer_fingerprint} != {fp}")
            if record.model.get("revision") != model_cfg["revision"]:
                raise ValueError(f"{record.run_id}: model revision {record.model.get('revision')} != pinned")
            if len(record.completion_logprobs) != len(record.completion_token_ids):
                raise ValueError(f"{record.run_id}: {len(record.completion_logprobs)} log-probs for "
                                 f"{len(record.completion_token_ids)} completion tokens")
            bad_letters = sorted(set(record.letters) - set(reader.tracked.letters))
            if bad_letters:
                raise ValueError(f"{record.run_id}: option letters {bad_letters} are not tracked")
            plan = reader.plan(record)
            problems = check_plan(plan, tokenizer)
            if problems:
                raise ValueError(f"{record.run_id}: anchor problems {problems}")

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            arrays = reader.read(record, plan)
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30

            tf = arrays["completion_tf_logprob"].astype(np.float64)
            vl = np.asarray(record.completion_logprobs, dtype=np.float64)
            diffs.append(np.abs(tf - vl))
            shifted.append(np.abs(tf[1:] - vl[:-1]))
            n_tok += len(plan.ids)
            ov = rank_vs_top(arrays, reader.tracked.ids)
            overlap = [a + b for a, b in zip(overlap, ov)]
            if "x_ids" in arrays:
                ov = rank_vs_top(arrays, arrays["x_ids"].tolist(), prefix="x_")
                x_overlap = [a + b for a, b in zip(x_overlap, ov)]

            meta = reader.meta(record, extras, plan, arrays)
            meta.update({"file": file_name(record.run_id), "seconds": round(dt, 3), "peak_gpu_gib": round(peak, 2),
                         "seq_len": len(plan.ids)})
            if i < args.validate:
                v = validate_against_single(reader, lens_model, lens, record, plan, arrays)
                validations.append({"run_id": record.run_id, **v})
                print(json.dumps(validations[-1]), flush=True)
            arrays["meta_json"] = np.array(json.dumps(meta))
            pending.append((pool.submit(save_npz, out / meta["file"], arrays, not args.no_compress), meta))
            # Write index lines in order, once their file is on disk.
            while pending and pending[0][0].done():
                fut, m = pending.pop(0)
                sizes.append(fut.result())
                index.write(json.dumps(m) + "\n")
                index.flush()
            while len(pending) > 4:
                fut, m = pending.pop(0)
                sizes.append(fut.result())
                index.write(json.dumps(m) + "\n")
                index.flush()
            peaks.append(peak)
            times.append((len(plan.ids), dt))
            if i % 20 == 0 or i == len(todo) - 1:
                el = time.time() - t_start
                print(f"[{i + 1}/{len(todo)}] len {len(plan.ids)} {dt:.2f}s peak {peak:.1f} GiB | "
                      f"{(i + 1) / el * 60:.1f} runs/min, {n_tok / el:.0f} tok/s", flush=True)
        for fut, m in pending:
            sizes.append(fut.result())
            index.write(json.dumps(m) + "\n")
        index.flush()
    pool.shutdown()

    elapsed = time.time() - t_start
    summary = {"runs_file": args.runs, "n_runs_total": len(runs), "n_new": len(todo), "elapsed_s": round(elapsed, 1)}
    if todo:
        d, s = np.concatenate(diffs), np.concatenate(shifted)
        median, p99 = float(np.median(d)), float(np.quantile(d, 0.99))
        summary.update({
            "runs_per_min": round(len(todo) / elapsed * 60, 2),
            "tokens_per_s": round(n_tok / elapsed, 1),
            "peak_gpu_gib_max": round(max(peaks), 2),
            "mean_seq_len": round(n_tok / len(todo), 1),
            "disk_bytes_per_run_mean": int(np.mean(sizes)),
            "disk_bytes_total_new": int(np.sum(sizes)),
            "vllm_logprob_absdiff": {"n_tokens": int(d.size), "median": median, "p99": p99, "max": float(d.max()),
                                     "shifted_median": float(np.median(s))},
            "m0_4_pass": {"median": median <= MEDIAN_MAX, "p99": p99 <= P99_MAX,
                          "shift": float(np.median(s)) >= SHIFT_FACTOR * median},
            "rank_vs_top25": {"overlaps": overlap[0], "exact": overlap[1], "within_fp16_tie_band": overlap[2]},
            "content_rank_vs_top25": {"overlaps": x_overlap[0], "exact": x_overlap[1],
                                      "within_fp16_tie_band": x_overlap[2]} if (args.content_tokens or args.concept_tokens) else None,
            "seconds_by_length": [[int(n), round(t, 3)] for n, t in times],
            "validation_vs_single_run_readout": validations,
        })
    path = out / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "seconds_by_length"}, indent=1))
    print(f"→ {path}")
    if todo and not all(summary["m0_4_pass"].values()):
        raise SystemExit("FAIL: teacher-forced log-probs disagree with vLLM beyond the M0.4 tolerance")


if __name__ == "__main__":
    main()
