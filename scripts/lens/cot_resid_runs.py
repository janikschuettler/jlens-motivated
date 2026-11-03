"""Block outputs only, for probe training data (lens env, GPU).

For each run: one teacher-forced pass over exactly the saved token IDs (the
same forward and hooks as jmr.lens.batch.BatchReader), storing
  cot_resid [10 workspace layers, 16 CoT anchors, d] fp16, cot_resid_pos [16],
  resid [36 blocks, 3 anchors (last_prompt, think_end, pre_answer), d] fp16,
  resid_pos [3]
plus a meta line in index.jsonl (same fields as the batch readout's meta).
No lens or vocabulary output is computed or stored. The same rule-1 checks as
readout_runs.py run before each run (tokenizer fingerprint, revision, anchors).
--check-against <readout dir>: compares the stored arrays with a full batch
readout of the same runs (must be bit-identical).

--options-only: instead, store only the block outputs over each option's text in
the prompt, for an option-matching probe: opt_resid [4 layers (OPT_LAYERS), 4
options, d] fp16 = the mean over the option's text tokens (after "(X" and ")"
up to the token before the next "(Y", or before the first newline after the
last option), opt_span [4, 2]. The spans are found from the "(X" token IDs in
configs/lens.yaml and checked: the decoded span must equal the item's option
text (whitespace-stripped); the run stops otherwise (rule 1).

uv run --project envs/lens python scripts/lens/cot_resid_runs.py --runs <runs.jsonl> --out <dir>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jmr import config
from jmr.lens.batch import BatchReader, check_plan, read_runs_lenient
from jmr.lens.load import lens_file, load_lens, load_model
from jmr.tokens import fingerprint

from readout_runs import file_name, save_npz  # noqa: E402  (same folder)


# Layers for --options-only (a subset of the workspace layers, to keep disk use low).
OPT_LAYERS = [16, 20, 24, 28]


def option_spans(record, reader, tokenizer, item) -> list[tuple[int, int]]:
    """[start, end) token spans of the option texts in the prompt (see module doc)."""
    ids = record.prompt_token_ids
    tr = reader.tracked
    paren = [tr.ids[tr.index(L, "paren")] for L in record.letters]
    starts, k = [], 0
    for L, pid in zip(record.letters, paren):
        while k < len(ids) - 1 and not (ids[k] == pid and tokenizer.decode([ids[k + 1]]).startswith(")")):
            k += 1
        if k >= len(ids) - 1:
            raise ValueError(f"{record.run_id}: option ({L}) not found in the prompt")
        starts.append(k)
        k += 1
    spans = []
    for j, st in enumerate(starts):
        a = st + 2
        if j + 1 < len(starts):
            b = starts[j + 1]
        else:
            b = a
            while b < len(ids) and "\n" not in tokenizer.decode([ids[b]]):
                b += 1
        while b > a and not tokenizer.decode([ids[b - 1]]).strip():
            b -= 1  # drop trailing newline / space tokens
        spans.append((a, b))
    for j, ((a, b), opt, L) in enumerate(zip(spans, item["options"], record.letters)):
        text, want = tokenizer.decode(ids[a:b]).strip(), opt.strip()
        if j == len(spans) - 1 and text != want and want.startswith(text) and b < len(ids):
            # the last option's final punctuation can share a token with the newline (".\n\n");
            # accept only if that next token starts with exactly the missing text
            if tokenizer.decode([ids[b]]).startswith(want[len(text):]):
                text = want
        if b <= a or text != want:
            raise ValueError(f"{record.run_id}: option ({L}) span decodes to {text!r}, item says {opt!r}")
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-against", default=None)
    ap.add_argument("--options-only", action="store_true")
    ap.add_argument("--items", default=None, help="items.jsonl with option texts (needed for --options-only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = read_runs_lenient(Path(args.runs))
    index_path = out / "index.jsonl"
    done = set()
    if index_path.exists():
        done = {json.loads(l)["run_id"] for l in open(index_path)}
    todo = [(r, e) for r, e in runs if r.run_id not in done]
    print(f"{len(runs)} runs, {len(todo)} to do → {out}", flush=True)
    config.snapshot(out)
    mcfg = config.load("model")
    _, tokenizer, lens_model = load_model()
    reader = BatchReader(lens_model, load_lens(lens_file()), tokenizer, cot_resid=True)
    fp = fingerprint(tokenizer)
    items = {}
    if args.options_only:
        items = {d["id"]: d for d in map(json.loads, open(args.items))}
    t0 = time.time()
    with open(index_path, "a") as index:
        for i, (record, extras) in enumerate(todo):
            if record.tokenizer_fingerprint != fp:
                raise ValueError(f"{record.run_id}: tokenizer fingerprint mismatch")
            if record.model.get("revision") != mcfg["revision"]:
                raise ValueError(f"{record.run_id}: model revision mismatch")
            plan = reader.plan(record)
            problems = check_plan(plan, tokenizer)
            if problems:
                raise ValueError(f"{record.run_id}: anchor problems {problems}")
            if args.options_only:
                spans = option_spans(record, reader, tokenizer, items[record.question_id])
                with torch.no_grad():
                    full, _ = reader._forward(plan.ids, [0])
                    ores = torch.stack([torch.stack([full[l][a:b].float().mean(0) for a, b in spans])
                                        for l in OPT_LAYERS]).half()  # [layers, 4, d]
                if not torch.isfinite(ores).all():
                    raise ValueError(f"{record.run_id}: non-finite option block outputs")
                arrays = {"layers": np.array(OPT_LAYERS, dtype=np.int16), "opt_resid": ores.cpu().numpy(),
                          "opt_span": np.array(spans, dtype=np.int32)}
                meta = {"run_id": record.run_id, "question_id": record.question_id,
                        "correct_letter": record.correct_letter, "letters": record.letters,
                        "extras": extras, "file": file_name(record.run_id)}
                save_npz(out / meta["file"], arrays, True)
                index.write(json.dumps(meta) + "\n")
                index.flush()
                if i % 50 == 0 or i == len(todo) - 1:
                    print(f"[{i + 1}/{len(todo)}] {(i + 1) / (time.time() - t0) * 60:.1f} runs/min", flush=True)
                continue
            resid_pos = [plan.named[n] for n in reader.resid_anchors]
            with torch.no_grad():
                full, anchor = reader._forward(plan.ids, [p if p >= 0 else 0 for p in resid_pos])
                resid16 = anchor.half()
                cpos = [plan.named.get(f"cot_{k:02d}", -1) for k in range(reader.n_cot)]
                sel = torch.tensor([p if p >= 0 else 0 for p in cpos], device=reader.device)
                cres = torch.stack([full[l][sel] for l in reader.layers]).half()
            if not (torch.isfinite(resid16).all() and torch.isfinite(cres).all()):
                raise ValueError(f"{record.run_id}: block outputs overflow fp16")
            for j, p in enumerate(resid_pos):
                if p < 0:
                    resid16[:, j] = float("nan")
            for k, p in enumerate(cpos):
                if p < 0:
                    cres[:, k] = float("nan")
            arrays = {"layers": np.array(reader.layers, dtype=np.int16),
                      "resid_pos": np.array(resid_pos, dtype=np.int32), "resid": resid16.cpu().numpy(),
                      "cot_resid_pos": np.array(cpos, dtype=np.int32), "cot_resid": cres.cpu().numpy()}
            meta = {"run_id": record.run_id, "question_id": record.question_id, "condition": record.condition,
                    "hint_letter": record.hint_letter, "correct_letter": record.correct_letter,
                    "letters": record.letters, "answer": plan.answer, "named_positions": plan.named,
                    "cot_span": plan.cot_span, "resid_anchor_names": reader.resid_anchors,
                    "cot_resid_layers": reader.layers, "completion_len": len(record.completion_token_ids),
                    "model": record.model, "extras": extras, "file": file_name(record.run_id)}
            save_npz(out / meta["file"], arrays, True)
            index.write(json.dumps(meta) + "\n")
            index.flush()
            if i % 50 == 0 or i == len(todo) - 1:
                print(f"[{i + 1}/{len(todo)}] {(i + 1) / (time.time() - t0) * 60:.1f} runs/min", flush=True)
    if args.check_against:
        ref_dir = Path(args.check_against)
        ref = {json.loads(l)["run_id"]: json.loads(l) for l in open(ref_dir / "index.jsonl")}
        n = bad = 0
        for line in open(index_path):
            m = json.loads(line)
            if m["run_id"] not in ref:
                continue
            a, b = np.load(out / m["file"]), np.load(ref_dir / ref[m["run_id"]]["file"])
            for k in ("cot_resid", "resid", "cot_resid_pos", "resid_pos"):
                n += 1
                bad += not np.array_equal(a[k], b[k], equal_nan=True)
        print(f"check against {ref_dir}: {n - bad}/{n} arrays bit-identical", flush=True)
        if bad:
            raise SystemExit("FAIL: block outputs differ from the batch readout")


if __name__ == "__main__":
    main()
