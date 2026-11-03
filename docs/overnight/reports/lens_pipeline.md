# Batch lens readout pipeline

Built and validated on arena9-mayor (RTX 4090, 24 GB) on 2026-11-02. Data: the existing MMLU-Redux run sets `temp_check` (1,500 thinking runs) and `m1_smoke` (84 thinking runs), plus 120 non-thinking runs generated for this test on mayor (`configs/explore/gap_step0.yaml --limit-items 8 --out-suffix _lenstest`, i.e. M&B format on 8 `mb_replication` items, test data only, at `data/explore/gap_factorial/step0_lenstest/` on mayor). Lens env (torch 2.11, transformers 5.11, jlens `581d3986`).

**Result:** it works. The batch readout gives exactly the same numbers as the M0.2-validated single-run readout, teacher-forced log-probs match vLLM's within the M0.4 tolerance, and the stored ranks agree with the stored top-25 everywhere they overlap. Throughput on the 4090: 85 runs/min for thinking runs (temp_check, mean 1,597 tokens) and about 200 runs/min for non-thinking runs (mean 539 tokens), with peak GPU memory 17.5 GiB on the longest run (8,348 tokens, no truncation). Disk: 2.4 MB per thinking run, 1.0 MB per non-thinking run.

## Usage

One command per runs.jsonl (resumable: rerun the same command after a crash and it skips finished runs):

```
ssh arena9-mayor 'cd /root/jlens-motivated && unset VIRTUAL_ENV && nohup uv run --project envs/lens python \
  scripts/lens/readout_runs.py --runs data/runs/<run_set>/runs.jsonl --out data/readouts/<run_set> \
  > data/logs/readout_<run_set>.log 2>&1 &'
```

Options: `--limit N` (first N runs only), `--validate N` (recompute the first N new runs with `jmr.lens.readout` and compare; costs about 5× a normal run), `--no-compress`.

It reads any runs.jsonl with the `RunRecord` fields, and keeps extra fields (such as `arm` and its spec) in `meta["extras"]`. Before reading a run it checks, and stops on failure (rule 1): the tokenizer fingerprint, the pinned model revision, one vLLM log-prob per completion token, option letters within A–D, and that the anchors decode to the intended tokens (as in M0.4). It feeds exactly `prompt_token_ids + completion_token_ids`; nothing is re-tokenized or truncated. At the end it fails if the vLLM agreement misses the M0.4 tolerance.

Load the results with `jmr.analysis.readouts.Readouts` (numpy only, either env):

```python
from jmr.analysis.readouts import Readouts
r = Readouts("data/readouts/temp_check", labels="data/runs/temp_check/labels.jsonl",
             items="data/runs/temp_check/items.jsonl")
idx = r.select(condition="known_evidence", label="resisted")
r.at_anchor("jlens", "correct", "pre_answer", form="any", runs=idx)       # [runs, 10 layers] log-prob
r.at_anchor("logit", "hinted", "think_end", what="rank", runs=idx)        # ranks
r.cot_bins("jlens", "correct", n_bins=20, what="logrank", runs=idx)       # [runs, layers, 20 bins] along the CoT
r.top_tokens(idx[0], "jlens", "pre_answer", layer=26)                     # stored top-25
r.resid(idx[0], "pre_answer")                                              # [36, 4096] fp16 block outputs
```

Lenses: `jlens`, `logit` (workspace layers 14, 16, …, 32) and `model` (the model's own distribution, block 35). Roles: `correct`, `hinted` (the run's hint, or for unhinted runs the item's hinted option from items.jsonl), `answer` (parsed final answer), `other_wrong_max`, `other_wrong_mean`, or a literal letter. Forms: `space`, `bare`, `paren`, or `any` (log-sum-exp of the three log-probs; minimum of the ranks). Missing letters or anchors give NaN.

## What is stored

Code: `src/jmr/lens/batch.py` (readout), `scripts/lens/readout_runs.py` (CLI and checks), `src/jmr/analysis/readouts.py` (loader), `tests/test_batch_plan.py` (anchor planning, thinking and non-thinking). Settings in `configs/lens.yaml` (`tracked_tokens`, `batch`).

The output directory holds one `<sha1(run_id)[:20]>.npz` per run, `index.jsonl` (one meta line per run: IDs, condition, hint, correct letter, parsed answer, anchors, CoT span, extras, per-run vLLM agreement, seconds, peak GPU memory), `summary_<time>.json` per invocation (validation numbers, throughput) and a snapshot of `configs/`.

**Tracked tokens** (12, all checked to be single tokens; the run refuses to start if the tokenizer disagrees with the IDs in `configs/lens.yaml`):

| Form | A | B | C | D | Where it occurs |
| --- | --- | --- | --- | --- | --- |
| `space` " X" | 362 | 425 | 356 | 422 | "Answer: X", the final-answer letter in 1,387 of 1,493 temp_check runs |
| `bare` "X" | 32 | 33 | 34 | 35 | `\boxed{X}`, `**X` (the other 106) |
| `paren` "(X" | 4346 | 5349 | 3025 | 5432 | how the options are written in the prompt |

Not single tokens: " (A", "A)", "**A", "{A". ":A" is a single token but never occurs at the answer position, so it is not tracked. Adding a form later means re-running the readout (rule 3).

**Per run (no full-vocabulary output is stored; the softmax is computed in memory in 512-position chunks):**

| Array | Shape, dtype | Content |
| --- | --- | --- |
| `lp`, `rank` | [2 lenses, 10 layers, T, 12], fp16 / int32 | tracked-token log-prob and rank, J-lens and logit lens, at T = the last 64 prompt positions + every completion position |
| `model_lp`, `model_rank` | [T, 12] | the same for the model's own final-layer distribution |
| `top_ids`, `top_lp` | [2, 10, A, 25] | top-25 at A = 21 anchors |
| `model_top_ids`, `model_top_lp` | [A, 25] | model's top-25 at the anchors |
| `completion_tf_logprob` | [n_completion], fp32 | teacher-forced log-prob of every sampled token |
| `resid` | [36 blocks, 3, 4096], fp16 | block outputs at `last_prompt`, `think_end`, `pre_answer` (NaN if the anchor is absent: `think_end` is absent in non-thinking runs, where the empty think block is in the prompt, and in truncated runs; `pre_answer` is absent when no answer parses) |

Position t holds the distribution of the token at t + 1. **Anchors:** `last_prompt` (P − 1), `think_start` (`<think>`), `think_end` (`</think>`), `pre_first_stated` and `pre_answer` (the position whose next token is the first-stated or the final answer letter), and `cot_00`…`cot_15` (evenly spaced over the reasoning span, inside `<think>…</think>`; for a non-thinking run, the completion up to the answer). **Rank tie rule:** rank = 1 + the number of tokens with a strictly higher log-prob (tied tokens share the best rank). Block outputs are bf16 in the model and stored as fp16, which is exact in fp16's range; the run stops if any value overflows fp16.

## Validation

Each check below ran inside `readout_runs.py` (`--validate N` for the first two).

**Batch readout vs the M0.2-validated single-run readout (`jmr.lens.readout`)**, recomputed from scratch on 15 runs (5 temp_check, 5 m1_smoke, 5 non-thinking):

| Check | Result |
| --- | --- |
| Tracked-token log-probs, J-lens and logit lens, every stored position × layer | max abs diff 0.0 (100% bit-identical) |
| Top-25 at every anchor × layer (equal up to ties at the 25th value) | all equal (210/210, 200/200 or 180/180 per run and lens) |
| Ranks: compiled kernel vs a plain per-token masked count | 0 mismatches |
| Model's own tracked log-probs vs logit lens at block 35 | max abs diff 0.0 |
| Stored teacher-forced log-prob of each sampled token vs `next_token_logprobs` | 0.0 on 14 runs, 0.002 nats on 1 (the last chunk has a different shape, so bf16 rounding differs) |
| Stored anchor block outputs (own hooks, all 36 blocks) vs jlens's `ActivationRecorder` at the 11 captured layers | max abs diff 0.0 (10 runs: 5 m1_smoke, 5 non-thinking) |

**Teacher-forced log-probs vs vLLM's**, all sampled tokens (M0.4 tolerance: median ≤ 0.05, p99 ≤ 0.5, median at least 10× worse when shifted by one position):

| Run set | Tokens | Median | p99 | Max | Shifted median | M0.4 |
| --- | --- | --- | --- | --- | --- | --- |
| temp_check (thinking) | 2,238,620 | 0.00007 | 0.163 | 1.22 | 0.101 | pass |
| m1_smoke (thinking) | 161,951 | 0.00046 | 0.143 | 1.17 | 0.171 | pass |
| step0_lenstest (non-thinking) | 45,798 | 0.00008 | 0.242 | 0.85 | 0.192 | pass |

**Tracked ranks vs the stored top-25** wherever a tracked token is in the top-25 (same log-prob, and the rank equals its place in the top-25 under the min-rank rule): 102,596 / 102,596 (temp_check), 5,338 / 5,338 (m1_smoke) and 8,310 / 8,310 (non-thinking) exact. Tracked tokens outside the top-25 are also checked to have a rank no better than the 25th entry.

**Anchors** decode to the intended tokens on every run (the script stops otherwise): 1,500 + 84 + 120 runs, none stopped. CPU tests: `tests/test_batch_plan.py` 3 passed (lens env on mayor).

**Sanity:** with `form="any"`, the model's own log-prob of the correct letter at `pre_answer` is −0.03 (unhinted) and −0.09 (known_evidence) nats on temp_check. With `form="space"` it is −1.6, because about 7% of runs write the final letter bare (`\boxed{X}`, `**X`), where " X" is not the next token. **Use `form="any"` (or the run's own form) at `pre_answer`.**

## Throughput and memory (RTX 4090)

Batch size 1, one teacher-forced pass per run, then 2 lenses × 10 layers + the final layer unembedded in 512-position chunks. Read time grows linearly with sequence length, about 0.42 ms per token (≈2,400 tokens/s), plus ~0.05 s per run for checks and saving (in a background thread).

| Run set | Mode | Runs | Mean tokens | Runs/min (wall; model load excluded, validations included) | Peak GPU |
| --- | --- | --- | --- | --- | --- |
| temp_check | thinking | 1,500 | 1,597 | 85 | 17.4 GiB |
| m1_smoke | thinking | 84 | 2,065 | 58 (5 validated runs included) | 17.5 GiB |
| step0_lenstest | non-thinking | 120 | 539 | 199 (3 validated runs included) | 16.8 GiB |
| step0_mayor | non-thinking | 480 | 601 | 202 (2 validated runs included) | 17.0 GiB |

Read time by length on temp_check (GPU time per run): < 700 tokens 0.24 s, 700–1,500 0.42 s, 1,500–3,000 0.92 s, > 3,000 1.56 s; the longest run anywhere (8,348 tokens, m1_smoke) took 3.5 s at 17.5 GiB. So thinking runs of 1–3k tokens give roughly 45–130 runs/min, non-thinking runs of ~500 tokens roughly 200/min. Model load and `torch.compile` warm-up add about 1 minute per invocation. Memory is the 16.4 GB of weights plus ~1 GB, so nothing in our run sets (max 8,192 new tokens) comes near the 24 GB limit.

First real use: [lens_step0.md](lens_step0.md).

**L40S estimate (not measured):** per run about the same as the 4090 to ~1.3× faster (same Ada generation; more tensor throughput with fp32 accumulation, a bit less memory bandwidth). Since one process needs ≤ 17.5 GiB, two processes on disjoint halves of a runs file fit in the L40S's 45 GiB, for roughly 2× aggregate. That is about 10,000 thinking runs of ~1.6k tokens in ~1–2 hours.

Loader speed: `Readouts.at_anchor` decompresses each run's arrays on a cache miss (LRU of 256 runs); the demo below (8 anchor queries over 900 runs) took 90 s on mayor's CPU.

## Disk per run

Compressed npz (`np.savez_compressed`):

| Run set | Mean tokens | Disk per run | Total |
| --- | --- | --- | --- |
| temp_check (thinking) | 1,597 | 2.43 MB | 3.6 GB for 1,500 runs |
| m1_smoke (thinking) | 2,065 | 2.90 MB | 0.24 GB for 84 |
| step0_lenstest (non-thinking) | 539 | 0.98 MB | 0.12 GB for 120 |

Rule of thumb: 0.75 MB fixed (the anchor block outputs are 0.68 MB of it) + 1.1 KB per stored position (completion length + 64), of which the int32 ranks are about two thirds. 10,000 thinking runs ≈ 25 GB. That matters on octavia (100 GB disk). If it gets tight, the options are storing ranks as uint32 → log-rank fp16 or dropping the prompt tail; neither is done (both change the format, the user's call).

**Where the data is:** the readouts exist only on mayor (`data/readouts/{temp_check,m1_smoke,step0_lenstest,step0_mayor}`, 4.5 GB). Mayor is available tonight only. Regenerating temp_check takes 18 min, so moving it isn't necessary.

## Demo: correct vs hinted letter at the pre-answer position (temp_check)

Command (numpy only, CPU):

```
uv run --project envs/lens python scripts/lens/readout_demo.py --readouts data/readouts/temp_check \
  --run-set data/runs/temp_check --conditions unhinted known_evidence --form any
```

Mean log-prob (nats) of the correct letter and the hinted letter (for unhinted runs, the item's hinted option), `form="any"`, at `pre_answer` (the position whose next token is the final-answer letter). Both temperatures (0.6 and 1.0) pooled; runs without a parsed answer drop out (n).

| Layer | unhinted J-lens correct | unhinted J-lens hinted | unhinted logit correct | unhinted logit hinted | known_evidence J-lens correct | known_evidence J-lens hinted | known_evidence logit correct | known_evidence logit hinted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n | 446 | 446 | 446 | 446 | 449 | 449 | 449 | 449 |
| 14 | −16.80 | −17.04 | −8.21 | −8.10 | −17.20 | −17.43 | −8.54 | −8.36 |
| 16 | −24.70 | −25.00 | −10.48 | −10.02 | −25.52 | −25.84 | −11.19 | −10.67 |
| 18 | −25.46 | −25.77 | −9.26 | −9.43 | −25.94 | −26.21 | −9.91 | −9.89 |
| 20 | −27.97 | −28.72 | −6.57 | −6.77 | −28.17 | −28.92 | −7.03 | −7.28 |
| 22 | −19.75 | −20.23 | −6.84 | −7.74 | −21.02 | −21.54 | −7.17 | −8.00 |
| 24 | −14.97 | −22.15 | −1.28 | −9.19 | −14.95 | −23.25 | −1.27 | −9.61 |
| 26 | −19.52 | −25.32 | −1.17 | −8.03 | −20.24 | −26.71 | −1.16 | −8.06 |
| 28 | −12.10 | −19.07 | −1.13 | −6.47 | −11.91 | −19.37 | −1.21 | −6.45 |
| 30 | −4.61 | −21.80 | −0.14 | −14.29 | −3.71 | −20.61 | −0.19 | −13.21 |
| 32 | −1.08 | −26.35 | −0.52 | −19.26 | −1.21 | −24.76 | −0.58 | −18.43 |
| model (35) | −0.03 | −28.66 | | | −0.09 | −23.56 | | |

No claims. This is a pipeline check: in temp_check almost every known_evidence run resisted (447 resisted, 1 acknowledged switch, 0 silent; labels.jsonl), so the table compares two sets of runs that both answered correctly. The followed-vs-resisted contrast needs a run set with followed runs (from the gap factorial or wave 2). The J-lens log-probs of letter tokens are much lower than the logit lens's at layers 14–28. That is a property of the readout to keep in mind when reading log-probs, and a reason to also look at ranks (`what="rank"`) and at the correct-vs-other-wrong difference.

## Known limitations

- `think_end` (and its block outputs) is absent in non-thinking runs and in truncated runs. For probes at "end of reasoning" in non-thinking runs, use `pre_answer`. A separate `reasoning_end` anchor would be a format change; not added.
- `summary_*.json` throughput includes the `--validate` runs (each costs ~5× a normal run) and the compile warm-up of the first run.
- One process reads one sequence at a time. Batching several short sequences would speed up non-thinking readouts, but would change the matmul shapes and so the bit-exact match with the single-run readout; not done.
