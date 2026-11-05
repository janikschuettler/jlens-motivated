---
id: lens-pipeline
title: Batch lens readout for thousands of saved runs
strand: T
also:
- C
status: done
kind: tooling
verdict: supports
confidence: high
dates:
  start: 2026-11-02T21:51
  end: 2026-11-02T22:35
  approx: false
timing:
  wall: about 45 min
  gpu_hours: null
  api_usd: 0
  note: GPU time not recorded.
machine: arena9-mayor (RTX 4090)
n:
  items: null
  runs: 1,704 runs in the validation sets
numbers:
- label: Batch vs single-run readout
  value: bit-identical on 15 runs
- label: Speed on the RTX 4090
  value: 85 runs/min (thinking, mean 1,597 tokens); about 200/min (non-thinking)
- label: Disk per run
  value: 2.4 MB per thinking run, 1.0 MB per non-thinking run
- label: Peak GPU memory
  value: 17.5 GiB on the longest run (8,348 tokens)
numbers_note: null
reports:
- report.md
figure: null
data: []
commits:
- f2781bf
related:
- m0-tools
- lens-step0
---

# Batch lens readout for thousands of saved runs

## Question

Can we read out the J-lens and logit lens for thousands of saved runs, with exactly the result of the checked single-run readout?

## Method

scripts/lens/readout_runs.py re-runs the model on each run's saved token IDs (teacher forcing). Per run it stores: the log-probability and rank of each option letter at every position; the top-25 tokens at key positions; and the hidden states at three positions. It never stores full-vocabulary outputs (rule 3). Tested on three run sets: temp_check (1,500 thinking-mode runs), m1_smoke (84) and 120 runs with thinking off.

## Result

Yes. On 15 runs the batch readout is bit-identical to the single-run readout, and every run set passes the check against vLLM's log-probabilities (M0.4).

## Interpretation

Every later readout uses this pipeline and repeats its checks. For example, in the main readout sets the median difference from vLLM was at most 0.0007 nats (median |Δ| ≤ 0.0007) and the 99th percentile at most 0.14 nats.

## Caveats

- Speed on the L40S was first estimated, not measured; later sets measured 35–220 runs/min depending on run length.

## Files

- Report: [report.md](report.md)

Code and configs named in the text above:

- `scripts/lens/readout_runs.py`

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `src/jmr/lens/batch.py`
- `scripts/lens/readout_runs.py`
