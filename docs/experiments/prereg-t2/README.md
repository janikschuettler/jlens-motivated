---
id: prereg-t2
title: 'Pre-registered test: complied vs persuaded under the same hint (T2)'
strand: C
also: []
status: done
kind: pre-registered
verdict: against
confidence: high
dates:
  start: 2026-11-03T02:50
  end: 2026-11-03T03:10
  approx: true
timing:
  wall: null
  gpu_hours: null
  api_usd: 0
  note: Not timed.
machine: arena9-mayor (RTX 4090)
n:
  items: 36 vs 18
  runs: '301'
numbers:
- label: 'J-lens: complied vs persuaded (AUROC of S)'
  value: 0.46 [0.32, 0.61]
- label: Logit lens
  value: 0.43 [0.25, 0.62]
- label: Runs and questions compared
  value: 179 vs 122 runs; 36 vs 18 items
- label: 'Exploratory: the main readout''s layer 30–32 score'
  value: 0.62 [0.47, 0.76]
- label: 'Exploratory: probe on mid-reasoning hidden states (layer 20) / reasoning length alone'
  value: 0.70 [0.54, 0.85] / 0.72 [0.60, 0.82]
numbers_note: 'AUROC: 0.5 is chance, 1 is perfect. The length row is a baseline: a readout is only useful if it beats it.'
reports:
- ../prereg-t1/report.md
- ../../preregistration.md
figure: ../prereg-t1/fig07_auroc.svg
data:
- ../prereg-t1/data/lens_prereg.json
commits:
- 700e955
related:
- reelicitation-e7
- replicate-leads
---

# Pre-registered test: complied vs persuaded under the same hint (T2)

## Question

Among runs that end on B under the same hint, do those that later give the correct answer when asked sincerely ('complied') keep more of the correct letter than those that repeat B ('persuaded')?

## Method

Thinking mode, the wording-dissection runs on known questions, pooled over the four 'the answer is X' / 'indicates the answer is X' wordings. Groups from the sincere follow-up question (E7). Both groups saw the same prompt, so prompt differences can't explain a result. Pass rule: J-lens AUROC lower bound above 0.5, with at least 15 questions per group.

## Result

No: failed. J-lens AUROC 0.46 [0.32, 0.61], no better than chance.

## Interpretation

The one lead with enough data for a fair test failed. Given the readout's limits, this says little about whether the knowledge is present. But the J-lens score does not detect 'knows the answer, says B' better than reasoning length does.

## Caveats

- S inherits the letter readout's limits.

## Files

- Report: [../prereg-t1/report.md](../prereg-t1/report.md)
- Report: [../../preregistration.md](../../preregistration.md)
- Figure: [../prereg-t1/fig07_auroc.svg](../prereg-t1/fig07_auroc.svg)
- Derived numbers: [../prereg-t1/data/lens_prereg.json](../prereg-t1/data/lens_prereg.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_prereg.py`
- `scripts/lens/analyze_t1b.py`
- `configs/explore/t1b_*.yaml`
