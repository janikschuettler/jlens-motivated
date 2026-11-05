---
id: prereg-t1b
title: 'Pre-registered test on 1,744 fresh questions: silent vs open switches (T1b)'
strand: C
also: []
status: done
kind: pre-registered
verdict: inconclusive
confidence: high
dates:
  start: 2026-11-03T03:05
  end: 2026-11-03T04:34
  approx: false
timing:
  wall: about 1.5 h
  gpu_hours: null
  api_usd: 0
  note: octavia about 70 GPU-min (screen 26, hinted 20, readout 24); mayor screen about 34 min and readout about 21 min; mayor's hinted arm not timed.
machine: arena9-octavia (L40S) and arena9-mayor (RTX 4090)
n:
  items: 2,300 screened, 1,744 known
  runs: about 24,000 screen + 16,400 hinted (both machines)
numbers:
- label: 'J-lens: silent vs open switches (AUROC of S)'
  value: 0.54 [0.504, 0.577]
- label: Logit lens
  value: 0.46 [0.424, 0.500]
- label: Runs and questions compared
  value: 986 vs 7,268 switches; 333 vs 1,492 items
- label: 'Exploratory: response length alone'
  value: 0.54 [0.506, 0.572]
- label: 'Exploratory: J-lens within short / medium / long responses'
  value: 0.53, 0.51, 0.57 (only the longest excludes 0.5)
numbers_note: 'AUROC: 0.5 is chance, 1 is perfect. The pass rule needed the J-lens interval to stay above 0.5; its lower bound, 0.504, just does.'
reports:
- ../prereg-t1/report.md
- ../../preregistration.md
figure: ../prereg-t1/fig07_auroc.svg
data:
- ../prereg-t1/data/lens_prereg.json
commits:
- 700e955
related:
- prereg-t1
- replicate-leads
---

# Pre-registered test on 1,744 fresh questions: silent vs open switches (T1b)

## Question

The same test with enough data: with thinking off, do silent switches keep more of the correct letter in the J-lens than open switches?

## Method

Written after T1 came back underpowered, before any T1b data existed. A fresh draw (item seed 3) of 2,300 Redux questions, screened with 10 unhinted samples in M&B's thinking-off setting; 1,744 known. M&B's hint × 8 samples. Same score, groups and pass rule as T1. The screen ran on two machines, with a rule for their overlap fixed in advance.

## Result

Passed by its rule, but only just, and length explains as much. The J-lens gives AUROC 0.54 [0.504, 0.577]; response length alone also gives 0.54.

## Interpretation

The effect is tiny: the main readout's 0.65 shrank to 0.54, and length explains about as much. Any effect of S is small.

## Caveats

- S is a letter margin at the answer position, with the limits found in the readout-position entry.
- Silent and open are keyword labels.

## Files

- Report: [../prereg-t1/report.md](../prereg-t1/report.md)
- Report: [../../preregistration.md](../../preregistration.md)
- Figure: [../prereg-t1/fig07_auroc.svg](../prereg-t1/fig07_auroc.svg)
- Derived numbers: [../prereg-t1/data/lens_prereg.json](../prereg-t1/data/lens_prereg.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_prereg.py`
- `scripts/lens/analyze_t1b.py`
- `configs/explore/t1b_*.yaml`
