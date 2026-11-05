---
id: prereg-t1
title: 'Pre-registered test: do silent switches keep more of the correct letter? (thinking off, T1)'
strand: C
also: []
status: done
kind: pre-registered
verdict: inconclusive
confidence: low
dates:
  start: 2026-11-03T02:50
  end: 2026-11-03T03:10
  approx: true
timing:
  wall: null
  gpu_hours: null
  api_usd: 0
  note: Readout counted under the thinking-off silent-switch entry.
machine: arena9-mayor (RTX 4090)
n:
  items: 149 known (107 after exclusion)
  runs: 438 switches
numbers:
- label: 'J-lens: silent vs open switches (AUROC of S)'
  value: 0.74 [0.60, 0.86]
- label: Logit lens
  value: 0.59 [0.37, 0.79]
- label: Runs compared
  value: 9 silent runs on 7 items vs 429 open
- label: Silent switches that fell on the left-out questions
  value: 115 of 124
- label: 'Exploratory: new samples of all questions, J-lens'
  value: J-lens 0.63 [0.52, 0.72]
numbers_note: 'AUROC: the chance that a silent switch scores higher than an open one. 0.5 is chance.'
reports:
- report.md
- ../../preregistration.md
figure: fig07_auroc.svg
data:
- data/lens_prereg.json
commits:
- 700e955
related:
- prereg-t1b
- nothink-scale
---

# Pre-registered test: do silent switches keep more of the correct letter? (thinking off, T1)

## Question

The main readout's lead: with thinking off, silent switches kept more of the correct letter in the J-lens than open switches. Does that hold on new questions, with the score fixed in advance?

## Method

Score S (the correct letter's margin at the answer position, averaged over layers 14–32), with the J-lens; the logit lens for comparison. Thinking off, M&B's setting, 149 known questions × 8 samples, leaving out the 42 questions that supplied the main readout's silent switches. Pass rule: the J-lens AUROC's lower bound above 0.5, and the logit lens's interval including 0.5. Underpowered if fewer than 15 questions have a silent switch.

## Result

Can't tell: too few silent switches were left (9 runs on 7 questions).

## Interpretation

Silent switching concentrates on a few questions, so leaving them out left too little data. The next test [T1b] repeats it on fresh questions.

## Caveats

- Silent and open are keyword labels.

## Files

- Report: [report.md](report.md)
- Report: [../../preregistration.md](../../preregistration.md)
- Figure: [fig07_auroc.svg](fig07_auroc.svg)
- Derived numbers: [data/lens_prereg.json](data/lens_prereg.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_prereg.py`
- `scripts/lens/analyze_t1b.py`
- `configs/explore/t1b_*.yaml`
