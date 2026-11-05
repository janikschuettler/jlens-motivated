---
id: jlens-option-tokens
title: J-lens at the prompt's option tokens
strand: C
also: []
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: '2026-11-04'
  end: '2026-11-04'
  approx: true
timing:
  wall: null
  gpu_hours: null
  api_usd: null
  note: null
machine: arena9-octavia (CPU)
n:
  items: null
  runs: null
numbers: []
numbers_note: null
reports:
- ../jlens-vs-competitors/report.md
- ../instrument-q1q3/report.md
figure: null
data: []
commits:
- d30ea1a
related:
- instrument-q1q3
- jlens-vs-competitors
---

# J-lens at the prompt's option tokens

## Question

Does the J-lens put into words, at the prompt's option tokens, what the option-text probe reads there?

## Method

Answered inside the J-lens-versus-competitors comparison: the J-lens, the logit lens and a trained probe were all applied to the same stored hidden states at the prompt's option words (no new readout was needed).

## Result

No. At the option words, a trained probe ranks the correct option first in 72% of no-hint runs. The J-lens does so in 42% and the logit lens in 43% (25% would be chance). Telling questions the model knows from honest errors: probe AUROC 0.80, J-lens 0.61, logit lens 0.53 (0.5 is chance, 1 is perfect).

## Interpretation

The answer knowledge is there in the model's hidden states at the option words, but the J-lens reads only part of it, no more than the plain logit lens. A trained probe sees it much better.

## Files

- Report: [../jlens-vs-competitors/report.md](../jlens-vs-competitors/report.md)
- Report: [../instrument-q1q3/report.md](../instrument-q1q3/report.md)
