---
id: lens-c3
title: 'Planted claims: the one internal trace of the correct answer'
strand: C
also: []
status: done
kind: exploratory
verdict: inconclusive
confidence: low
dates:
  start: 2026-11-03T10:20
  end: 2026-11-03T11:20
  approx: true
timing:
  wall: about 1 h
  gpu_hours: 0
  api_usd: 0
  note: CPU only, on existing readouts; the C3 readout itself ran on octavia at 02:50 (not timed).
machine: arena9-octavia (CPU)
n:
  items: 21 (kept B)
  runs: 200 (61 kept B)
numbers:
- label: S, runs that kept B (J-lens / logit lens)
  value: +1.4 [+0.8, +2.1] / +1.7
- label: 'Salience control: the planted letter in runs that rejected it (J-lens / logit lens)'
  value: −0.2 [−0.6, +0.3] / +0.5
- label: Runs that kept B with no outside source and no reconsidering (53 runs)
  value: same S (+1.4)
- label: 'Predicting recovery from the first 50 tokens (AUROC): J-lens / forced early answer / reasoning length'
  value: 0.46 [0.31, 0.61] / 0.70 [0.52, 0.85] / 0.73 [0.61, 0.84]
- label: 'Lead: kept-B runs that give the correct answer when asked vs not (AUROC, J-lens / logit lens)'
  value: 0.86 [0.77, 0.96] / 0.77 [0.61, 0.91]
numbers_note: 'S and the salience control are margins in nats at the answer position, averaged over layers 14–32: how far a letter sits above the other unchosen wrong letters. AUROC: 0.5 is chance, 1 is perfect.'
reports:
- report.md
figure: fig08_c3.svg
data:
- data/lens_c3.json
commits:
- bd40825
related:
- planted-claim-c3
- replicate-leads
---

# Planted claims: the one internal trace of the correct answer

## Question

When the model keeps a planted wrong claim, is the correct answer still visible inside?

## Method

Existing readouts of all 200 planted-claim runs (C3): score S at the answer position, plus readouts along the reasoning and in the first 50 generated tokens. Groups, layers and positions were not fixed in advance.

## Result

Yes, at the answer position, and more than the salience control explains. Runs that keep the planted B hold the correct letter at S = +1.4 [+0.8, +2.1] nats, against −0.2 for the salience control. The logit lens shows the same (+1.7 vs +0.5).

## Interpretation

The only internal trace of the correct answer beyond the salience control. The logit lens sees it too, so it sits in the model's output probabilities at the answer position, not somewhere only the J-lens reveals. It rests on 21 questions.

## Caveats

- 21 questions; several scores were tried.
- The salience control isn't symmetric.
- Runs that kept B have much shorter reasoning than other groups.
- The follow-up-question lead may be an effect of the questions, not of the runs.

## Files

- Report: [report.md](report.md)
- Figure: [fig08_c3.svg](fig08_c3.svg)
- Derived numbers: [data/lens_c3.json](data/lens_c3.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_c3.py`
