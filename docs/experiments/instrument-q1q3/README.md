---
id: instrument-q1q3
title: Can any readout see the answer during the reasoning? (instrument check)
strand: C
also:
- T
status: done
kind: exploratory
verdict: instrument-limit
confidence: moderate
dates:
  start: 2026-11-03T15:55
  end: 2026-11-03T16:55
  approx: true
timing:
  wall: about 1 h
  gpu_hours: null
  api_usd: 0
  note: GPU readouts ended 16:35, start not recorded; CPU analysis about 7 min with 64 processes. About 10 GB of readouts.
machine: arena9-octavia (L40S)
n:
  items: up to 164 known + 14 honest-error
  runs: 2,779
numbers:
- label: 'Letter probe, mid-reasoning: no hint / told to argue for B'
  value: 0.30 [0.27, 0.33] / 0.26
- label: 'J-lens on option letters, mid-reasoning: same groups'
  value: 0.28 / 0.24
- label: Option content words, J-lens / logit lens, no hint
  value: 0.40 [0.34, 0.46] / 0.39
- label: 'Reasoning-text baseline (last option named): no hint / told to argue'
  value: 0.38 / 0.30
- label: 'Option-text probe: no hint / told to argue / honest errors'
  value: 72% [65, 78] / 62% [47, 77] / 26% [7, 48]
- label: J-lens vs logit lens
  value: no difference anywhere
- label: Content-word score on honest errors vs the same questions' correct runs (J-lens)
  value: '0.19 vs 0.18: an item effect, not specificity'
numbers_note: Each value is the share of runs in which the readout's top option is the correct one (0.30 = 30%). Chance is 0.25, with four options. A good instrument scores high where the model knows the answer and low on honest errors.
reports:
- report.md
figure: fig10_instrument.svg
data:
- data/instrument_q1q3.json
commits:
- 7e8a9bd
- 682f6f2
- afe77a8
related:
- lens-positions
- jlens-option-tokens
- option-probe-cot
---

# Can any readout see the answer during the reasoning? (instrument check)

## Question

Is there any readout that, during the reasoning, points to the correct option where the model knows it (sensitivity) and not where it doesn't (specificity)?

## Method

Thinking mode, 2,779 runs read out. Groups: runs with no hint that answer correctly on known questions; runs told to argue for B; the debate condition; honest errors, and the same questions' correct runs.
Readouts at 16 reasoning positions, at </think> and at the answer:
- option letters and option content words, with the J-lens and the logit lens;
- a probe that predicts the correct letter, and a probe that matches the reasoning state to each option, both trained on no-hint runs with questions held out;
- a probe on the prompt's option text;
- a baseline that reads the reasoning text (the option it named last).
The mid-reasoning window and the layer rule were fixed before looking; the option-text probe was added after.

## Result

During the reasoning, nothing beats the text. Readouts point to the correct option in 24–40% of runs, against 25% by chance and 38% for the text baseline. Only a probe on how the prompt represents each option's text finds the answer: 72% of no-hint runs.

## Interpretation

The model's knowledge of the answer can be read with a linear probe at the option tokens, on questions it knows. During the reasoning, none of our readouts recovers it beyond what the text says. So we still have no instrument that can tell, within one question, a run that knows the answer while arguing B from one that doesn't. The lens nulls stay uninformative about hidden knowledge.

## Caveats

- Only 14 honest-error questions (8 with content tokens).
- The option-text probe has one value per question, so it measures the question, not the run.
- Content tokens cover only 96 of 200 questions.
- The J-lens was not read at the option positions.
- Calibrated letter readouts are confounded in told-to-argue runs (the end of the prompt says 'argue that the answer is B') and are not used.

## Files

- Report: [report.md](report.md)
- Figure: [fig10_instrument.svg](fig10_instrument.svg)
- Derived numbers: [data/instrument_q1q3.json](data/instrument_q1q3.json)
