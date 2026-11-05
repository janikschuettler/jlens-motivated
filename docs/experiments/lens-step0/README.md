---
id: lens-step0
title: First lens readout (M&B's setting, thinking off)
strand: C
also: []
status: done
kind: exploratory
verdict: inconclusive
confidence: low
dates:
  start: 2026-11-02T22:20
  end: 2026-11-02T22:35
  approx: true
timing:
  wall: null
  gpu_hours: 0.05
  api_usd: 0
  note: 47 s of generation plus 2.4 min of readout.
machine: arena9-mayor (RTX 4090)
n:
  items: 32 (24 known)
  runs: '480'
numbers:
- label: Runs that followed / resisted
  value: 109 / 103 (on 22 / 21 items)
- label: 'Prominent unchosen letter, layers 24–32: margin over the other wrong letters'
  value: 1–3 nats above the other wrong letters
numbers_note: null
reports:
- report.md
figure: null
data:
- data/lens_step0.json
commits:
- f2781bf
related:
- lens-main
---

# First lens readout (M&B's setting, thinking off)

## Question

In a first small readout, do runs that followed the hint still hold the correct letter inside?

## Method

M&B's setup re-sampled: their format, thinking off, their decoding; 32 questions; 3 unhinted samples plus a hint on each option × 3 (480 runs). J-lens and logit lens at layers 14–32. 24 questions known (3 of 3 correct); 109 runs followed a wrong hint and 103 resisted.

## Result

Option letters differ only at the answer position. There, any unchosen letter that the prompt made prominent stays 1–3 nats above the other wrong letters: the correct letter in runs that followed, and the hinted letter in runs that resisted.

## Interpretation

The first sign that a correct-letter margin at the answer position may just be salience: the letter is prominent, not necessarily known. The main lens readout then tested this with a control.

## Caveats

- Small; 'known' is 3 of 3.
- Thinking off only.

## Files

- Report: [report.md](report.md)
- Derived numbers: [data/lens_step0.json](data/lens_step0.json)
