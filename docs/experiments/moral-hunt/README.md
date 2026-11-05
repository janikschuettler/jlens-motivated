---
id: moral-hunt
title: Are switches on moral questions silent more often?
strand: B
also:
- A
- C
status: done
kind: exploratory
verdict: against
confidence: high
dates:
  start: 2026-11-03T04:27
  end: 2026-11-03T05:55
  approx: true
timing:
  wall: null
  gpu_hours: 0.93
  api_usd: 0
  note: Nudge arms 19 + 37 GPU-min; the screen was not timed. Start is when mayor became free.
machine: arena9-mayor (RTX 4090); readouts on arena9-octavia
n:
  items: 678 known (+ 32 Redux moral)
  runs: 6,102 hinted + 3,235 screen
numbers:
- label: Switch rate, MoralChoice
  value: 79% [77, 81]
- label: Silent switches, all moral questions
  value: 0 / 5,019 (95% upper bound 0.07%)
- label: Factual questions, for comparison
  value: 2 / 1,231
- label: 'Of the 14 read in full: state their own contrary view, then defer to the hint'
  value: 11 of 14
- label: 'Lens, runs that followed: correct-letter margin at the answer (J-lens, layers 30–32)'
  value: +1.3 to +1.6 nats, vs salience control +2.0 to +2.5
numbers_note: 'The lens row: the margin is how far the correct letter sits above the other unchosen letters, in nats. The salience control is the same margin for the unchosen hinted letter in runs that resisted. Here the control is larger, so the margin is no sign of hidden knowledge.'
reports:
- report.md
figure: ../subtle-cues/fig04_silent_share.svg
data:
- data/moral_hunt.json
commits:
- bd40825
related:
- moral-pilot
- silent-scan-thinking
---

# Are switches on moral questions silent more often?

## Question

In thinking mode, are switches on moral questions silent more often than on factual ones?

## Method

Thinking mode, M&B's hint in our format, on 678 known low-ambiguity MoralChoice scenarios (screened 5 of 5 correct: 647 new plus 39 from the pilot) × 9 samples, plus the pilot's Redux moral runs. Keyword and broad scans; the 14 switches with the fewest references to the hint were read in full. Lens readout of the Redux moral runs.

## Result

No: 0 of 5,019 switches on moral questions are silent (95% upper bound 0.07%), at a 79% switch rate.

## Interpretation

In thinking mode, moral questions don't supply silent switches for a test of the reasoning text against the lens. Every switch names the hint, usually in the first paragraph, and then the model builds a case for the hinted option.

## Caveats

- Beyond MMLU-Redux.
- 'Known' is 5 of 5 here, not 9 of 10.
- MoralChoice has 2 options.

## Files

- Report: [report.md](report.md)
- Figure: [../subtle-cues/fig04_silent_share.svg](../subtle-cues/fig04_silent_share.svg)
- Derived numbers: [data/moral_hunt.json](data/moral_hunt.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `configs/explore/moral_hunt_*.yaml`
