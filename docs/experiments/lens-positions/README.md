---
id: lens-positions
title: The letter readout can't see the answer during the reasoning
strand: C
also:
- T
status: done
kind: exploratory
verdict: instrument-limit
confidence: high
dates:
  start: '2026-11-03'
  end: '2026-11-03'
  approx: false
timing:
  wall: null
  gpu_hours: 0
  api_usd: 0
  note: Re-analysis of existing readouts.
machine: arena9-octavia (CPU)
n:
  items: as lens_main
  runs: as lens_main
numbers:
- label: Correct-letter margin before the answer position, all groups
  value: '|margin| ≤ 0.6 nats, intervals cover 0'
- label: Just before the reasoning first names the correct option
  value: +1.1 to +3.4 nats, the same in every group
- label: Probe at the end of the reasoning (</think>)
  value: near chance, leans towards the run's final answer
numbers_note: Margins are in nats over the other unchosen wrong letters; 0 means no difference.
reports:
- ../lens-main/report.md
figure: fig06_positions.svg
data:
- ../lens-main/data/lens_main.json
commits:
- 08b1e87
related:
- lens-main
- instrument-q1q3
---

# The letter readout can't see the answer during the reasoning

## Question

The lenses show no correct-letter signal during the reasoning of runs that followed the hint. Is that evidence that the knowledge is gone?

## Method

The main thinking-mode readouts, at four positions before the answer: the last prompt token; reasoning positions before the reasoning first names the correct option; the position just before that first mention; and the end of the reasoning (</think>). Re-read on 2026-11-03.

## Result

No. Before the answer position, the correct letter has no margin in any group, including runs with no hint that go on to answer correctly. So the readout can't show the answer there at all.

## Interpretation

The letter readout effectively asks 'is the answer letter the next token?'. During the reasoning it isn't, so the readout can't show the answer there, whether or not the model knows it. The lens nulls during reasoning are a limit of the readout. This corrects how the nulls were read overnight.

## Caveats

- Only option letters were tracked (rule 3 fixes the tracked tokens before a readout).

## Files

- Report: [../lens-main/report.md](../lens-main/report.md)
- Figure: [fig06_positions.svg](fig06_positions.svg)
- Derived numbers: [../lens-main/data/lens_main.json](../lens-main/data/lens_main.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_main.py`
- `scripts/lens/analyze_nothink.py`
