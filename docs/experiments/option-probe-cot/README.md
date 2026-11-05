---
id: option-probe-cot
title: Follow the option-text probe along the reasoning
strand: C
also: []
status: upcoming
kind: exploratory
verdict: pending
confidence: null
dates:
  start: null
  end: null
  approx: false
timing:
  wall: null
  gpu_hours: null
  api_usd: null
  note: null
machine: arena9-octavia, planned
n:
  items: null
  runs: null
numbers: []
numbers_note: null
reports:
- ../instrument-q1q3/report.md
figure: null
data: []
commits: []
related:
- instrument-q1q3
- forced-early-e5
---

# Follow the option-text probe along the reasoning

## Question

Can the direction that reads the answer at the option tokens also read it at reasoning positions, run by run?

## Method

Apply the option-position probe to the hidden states along the reasoning. Add more honest-error questions (2–8 of 10 correct) to test specificity, and use forced early answers as the behavioural baseline any internal score must beat.

## Result

Not run yet.

## Interpretation

A run-by-run readout of what the model knows is what the project still lacks.

## Files

- Report: [../instrument-q1q3/report.md](../instrument-q1q3/report.md)
