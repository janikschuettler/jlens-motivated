---
id: impossiblebench-pilot
title: 'Gated pilot: multi-turn ImpossibleBench with Qwen3-14B and Qwen3-8B'
strand: N
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
machine: arena9-octavia (L40S), planned
n:
  items: null
  runs: null
numbers: []
numbers_note: null
reports:
- ../q4-scoping/report.md
figure: null
data: []
commits: []
related:
- q4-scoping
---

# Gated pilot: multi-turn ImpossibleBench with Qwen3-14B and Qwen3-8B

## Question

Do Qwen3-14B and Qwen3-8B, in thinking mode, special-case impossible tests on their own when given repeated failure feedback?

## Method

A multi-turn ImpossibleBench-LiveCodeBench loop (up to 10 submissions with feedback), thinking mode. Gated: stop if the cheating rate is near zero. Tokens to track in any readout are fixed in advance (rule 3).

## Result

Not run yet.

## Interpretation

Decides whether a version of the core question about rule-breaking the model chooses itself is feasible on models we can read with a lens.

## Files

- Report: [../q4-scoping/report.md](../q4-scoping/report.md)
