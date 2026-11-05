---
id: design-survey
title: 'Design survey: paradigms, datasets, grader-bug variants'
strand: N
also:
- A
status: done
kind: exploratory
verdict: context
confidence: null
dates:
  start: 2026-11-02T21:35
  end: 2026-11-02T21:48
  approx: true
timing:
  wall: null
  gpu_hours: 0
  api_usd: 0
  note: Reading only.
machine: arena9-cloud (no GPU)
n:
  items: null
  runs: null
numbers: []
numbers_note: null
reports:
- report.md
- normative_datasets.md
figure: null
data: []
commits:
- 3397f78
related:
- instructed-e1
- grader-bug-e3
- moral-pilot
---

# Design survey: paradigms, datasets, grader-bug variants

## Question

What is a good design to pin down 'arguing against its own knowledge', and which levers make models give up a correct answer?

## Method

Web research only: experimental paradigms, multiple-choice datasets with moral or normative questions (checked on the Hugging Face Hub), variants of the grader-bug note, and 8 ranked experiments for the night.

## Result

Our near-zero switch rate is what the literature predicts for a calibrated nudge that reports a third person's choice. The reliable levers (pressure, instruction, prefilled text) mostly give open deference.

## Interpretation

Shaped the second wave of experiments (E1–E8 in the reports) and the moral pilot.

## Caveats

- Papers dated 2026 were read at abstract level unless noted.

## Files

- Report: [report.md](report.md)
- Report: [normative_datasets.md](normative_datasets.md)
