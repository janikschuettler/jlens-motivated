---
id: silent-scan-thinking
title: How often are thinking-mode switches silent?
strand: B
also: []
status: done
kind: exploratory
verdict: against
confidence: high
dates:
  start: 2026-11-03T01:43
  end: 2026-11-03T01:43
  approx: false
timing:
  wall: null
  gpu_hours: 0
  api_usd: 0
  note: CPU scan of existing runs.
machine: arena9-octavia (CPU)
n:
  items: known items of the factorial and wording arms
  runs: 1,231 switches
numbers:
- label: Silent switches, factual questions, thinking mode
  value: 2 / 1,231 (0.16% [0.02, 0.59])
- label: Case 1
  value: argues for the hinted D on its merits, never refers to the hint
- label: Case 2
  value: works out the correct content, then writes the hinted letter
numbers_note: null
reports:
- ../wording-dissection/report.md
figure: ../subtle-cues/fig04_silent_share.svg
data:
- ../wording-dissection/data/wave2_part2.json
commits:
- af18607
related:
- nothink-scale
- moral-hunt
- judge-validation
---

# How often are thinking-mode switches silent?

## Question

In thinking mode, how often does the model switch to the hinted option without its reasoning mentioning the hint? Those are the cases where a lens could add something.

## Method

Every switch to a wrong hint on known questions in the factorial and the wording dissection, scanned with broad and hint-specific keyword lists. All candidates were read by eye.

## Result

Almost never: 2 of 1,231 switches are silent (0.16% [0.02, 0.59]). Both are genuine on reading.

## Interpretation

In thinking mode, someone reading the reasoning would catch almost every case of arguing against its knowledge under an explicit hint. So the detection question has to move away from the plan's thinking-mode silent switches.

## Caveats

- Keyword lists, not a validated judge.
- Silent switches cluster on a few questions.

## Files

- Report: [../wording-dissection/report.md](../wording-dissection/report.md)
- Figure: [../subtle-cues/fig04_silent_share.svg](../subtle-cues/fig04_silent_share.svg)
- Derived numbers: [../wording-dissection/data/wave2_part2.json](../wording-dissection/data/wave2_part2.json)
