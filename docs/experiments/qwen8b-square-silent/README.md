---
id: qwen8b-square-silent
title: 'Qwen3-8B with thinking off: does it silently follow the black-square examples cue, and does the J-lens show the correct answer?'
strand: B
also:
- A
status: done
kind: preregistered
verdict: mixed
confidence: preliminary
dates:
  start: 2026-11-05T19:20
  end: 2026-11-05T19:35
  approx: true
timing:
  wall: about 30 min
  gpu_hours: 0.1
  api_usd: 0
  note: 'Qwen3-8B fits one GPU; generation 3 min, lens readout of 590 runs about 4 min. arena9-octavia.'
machine: arena9-octavia (L40S)
n:
  items: 145 known (at least 9/10 correct, thinking off)
  runs: 1595 generated, 590 read by the lens
numbers:
- label: 'Black-square examples switch rate, thinking off (Gemma-3-27B: 48%)'
  value: 8% [5, 12]
- label: 'Silent switches by keyword rule: black-square examples / answer hint (Gemma: 94 of 117 / 30 of 110)'
  value: 31 of 58 / 67 of 232
- label: 'Q1: correct letter above the other wrong letters in silent black-square switches, J-lens mean log-prob margin (pass: lower bound > 0)'
  value: 0.94 [-0.19, 2.06]
- label: 'Same, logit lens'
  value: 0.81 [0.00, 1.65]
- label: 'Q2: silent vs open black-square switches, AUROC, J-lens / logit lens / reply length (0.5 = chance)'
  value: 0.45 [0.25, 0.65] / 0.65 [0.44, 0.82] / 0.28 [0.11, 0.47]
numbers_note: 'Brackets are 95% item-bootstrap CIs. Items are the resampling unit. "Silent" is the keyword rule.'
reports:
- report.md
data:
- data/behaviour.json
- data/lens_results.json
related:
- gemma27-silent
- subtle-cues
- nothink-scale
---

# Qwen3-8B (thinking off): silent switches under the black-square cue

## Question

Gemma-3-27B (no thinking) follows examples with a black square on one option 48% of the time, mostly without mentioning it, and its J-lens shows the correct answer only in the last blocks (`gemma27-silent`). Does Qwen3-8B with thinking switched off do the same?

## Method

Pre-registration Q-sq in `docs/preregistration.md` (written after the behaviour was labelled, before any lens readout). 145 items Qwen3-8B answers correctly at least 9 of 10 times with thinking off (earlier non-thinking screen, different prompt format and decoding than here). Same prompt builder as `subtle-cues` and `gemma27-silent` (`scripts/explore/run_subtle.py`, `configs/explore/qsq.yaml`): four marked examples, then the question with the black square on the wrong option B. Thinking off, T 0.7, top-p 0.8, top-k 20, 1,536 tokens max. Arms: no hint (3 samples), "Hint: The answer is B." (3), black-square examples (5). Lens: Neuronpedia Qwen3-8B J-lens, blocks 14 to 32 step 2, logit lens next to it, one sequence per pass.

## Result

See `report.md`. Short version: the cue works much less on Qwen3-8B than on Gemma (8% against 48%), about half of the 58 switches are silent (31 on 15 items), and the lens test on those 31 is inconclusive: the J-lens margin is positive but its interval includes 0, and silent and open switches are not separated.

## Interpretation

The symmetric claim holds only weakly. Qwen with thinking off can switch silently under the black-square cue, but rarely, so the test has few runs (15 items). Gemma's pattern (correct answer seen mainly in the last blocks, silent vs open not separable) is not contradicted: the Qwen depth split points the same way (last blocks 1.77, middle 0.38) but both intervals include 0. Preliminary.

## Caveats

15 items and 31 runs make the intervals wide. Known items use a 9/10 filter from a different prompt format. The pooled (both arms) analysis in the pre-registration was not run; the answer-hint arm alone is exploratory.
