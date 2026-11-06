---
id: steering
title: 'Steering: do the directions the J-lens reads drive the model''s answer?'
strand: C
also:
- T
status: done
kind: pre-registered
verdict: against
confidence: moderate
dates:
  start: 2026-11-04T00:05
  end: 2026-11-04T05:00
  approx: true
timing:
  wall: about 5 h (including about 3 h waiting for the GPU)
  gpu_hours: 2.1
  api_usd: 0
  note: 'GPU: pilot 9 min, answer-position part 9 min including a restart, generation 105 min (01:00–01:16 and 02:53–04:38). CPU smoke tests and analysis on octavia.'
machine: arena9-octavia (L40S)
n:
  items: 40
  runs: 432 saved runs (part 1); 920 generated (part 2) + 80 pilot
numbers:
- label: 'Answer position, wrong answers switched to correct (mean over 7 strengths): J-lens / logit lens / probe / random'
  value: 47% [45, 49] / 27% [24, 31] / 59% [58, 60] / 2% [1, 3]
- label: 'Answer position: J-lens minus logit lens; J-lens minus probe'
  value: +20 pp [+16, +24]; −12 pp [−13, −10]
- label: 'Answer position, correct unhinted answers pushed away: J-lens / logit lens / probe / random'
  value: 39% / 33% / 51% / 1%
- label: 'Push strength at which half the wrong answers switch: J-lens / logit lens / probe'
  value: 0.49 / 0.57 / 0.27
- label: 'Pushing toward the other wrong letter instead of correct: difference, J-lens'
  value: +0.5 pp [−1.9, +2.7] (correct is no easier to reach)
- label: 'Generation, strength 0.2, final answer correct (forced read at the cap): no push / J-lens / logit / probe / random'
  value: 33% / 53% / 42% / 82% / 37%
- label: Same, J-lens minus no push; J-lens minus probe
  value: +20 pp [+10, +30]; −29 pp [−40, −19]
- label: 'Generation, runs that hit the cap counted as no answer: correct, no push / J-lens / probe'
  value: 24% / 25% / 42%
- label: 'Generation, share of runs that hit the 2,048-token cap: no push / J-lens / logit / probe / random'
  value: 34% / 43% / 43% / 47% / 37%
- label: 'Generation, strength 0.8: letter pushes with degenerate repetition'
  value: 79–97% (random 4%)
numbers_note: 'Answer position: share of saved wrong-answer runs whose answer letter switches to the correct one when the push is added at that single position (432 runs, 39–40 questions). Generation: share of fresh runs whose final answer is correct, 120 runs and 40 questions per cell, wrong-answer groups pooled. pp = percentage points, paired by question. Intervals: 95% bootstrap over questions.'
reports:
- report.md
figure: fig13_steering.svg
data:
- data/steering.json
commits:
- 321715f
- 8257e49
- a916043
- 38a778f
- b1e4914
related:
- lens-main
- instrument-q1q3
- jlens-vs-competitors
- planted-claim-c3
- instructed-e1
- wording-dissection
---

# Steering: do the directions the J-lens reads drive the model's answer?

## Question

If we push Qwen3-8B's hidden state along the J-lens direction of the correct option's letter, does its answer change more than with the logit-lens direction, a probe direction or a random direction of the same size?

## Method

40 known questions, thinking mode, HF transformers. Directions at block 24 for each letter, centred over the four letters and of equal length: J-lens (J₂₄ᵀ times the letter's LM-head row and the final norm's weights), logit lens (the LM-head row), probe (difference of mean hidden states of unhinted runs answering each letter, on other questions) and random. Push = strength × the typical hidden-state length (254). Part 1: on 432 saved runs (told to argue for B, "Hint: The answer is B.", a planted "The answer is (B).", and correct unhinted runs), push at the answer position only, 7 strengths; count answers that switch toward the correct letter (or away from it in unhinted runs). Part 2: fresh runs with the push at every generated token, strength 0.2 (all 40 questions) and 0.8 (15 questions), one sample per question and cell, cap 2,048 tokens; final answer with and without a forced read for runs that hit the cap. Pre-registered S1-S3.

## Result

Partly: the J-lens direction moves the answer, more than the logit-lens direction and far more than random, but a simple probe direction moves it more. At the answer position, a push toward the correct letter switches wrong answers to it in 47% of cases (averaged over strengths) along the J-lens direction, 27% along the logit lens, 59% along the probe and 2% along random. A small push (0.2) throughout the reasoning raises correct final answers from 33% to 53% (J-lens), 42% (logit lens), 82% (probe) and 37% (random), counting forced answers for runs that hit the cap. Counting those runs as no answer, only the probe push helps (24% to 42%); the J-lens push mainly makes runs longer and lowers answers of B (41% to 31%). At strength 0.8 every letter push turns the reasoning into letter spam.

## Interpretation

The J-lens letter reading is causal at the moment of answering, so it is a valid readout of the answer the model is about to give, and better than the logit lens. But a probe learned from the model's own answer states is a stronger handle, at the answer position and during the reasoning. For the core question this favours trained probes over the J-lens as the readout of whether the correct answer is still represented.

## Re-run with a longer token cap (2026-11-06; pre-registered in [preregistration.md](preregistration.md))

Question: does the earlier generation result survive when fewer runs are cut off? Same 40 questions, same directions, push strength 0.2 at block 24 at every generated token, two samples per question and cell (240 runs per pooled cell), but cap 4,096 new tokens instead of 2,048. The intended 8,192 did not fit the night's time box (memory allows about 14 rows per batch at that length), so this is a partial fix; 12% of unsteered wrong-answer runs still hit the cap (34% before). Only the three wrong-answer groups were run (no unhinted runs); the answer-position part was not re-run because it does not depend on the cap.

Wrong-answer runs (told to argue for B, hint, planted claim, pooled), share ending on the correct letter:

| Push | natural finish only (cut-off = no answer) | forced answer at the cap |
| --- | --- | --- |
| none | 28% | 33% |
| J-lens | 41% | 48% |
| logit lens | 36% | 40% |
| probe | 60% | 73% |
| random | 29% | 33% |

Paired by question (percentage points, 95% interval over 40 questions):

| Contrast | natural finish only | forced answer |
| --- | --- | --- |
| J-lens minus logit lens (primary, R1 / R2) | +5.4 [-0.8, +11.3] | +7.9 [+2.5, +13.3] |
| J-lens minus no push | +13.3 [+7.1, +20.0] | +15.8 [+10.0, +22.5] |
| logit lens minus no push | +7.9 [+2.5, +12.9] | +7.9 [+2.5, +13.3] |
| J-lens minus random | +12.5 [+5.4, +19.2] | +15.4 [+8.3, +22.1] |
| J-lens minus probe | -18.8 [-27.5, -10.0] | -24.2 [-32.9, -16.7] |

How to read: each cell is the share of runs (each question counts once). Sample 0 alone gave J-lens minus logit lens +1.7 [-9, +13] (natural) and +5.8 [-3, +16] (forced).

Pre-registered reading: the primary measure (natural finish) does not show the J-lens ahead of the logit lens (lower bound just below 0); the forced-answer measure does. Prediction partly met: the J-lens push helps clearly versus no push and versus random, the probe push is far stronger, but the J-lens advantage over the logit lens is smaller than the answer-position result (47% vs 27%) and not secure. With fewer cut-off runs the J-lens gain over no push stays (about +13 to +16 points), so the earlier gain was not a cap artefact; the claim that it "mainly makes runs longer" is not supported here (share hitting the cap is unchanged, 12% vs 12%). Wrong-answer B drops 60% to 47% (J-lens), 53% (logit), 25% (probe). Per group, forced correct (none / J / logit / probe / random): argue 12 / 33 / 20 / 59 / 11; hint 21 / 34 / 34 / 69 / 23; planted 64 / 79 / 68 / 90 / 65.

Caveats: cap 4,096, not 8,192; no extra questions (no saved prompts for them, and generation time was the limit); same one layer and strength; the interval for the primary just includes 0, so "J-lens better than logit lens during reasoning" is suggestive only. Figure: [fig_rerun_cap4096.svg](fig_rerun_cap4096.svg). Numbers: [data/steering_rerun_cap4096.json](data/steering_rerun_cap4096.json). Raw runs (octavia, gitignored): `data/steer/gen4k/`.

## Caveats

- Deviation from the plan: token cap 2,048 instead of 8,192 (the plan said 3,072), chosen for time; needs the user's acceptance. It cuts off 34–47% of the wrong-answer runs; the same questions' original runs cut at 2,048 match the unpushed cell.
- Second deviation: strength 0.2 ran before 0.8 (the plan had 0.8 first), and 0.8 ran on 15 questions only. At 0.8 all letter pushes break the text, so 0.2 is the informative comparison.
- One sample per question and cell in generation; one layer (24), where the letter first becomes readable at the answer position.
- Equal length is one way to compare directions; the J-lens direction is the lens's average gradient, so part of its effect is expected by construction.
- Reasoning labels are keyword heuristics, checked by reading 18 runs.

## Files

- Report: [report.md](report.md)
- Figure: [fig13_steering.svg](fig13_steering.svg)
- Derived numbers: [data/steering.json](data/steering.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/steer_answer.py`
