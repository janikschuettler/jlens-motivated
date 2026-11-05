---
id: probe-control
title: Do published eval-awareness probes read the model's belief or the prompt's surface? (test a: same prompt, different belief)
strand: T
status: running
kind: pre-registered
verdict: pending
confidence: moderate
dates:
  start: 2026-11-05T15:45
  approx: true
timing:
  wall: about 1 h 30 min so far (test a only)
  gpu_hours: 0.3
  api_usd: 0.01
  note: Probe build 1 min, probe readout of 900 runs 2.5 min, J-lens and logit-lens readout 11 min on arena9-octavia (L40S). Judge check of the label on 100 runs cost $0.01.
machine: arena9-octavia (L40S; readouts); arena9-cloud (selection, analysis)
n:
  items: 143 questions (167 known items, 143 with a mixed group)
  runs: 900 Qwen3-14B grader-bug runs in 225 identical-prompt groups (336 say "testing me"); 738 aware-vs-unaware pairs
numbers:
- label: Within-group AUROC, runs that will say "testing me" vs siblings that won't, read before the sentence (0.5 = chance). Nguyen probe / Hua direction
  value: 0.63 [0.59, 0.67] / 0.58 [0.53, 0.62]
- label: Same, J-lens eval-word score / logit-lens score
  value: 0.67 [0.62, 0.71] / 0.68 [0.64, 0.72]
- label: Same, window length alone (the confound) / word n-gram text classifier
  value: 0.66 [0.63, 0.68] / 0.83 [0.79, 0.86]
- label: Length-matched pairs (exploratory): Nguyen / Hua / J-lens / text n-grams
  value: 0.57 [0.53, 0.61] / 0.56 [0.51, 0.61] / 0.61 [0.56, 0.66] / 0.76 [0.72, 0.80]
- label: Nguyen probe build check on held-out benchmark vs user prompts (bar 0.70)
  value: 0.697 at layer 2 (FAILED by 0.003; reversed below 0.5 at layers 17 and up)
numbers_note: Within-group AUROC pools the 738 pairs; 95% intervals resample the 143 questions (2,000 resamples). Random-direction bands (100 random directions, same layer) reach 0.64 (Nguyen) and 0.67 (Hua).
reports:
- report.md
- ../../design_probe_control.md
figure: fig_probe_control_a.svg
data:
- data/pc_test_a.json
- data/judge_check.json
related:
- evalaware-ablation-14b
- evalaware-replication
- test-awareness-14b
---

# Probe control, test (a)

## Question
Neither published eval-awareness probe (Nguyen et al. 2025 Simple Contrastive; Hua et al. 2025 real-user vs evaluator) was tested on identical prompts. On Qwen3-14B grader-bug prompts, some runs later write "the user is testing me" and siblings do not. Do the probes, read during the reasoning before that sentence, tell them apart? Design: `docs/design_probe_control.md`; pre-registration PC in `docs/preregistration.md`.

## Method
See report.md. Reordered for the 21:00 deadline: test (a), the text baselines and the J-lens/logit-lens scores first; tests (b) and the extras were not run.

## Result
See report.md. In short: the probes are weak (0.58-0.63) and not above random directions; window length alone scores 0.66; word n-grams on the same text score 0.83; the eval-word J-lens and logit-lens scores reach 0.67-0.68 and stay at 0.61 after length matching.

## Interpretation
The prompts being identical, some pre-verbal information about the coming "testing me" sentence exists and is linearly readable (supervised ceiling 0.82), but it is also readable from the text so far. The published probes do not capture it.

## Caveats
Nguyen build check failed (0.697 < 0.70). Pairs where the sibling's reasoning ended early have shorter windows (231 of 738). The length-matched and residualised analyses are exploratory. Regex label checked on a small judge sample only. One model, one prompt family.

## Files
report.md, fig_probe_control_a.svg, data/pc_test_a.json, data/judge_check.json. Code: scripts/explore/pc_select.py, scripts/lens/pc_probes.py, pc_analyze.py, pc_embed.py, scripts/explore/pc_judge_check.py.
