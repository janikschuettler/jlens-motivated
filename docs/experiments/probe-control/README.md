---
id: probe-control
title: 'Planned control: do published eval-awareness probes read the model''s belief or the prompt''s wording?'
strand: T
also: []
status: upcoming
kind: pre-registered
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
  note: 'Planned: about 4.7 GPU-hours on two L40S, about $2.5 judge API (cap $5).'
machine: arena9-octavia and arena9-derpy (L40S), planned
n:
  items: null
  runs: null
numbers: []
numbers_note: null
reports:
- ../../design_probe_control.md
figure: null
data: []
commits: []
related:
- evalaware-big-smoke
- evalaware-replication
- evalaware-ablation-14b
- evalaware-models-screen
---

# Planned control: do published eval-awareness probes read the model's belief or the prompt's wording?

## Question

Do the published eval-awareness probes (Nguyen et al. 2025, Hua et al. 2025), rebuilt on Qwen3-14B, detect whether the model thinks it is being tested, or only eval-flavoured wording and benchmark format?

## Method

Rebuild both probes from their public code and data on Qwen3-14B; they must separate benchmark from user prompts (AUROC at least 0.70) first. Then: (a) on 225 groups of identical grader-bug prompts where some runs later say 'you're testing me', score reasoning before that sentence; (b) matched prompt pairs that differ only in eval wording or format, with the model's stated belief measured; (c) bag-of-words and small-embedding text baselines on the same splits; (d) the J-lens and logit-lens eval-word scores on the same data.

## Result

Not run yet. Design for the user's review: docs/design_probe_control.md.

## Interpretation

If the probes fail the same controls as the J-lens, our J-lens null is about the phenomenon or the model, not the lens; if a probe predicts the sentence within a prompt beyond the text, it is a lead the J-lens missed.

## Files

- Report: [../../design_probe_control.md](../../design_probe_control.md)
