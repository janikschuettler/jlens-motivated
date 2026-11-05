---
id: firming-f1-f2
title: 'Firming the J-lens nulls (equivalence tests) and the planted-claim positive (difficulty control)'
strand: C
also: []
status: done
kind: pre-registered
verdict: inconclusive
confidence: moderate
dates:
  start: '2026-11-05'
  end: '2026-11-05'
  approx: false
timing:
  wall: about 1.5 h
  gpu_hours: 0
  api_usd: 0
  note: 'CPU only, on stored per-run scores and readouts. Pre-registration: docs/preregistration.md, F1-F3 (written 2026-11-04 21:10 GMT; not blind, the point estimates were already known).'
machine: arena9-cloud (CPU); one T2 recomputation from raw readouts on arena9-octavia (CPU)
n:
  items: 74
  runs: 389
numbers:
- label: 'F2: planted-claim runs, J-lens score at the answer, names correct vs not (AUROC, as before the control)'
  value: 0.71 [0.57, 0.84]
- label: 'F2: same, comparing only runs on questions of equal difficulty (J-lens / logit lens)'
  value: 0.73 [0.59, 0.85] / 0.60 [0.43, 0.79]
- label: 'F2: same, only pairs from the same question (11 questions; J-lens / logit lens)'
  value: 0.86 [0.72, 0.95] / 0.72 [0.59, 0.87]
- label: 'F2: question difficulty alone predicts the outcome about as well (AUROC; unhinted misses out of 20)'
  value: 0.72 [0.62, 0.82]
- label: 'F2: J-lens effect after adjusting for difficulty (log-odds per SD of the score)'
  value: 0.57 [0.12, 1.43]
- label: 'F1: J-lens vs chance, cases equivalent within +-0.10 at every position tested'
  value: 5 of 17 case analyses (silent vs open switches, GPT-OSS-20B and Qwen3-8B no-think grader bug, T1b pooled and its octavia half)
numbers_note: 'AUROC: chance that a random positive run has a higher score than a random negative run (0.5 = chance, 1 = perfect). Intervals resample questions. Full table in report.md.'
interpretation: 'The planted-claim positive survives the difficulty control as a run-level signal, but it is a small, single-case result. The main J-lens nulls are only partly equivalent to chance: several are too wide to rule out a moderate effect.'
reports:
- report.md
figure: null
data:
- data/f1_all.json
- data/f1_cases.json
- data/f2.json
- data/f2_sensitivity_drop_uninformative_control.json
commits: []
related:
- replicate-leads
- jlens-vs-competitors
- prereg-t1b
- prereg-t2
- hack-organism
- evalaware-big-smoke
---

# Firming the J-lens nulls and the planted-claim positive

See [report.md](report.md) for the results table and plain interpretation. Pre-registration: `docs/preregistration.md`, section F1-F3 (F3, the code review, is not part of this experiment).

Code: `scripts/lens/firming_f1_cases.py`, `scripts/lens/firming_f1_scores.py`, `scripts/lens/firming_f1_ea.py`, `scripts/lens/firming_f2.py`; T2 per-run scores from `scripts/lens/firming_recompute.py t2`.
