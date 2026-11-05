---
id: judge-validation
title: Check the labels with the LLM judge
strand: T
also:
- B
status: upcoming
kind: tooling
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
machine: arena9-cloud (API)
n:
  items: null
  runs: null
numbers: []
numbers_note: null
reports: []
figure: null
data: []
commits: []
related:
- judge-offline
- silent-scan-thinking
---

# Check the labels with the LLM judge

## Question

Do the keyword labels (mentions the hint, silent or open switch, stance) hold up against a checked judge?

## Method

Apply the four pending fixes. Run the judge on the reference set and on a sample of switches from each condition, and compare with the keyword labels. Also settle a disagreement in the answer extractor found in the subtle-cue runs: it ranks the pattern "answer is X" above \boxed{X} (changing the shared extractor is the user's call).

## Result

Not run yet.

## Interpretation

Needed before any claim rests on counts of silent or open switches, or on stance labels.

## Files

No report or data files.
