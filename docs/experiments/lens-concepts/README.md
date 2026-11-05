---
id: lens-concepts
title: Concept words in the J-lens's top-25 lists
strand: C
also: []
status: done
kind: exploratory
verdict: against
confidence: low
dates:
  start: 2026-11-03T09:30
  end: 2026-11-03T10:33
  approx: true
timing:
  wall: about 1 h
  gpu_hours: 0
  api_usd: 0
  note: CPU only, existing readouts.
machine: arena9-octavia (CPU)
n:
  items: known items
  runs: 3,796
numbers:
- label: '''Wrong'' words, J-lens layers 28–32, last quarter of the reasoning: followed (reasoning shows a conflict) / resisted / no hint'
  value: 16% [14, 19] / 20% [19, 21] / 7%
- label: 'Lead: hint words early in the reasoning (layers 14–20), followed vs resisted'
  value: 4.9% vs 3.0%
numbers_note: Share of top-25 lists that contain at least one word of the set.
reports:
- report.md
figure: null
data:
- data/lens_concepts.json
commits:
- bd40825
related:
- lens-main
---

# Concept words in the J-lens's top-25 lists

## Question

Beyond option letters, do concepts such as the hint, 'wrong', cheating or 'correct' show up differently in runs that followed the hint?

## Method

For English and Chinese word sets, the share of stored top-25 token lists that contain them, by group and position. Thinking mode, 3,796 runs on known questions, intervals over questions, and a check that the word wasn't just echoed from the text before it.

## Result

No. No concept separates runs that followed the hint from runs that resisted the same hints.

## Interpretation

We found no 'this is wrong, but I'm saying it' signal in the stored top-25 lists. They mostly show punctuation, markup, filler words and topic words.

## Caveats

- A top-25 list is a narrow window.
- Stance groups come from a keyword rule.

## Files

- Report: [report.md](report.md)
- Derived numbers: [data/lens_concepts.json](data/lens_concepts.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/lens/analyze_concepts.py`
