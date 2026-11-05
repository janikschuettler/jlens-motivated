---
id: subtle-cues
title: 'Subtle cues: metadata, grader code, marked examples'
strand: A
also:
- B
status: done
kind: exploratory
verdict: against
confidence: moderate
dates:
  start: 2026-11-03T11:06
  end: 2026-11-03T12:08
  approx: false
timing:
  wall: 62 min
  gpu_hours: 1.03
  api_usd: 0
  note: 'Approximate: the GPU ran for the whole 62 min; wave 1 alone took 27 min at 1,357 tokens/s.'
machine: arena9-octavia (L40S)
n:
  items: 160 known
  runs: 3,120
numbers:
- label: 'Metadata tag: extra runs ending on B'
  value: +7.9 pp [4.4, 11.9]
- label: Grader code
  value: +5.5 pp [2.9, 8.2]
- label: Examples with ■ on the right answer
  value: +4.4 pp [2.1, 7.1]
- label: Examples whose answer is always B
  value: +0.6 pp [−0.2, 1.7]
- label: 'Silent switches on reading: metadata and grader code / ■ examples / always-B examples'
  value: 3 of 66 / 11 of 23 / 4 of 5
numbers_note: pp = percentage points of extra runs ending on B, compared with the same questions without a cue.
reports:
- report.md
figure: fig04_silent_share.svg
data:
- data/subtle_cues.json
commits:
- bd40825
related:
- silent-scan-thinking
- silent-pool
---

# Subtle cues: metadata, grader code, marked examples

## Question

Do cues that aren't a sentence about the answer move the model on known questions, and are those switches silent?

## Method

Thinking mode, 160 known questions × 3 samples per cue (the marker alone: 80 questions). The cues:
- a metadata tag that gives the answer as B;
- a snippet of grader code that checks answer == "B";
- four solved example questions with a ■ on each right answer, and ■ on the target's B;
- four examples whose answer is always B;
- the ■ on B alone.
Controls: no hint, and four examples with no pattern. We compared the share of runs ending on B with the same questions without a cue (0.4%). Every switch was read for references to the cue.

## Result

They add little: 0.6 to 7.9 percentage points over no cue. Switches under the ■-marked examples are often silent (11 of 23), but they happen on weakly known questions.

## Interpretation

Non-verbal cues tip fragile questions without being named, but they don't override solid knowledge. The ■-marked examples, compared against the unpatterned examples, are a possible source of silent cases.

## Caveats

- Silent ■ switches cluster on questions that also flip under other cues or with no cue at all.
- One answer-extractor mislabel found (the "answer is" pattern is read before \boxed{}).

## Files

- Report: [report.md](report.md)
- Figure: [fig04_silent_share.svg](fig04_silent_share.svg)
- Derived numbers: [data/subtle_cues.json](data/subtle_cues.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `scripts/explore/run_subtle.py`
- `scripts/explore/analyze_subtle.py`
