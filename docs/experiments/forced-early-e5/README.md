---
id: forced-early-e5
title: When is the wrong answer decided? Forced early answers (E5)
strand: B
also:
- C
status: done
kind: exploratory
verdict: context
confidence: moderate
dates:
  start: 2026-11-02T22:38
  end: 2026-11-03T02:20
  approx: true
timing:
  wall: null
  gpu_hours: 0.16
  api_usd: 0
  note: 'Three batches: 32 s (mayor), about 3 min (mayor), about 6 min with the later follow-up questions (E7) (octavia).'
machine: arena9-mayor and arena9-octavia
n:
  items: 40–50 known per arm
  runs: 962 + 629 traces; 17,501 forks
numbers:
- label: '''…the answer is X'' hints, runs that followed, before any reasoning (f = 0)'
  value: P(B) 0.82, flat
- label: Same hints, runs that resisted, f = 0
  value: P(B) 0.43, then reason towards correct
- label: '''…chose X'' hints, runs that followed, f = 0'
  value: P(B) 0.36, drifting towards B
- label: Thinking off, M&B's setting, followed / resisted, f = 0
  value: P(hinted) 0.91 / 0.79
numbers_note: P(B) is the probability of the hinted wrong letter when the model is forced to answer; f is the share of the reasoning kept before the cut (0 = none, 1 = all).
reports:
- ../wording-dissection/report.md
- ../instructed-e1/report.md
figure: fig09_early_answer.svg
data:
- ../wording-dissection/data/wave2_part2.json
- ../instructed-e1/data/wave2_part1.json
commits:
- af18607
related:
- reelicitation-e7
- q6-point-of-no-return
---

# When is the wrong answer decided? Forced early answers (E5)

## Question

When a run ends on B, does the model reason its way there, or is B there from the start?

## Method

Cut each saved reasoning trace at a fraction f of its length (at the nearest line break), append '</think>\n\nAnswer:', and read the model's next-token probabilities for the option letters, rescaled to sum to 1. 962 thinking-mode traces, 10,582 cuts (the letters get 92–100% of the next-token probability). Also 629 thinking-off traces in M&B's setting (6,919 cuts).

## Result

Under hints that state the answer, B is there from the start. Runs that end on B already give it probability 0.82 before any reasoning, and it stays flat.

## Interpretation

Many switches look like a decision taken on reading the prompt and argued for afterwards: the reasoning presents reasons for a choice already made. This is the behavioural baseline any internal score must beat.

## Caveats

- Forcing an early answer is unlike normal generation: runs that resisted still give B probability 0.16 at f = 1.

## Files

- Report: [../wording-dissection/report.md](../wording-dissection/report.md)
- Report: [../instructed-e1/report.md](../instructed-e1/report.md)
- Figure: [fig09_early_answer.svg](fig09_early_answer.svg)
- Derived numbers: [../wording-dissection/data/wave2_part2.json](../wording-dissection/data/wave2_part2.json)
- Derived numbers: [../instructed-e1/data/wave2_part1.json](../instructed-e1/data/wave2_part1.json)

Code, configs and outputs (from the old experiments index; data/ paths are on the GPU machines, not in git):

- `src/jmr/explore/w2*.py`
- `configs/explore/w2_instructed.yaml`
- `configs/explore/w2_grader.yaml`
